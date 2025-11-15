# fed/server.py
import time
import queue
import numpy as np
import torch
from collections import OrderedDict, defaultdict

from utils.ipc import unpack_arrays
from agents.hetero import HeteroManager


class FedServer:
    def __init__(self, cfg, queues):
        """
        queues 约定:
          - queues["to_srv"]        : 所有客户端往服务器发消息的队列
          - queues[f"to_cli_{cid}"] : 服务器往第 cid 个客户端发消息的队列
        """
        self.cfg = cfg
        self.queues = queues
        self.N = int(cfg["data"]["n_clients"])       # 客户端总数
        # 每多少轮请客户端“合成/上传”一次（可在 cfg["server"]["synth_every"] 配置）
        self.K = int(cfg.get("server", {}).get("synth_every", 5))

        # FedAvg 的全局 score 权重（CPU tensors）
        self.global_state = None
        self.x_dim = None
        self.n_classes = None

        # 中央异构代理 (延迟初始化)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.central_hetero = None
        # 按 schema 汇聚的合成样本池（每轮清空）
        self._synth_pool = defaultdict(list)

        # 服务器侧的异构近线训练配置
        self._srv_hetero_epochs = int(cfg.get("gen", {}).get("hetero_epochs_srv", 1))
        self._srv_batch_size = int(cfg["train"]["batch_size"])
        # 每个 schema 每轮最多保留多少条样本，避免内存暴涨
        self._srv_pool_cap = int(cfg.get("comm", {}).get("srv_pool_cap_per_schema", 20000))
        # （可选）是否把中央异构代理的权重也随广播下发
        self._broadcast_hetero = bool(cfg.get("comm", {}).get("broadcast_hetero", False))

    @staticmethod
    def _fedavg(collected_states):
        """
        collected_states: List[dict(name->CPU tensor)]
        return: OrderedDict(name->CPU tensor)
        """
        agg = OrderedDict()
        keys = collected_states[0].keys()
        for k in keys:
            s = None
            for sd in collected_states:
                t = sd[k]
                s = (t.clone() if s is None else s.add_(t))
            agg[k] = s.div_(len(collected_states))
        return agg

    def _maybe_init_central(self, meta):
        """
        根据客户端附带的 meta（x_dim, n_classes）懒初始化中央异构代理。
        """
        if self.central_hetero is not None:
            return
        self.x_dim = int(meta["x_dim"])
        self.n_classes = int(meta["n_classes"])
        # 与客户端一致的超参：score_ms 的 tlist_len=3
        lr_agent = float(self.cfg["train"]["lr_agent"])
        self.central_hetero = HeteroManager(
            device=self.device,
            n_classes=self.n_classes,
            x_dim=self.x_dim,
            tlist_len=3,
            lr=lr_agent,
            class_weights=None  # 服务器端先不做加权；如需可在此写入
        )
        print(f"[Server] central_hetero initialized: x_dim={self.x_dim}, n_classes={self.n_classes}", flush=True)
    # === 在类 FedServer 里新增：===

    def _metrics_from_cm(self, cm_np, loss_sum, n_sum):
        import numpy as np
        cm = cm_np.astype(np.int64)
        total = cm.sum()
        acc = np.trace(cm) / max(1, total)

        # 宏平均 precision/recall/f1（按混淆矩阵从零计算，避免平均各 client 的宏值造成偏差）
        eps = 1e-9
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        prec = np.mean(tp / (tp + fp + eps))
        reca = np.mean(tp / (tp + fn + eps))
        f1   = 0.0 if (prec + reca) < eps else (2 * prec * reca / (prec + reca))
        loss = float(loss_sum) / max(1, int(n_sum))
        return loss, float(acc), float(prec), float(reca), float(f1)

    def _collect_and_print_global_metrics(self, r):
        """在每轮广播后，收一小段时间的客户端 E2E 验证统计，汇总成一行全局指标"""
        import time, numpy as np, queue
        wait_s = float(self.cfg.get("server", {}).get("metrics_wait_s", 3.0))

        cm_sum = None
        loss_sum = 0.0
        n_sum = 0
        got = 0

        t0 = time.time()
        while time.time() - t0 < wait_s and got < self.N:
            try:
                msg = self.queues["to_srv"].get(timeout=0.3)
            except queue.Empty:
                continue
            except Exception:
                continue

            if msg.get("round") != r or msg.get("type") != "metrics":
                # 非本轮或非指标消息：丢回队列尽量不打乱流程（可选，不丢回也问题不大）
                continue

            pay = msg.get("payload", {})
            if pay.get("split") != "val":
                continue

            cm = np.array(pay["cm"])
            if cm_sum is None:
                cm_sum = cm.copy()
            else:
                cm_sum += cm
            loss_sum += float(pay.get("loss_sum", 0.0))
            n_sum    += int(pay.get("n", 0))
            got += 1

        if cm_sum is None:
            print(f"[Global] R{r:03d} | val: (no metrics received)", flush=True)
            return

        loss, acc, prec, reca, f1 = self._metrics_from_cm(cm_sum, loss_sum, n_sum)
        print(f"[Global] R{r:03d} | val: loss={loss:.4f} acc={acc:.4f} "
            f"precision={prec:.4f} recall={reca:.4f} f1_macro={f1:.4f}", flush=True)

    def _ingest_synth(self, schema, blob):
        """
        把客户端上传的合成样本（压缩包 blob）解包后放入本轮池子。
        """
        try:
            arrs = unpack_arrays(blob)  # {"X": np.ndarray, "y": np.ndarray}
            X_np, y_np = arrs["X"], arrs["y"]
            # 限流：若太大则随机下采样（服务器侧再保一层保险）
            if len(X_np) > self._srv_pool_cap:
                idx = np.random.choice(len(X_np), size=self._srv_pool_cap, replace=False)
                X_np = X_np[idx]; y_np = y_np[idx]
            self._synth_pool[schema].append((X_np, y_np))
        except Exception as e:
            print(f"[Server][WARN] failed to unpack synth blob for schema={schema}: {e}", flush=True)

    def _train_central_hetero_once(self):
        """
        将当前轮已收集到的合成样本拼接→注入中央异构代理→做一次近线训练→清空池。
        """
        if self.central_hetero is None:
            # 尚未收到任何 meta（极小概率），直接跳过
            self._synth_pool.clear()
            return

        # 把同 schema 的块拼起来再注入
        for schema, chunks in list(self._synth_pool.items()):
            if not chunks:
                continue
            try:
                X = np.concatenate([x for x, _ in chunks], axis=0)
                y = np.concatenate([y for _, y in chunks], axis=0)
            except Exception as e:
                print(f"[Server][WARN] concat failed for schema={schema}: {e}", flush=True)
                continue
            # 注入到中央异构代理的对应分支
            try:
                self.central_hetero.inject(schema, X, y)
            except Exception as e:
                print(f"[Server][WARN] inject failed for schema={schema}: {e}", flush=True)

        # 近线训练（可把 epochs 调小；服务器可以只做 1 个 epoch 以求稳）
        try:
            logs = self.central_hetero.train_one_round(
                batch_size=self._srv_batch_size,
                epochs=self._srv_hetero_epochs
            )
            for k, v in (logs or {}).items():
                print(f"[Server] central hetero[{k}] loss={v['loss']:.4f} acc={v['acc']:.4f} n={v['n']}", flush=True)
        except Exception as e:
            print(f"[Server][WARN] central_hetero train failed: {e}", flush=True)

        # 清空样本池，进入下一轮
        self._synth_pool.clear()

    def _maybe_pack_hetero_state(self):
        """
        （可选）将中央异构代理权重打包随广播发下去，便于客户端做蒸馏/初始化。
        客户端目前不会显式加载它；你后续可在客户端加对应接收逻辑。
        """
        if not self._broadcast_hetero or self.central_hetero is None:
            return None
        state = {}
        for name, agent in self.central_hetero.agents.items():
            try:
                state[name] = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
            except Exception:
                # 某些代理可能未初始化完全，安全起见跳过
                pass
        return state if state else None

    def run(self):
        R = int(self.cfg["train"]["rounds"])
        print(f"[Server] start rounds={R}", flush=True)

        for r in range(1, R + 1):
            print(f"[Server] collecting round {r}/{R} ...", flush=True)

            states = []
            metas = []
            collected = 0

            # 持续拉取该轮消息：既要收集 N 份 weights，也要顺手接收 synth
            while collected < self.N:
                try:
                    msg = self.queues["to_srv"].get(timeout=10.0)
                except queue.Empty:
                    print(f"[Server][WAIT] round {r}: collected {collected}/{self.N}, still waiting ...", flush=True)
                    continue
                except Exception:
                    print(f"[Server][WAIT] round {r}: collected {collected}/{self.N}, still waiting ...", flush=True)
                    continue

                # 仅处理本轮消息
                if msg.get("round") != r:
                    continue

                mtype = msg.get("type", "")
                if mtype == "weights":
                    # 收集生成器 score 的 CPU 权重
                    try:
                        pay = msg["payload"]
                        state_cpu = {k: v.cpu() for k, v in pay["score_state"].items()}
                        states.append(state_cpu)
                        metas.append(pay["meta"])
                        collected += 1
                        # 懒初始化中央异构代理
                        self._maybe_init_central(pay["meta"])
                    except Exception as e:
                        print(f"[Server][WARN] bad weights msg: {e}", flush=True)

                elif mtype == "synth":
                    # 收集通过门控的合成样本（各 schema）
                    try:
                        schema = msg["payload"]["schema"]
                        blob = msg["payload"]["blob"]
                        self._ingest_synth(schema, blob)
                    except Exception as e:
                        print(f"[Server][WARN] bad synth msg: {e}", flush=True)

                else:
                    # 其它类型暂不处理
                    pass

            # —— FedAvg 聚合（CPU）——
            if not states:
                print(f"[Server][ERROR] no states collected at round {r}", flush=True)
                # 不中断，但也没法广播有效参数
                agg = self.global_state if self.global_state is not None else None
            else:
                agg = self._fedavg(states)
                self.global_state = agg

            print(f"[Server] agg round {r}/{R}", flush=True)

            # 每 K 轮触发“请本地采样”
            synth_cmd = {"command": "please_sample"} if (r % self.K == 0) else None

            # （可选）打包中央异构代理权重
            hetero_state = self._maybe_pack_hetero_state()

            # 广播给所有客户端：聚合后的 score_state + 采样指令 (+ 可选的 hetero_state)
            for cid in range(self.N):
                payload = {"score_state": self.global_state, "synth": synth_cmd}
                if hetero_state is not None:
                    payload["hetero_state"] = hetero_state
                self.queues[f"to_cli_{cid}"].put({
                    "round": r,
                    "type": "broadcast",
                    "payload": payload
                })

            print(f"[Server] round {r}/{R}: aggregated & broadcasted", flush=True)

            # —— 聚合完成后：用本轮收到的合成样本训练“中央异构代理” —— #
            if len(self._synth_pool) > 0:
                self._train_central_hetero_once()
        print(f"[Server] round {r}/{R}: aggregated & broadcasted", flush=True)

# ← 新增：收客户端E2E指标并打印全局一行
        self._collect_and_print_global_metrics(r)

        # 若本轮有收到合成样本，则训练中央异构代理
        if len(self._synth_pool) > 0:
            self._train_central_hetero_once()

        print("[Server] finished.", flush=True)
