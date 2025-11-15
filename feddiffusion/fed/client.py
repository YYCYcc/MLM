# fed/client.py
import os, csv, json, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils.dataset import make_loaders
from utils.ipc import pack_arrays
from agents.flow_mlp import FlowMLP
from agents.hetero import HeteroManager
from gen.tabddpm import SimpleDDPM
from gate.quality_gate import fidelity_gate


# --------------------------- 小型曲线记录器 --------------------------- #
class CurveLogger:
    """
    把每轮训练/验证/异构评测指标存 jsonl，并绘制 png 曲线。
    文件位置：
      outs/curves/client_{cid}_curves.jsonl
      outs/curves/*.png
    """
    def __init__(self, cid: int, out_dir: str):
        self.cid = cid
        self.records = []  # 每轮一条 dict
        self.base = Path(out_dir) / "curves"
        self.base.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.base / f"client_{cid}_curves.jsonl"

    def add_round(self, round_id: int,
                  train_cls_loss: float, train_gen_loss: float, train_steps: int,
                  val: dict, test: dict | None,
                  dispatch: dict | None, hetero: dict | None):
        rec = {
            "round": round_id,
            "train": {
                "cls_loss_mean": float(train_cls_loss),
                "gen_loss_mean": float(train_gen_loss),
                "steps": int(train_steps),
            },
            "val": val,
            "test": test,
            "dispatch": dispatch,
            "hetero": hetero,
            "ts": time.time(),
        }
        self.records.append(rec)
        try:
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _extract_series(self, key_path: list[str]):
        """从 self.records 中按 key_path 提取一条曲线，缺失用 None 填充"""
        vals = []
        for r in self.records:
            cur = r
            ok = True
            for k in key_path:
                if isinstance(cur, dict) and (k in cur):
                    cur = cur[k]
                else:
                    ok = False
                    break
            vals.append(float(cur) if ok and isinstance(cur, (int, float)) else None)
        return vals

    def save_png_curves(self, out_dir: str):
        """画若干常用曲线：训练/验证/异构"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return  # 环境缺图形库时静默

        xs = [rec["round"] for rec in self.records]
        if not xs:
            return

        # 1) 训练损失
        try:
            y1 = self._extract_series(["train", "cls_loss_mean"])
            y2 = self._extract_series(["train", "gen_loss_mean"])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xs, y1, label="train_cls_loss")
            ax.plot(xs, y2, label="train_gen_loss")
            ax.set_title(f"Client {self.cid} - Train Loss")
            ax.set_xlabel("round")
            ax.legend()
            fig.savefig(self.base / f"client_{self.cid}_train_loss.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        # 2) 验证指标
        try:
            v_loss = self._extract_series(["val", "loss"])
            v_acc  = self._extract_series(["val", "acc"])
            v_f1   = self._extract_series(["val", "f1"])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xs, v_loss, label="val_loss")
            ax.plot(xs, v_acc,  label="val_acc")
            ax.plot(xs, v_f1,   label="val_f1")
            ax.set_title(f"Client {self.cid} - Validation")
            ax.set_xlabel("round")
            ax.legend()
            fig.savefig(self.base / f"client_{self.cid}_val.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        # 3) 异构代理（若有）
        try:
            a_flow = self._extract_series(["hetero", "acc_flow"])
            a_temp = self._extract_series(["hetero", "acc_temporal"])
            a_gnn  = self._extract_series(["hetero", "acc_graph"])
            if any(v is not None for v in a_flow + a_temp + a_gnn):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(xs, a_flow, label="flow_acc")
                ax.plot(xs, a_temp, label="temporal_acc")
                ax.plot(xs, a_gnn,  label="graph_acc")
                ax.set_title(f"Client {self.cid} - Hetero Acc (buffers)")
                ax.set_xlabel("round")
                ax.legend()
                fig.savefig(self.base / f"client_{self.cid}_hetero.png", dpi=160, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass


# ------------------------------- Client ------------------------------- #
class Client:
    def _eval_end2end(self, loader):
        """端到端：用路由+各代理做推理（若没有路由方法则回退到主分类器）"""
        n_cls = self.n_classes
        total_loss, total_num = 0.0, 0
        cm = torch.zeros((n_cls, n_cls), dtype=torch.long, device=self.device)
        self.agent.eval()
        if hasattr(self.hetero, "eval"):
            try:
                self.hetero.eval()
            except Exception:
                pass

        with torch.no_grad():
            for xb, yb in loader:
                xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device)
                yb = yb.long().to(self.device)

                # 端到端前向：优先走 hetero.route_predict，否则回退主分类器
                if hasattr(self.hetero, "route_predict"):
                    try:
                        logits = self.hetero.route_predict(xb)
                    except Exception:
                        logits = self.agent(xb)
                else:
                    logits = self.agent(xb)

                # 端到端也做 Logit Adjustment（仅用于预测，不影响 loss）
                if getattr(self, "use_logit_adjust", False):
                    logits_for_pred = logits - self.logit_bias
                else:
                    logits_for_pred = logits

                loss = F.cross_entropy(logits, yb, reduction="sum", weight=self.class_weights)
                total_loss += loss.item()
                total_num  += yb.size(0)
                pred = logits_for_pred.argmax(dim=1)
                for t, p in zip(yb, pred):
                    cm[t, p] += 1

        loss = total_loss / max(1, total_num)
        acc  = cm.trace().item() / max(1, cm.sum().item())
        eps = 1e-9
        tp = torch.diag(cm).float()
        fp = cm.sum(0).float() - tp
        fn = cm.sum(1).float() - tp
        precision = (tp / (tp + fp + eps)).mean().item()
        recall    = (tp / (tp + fn + eps)).mean().item()
        f1_macro  = (2 * precision * recall / (precision + recall + eps)) if (precision+recall) > 0 else 0.0
        return loss, acc, precision, recall, float(f1_macro), cm.detach().cpu().numpy()

    def __init__(self, cid, cfg, queues, root_dir):
        print(f"[Client {cid}] cfg.train = {cfg['train']}", flush=True)
        print("[Client {}] types = {}".format(
            cid, {k: type(v).__name__ for k, v in cfg["train"].items()}
        ), flush=True)

        # ---- 将超参转成正确类型 ----
        lr_agent = float(cfg["train"]["lr_agent"])
        lr_gen   = float(cfg["train"]["lr_gen"])
        batch_sz = int(cfg["train"]["batch_size"])

        self.cid, self.cfg, self.queues = cid, cfg, queues
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 统一日志/制品目录
        self.root_dir = root_dir
        self.log_dir  = os.path.join(root_dir, "logs")
        self.out_dir  = os.path.join(root_dir, "outs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

        # ---- 数据目录（client_00 ~ client_05）----
        base_dir   = "/home/yyc/data/data/fed_splits"
        client_dir = os.path.join(base_dir, f"client_{cid:02d}")
        print(f"[Client {cid}] data dir -> {client_dir}", flush=True)
        if not os.path.isdir(client_dir):
            raise FileNotFoundError(f"[Client {cid}] dir not found: {client_dir}")

        # 统一类别词表（可选）
        global_vocab = cfg["data"].get("known_classes", None)

        # ---- 显式拼出三个 CSV ----
        train_csv = os.path.join(client_dir, "train.csv")
        val_csv   = os.path.join(client_dir, "val.csv")
        test_csv  = os.path.join(client_dir, "test.csv")
        for p in (train_csv, val_csv, test_csv):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"[Client {cid}] missing file: {p}")

        # ✅ 与 utils/dataset.py 的签名对齐
        (self.train_loader, self.val_loader, self.test_loader,
         self.classes, self.x_dim) = make_loaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            label_col=cfg["data"]["label_col"],
            batch_size=batch_sz,
            known_classes=global_vocab
        )

        self.n_classes = len(self.classes)
        print(
            f"[Client {cid}] x_dim={self.x_dim}, n_classes={self.n_classes}, "
            f"len(train)={len(self.train_loader.dataset)}, "
            f"len(val)={len(self.val_loader.dataset)}, "
            f"len(test)={len(self.test_loader.dataset)}",
            flush=True
        )

        # ---- 模型与优化器 ----
        self.agent = FlowMLP(self.x_dim, self.n_classes).to(self.device)
        self.opt_agent = Adam(self.agent.parameters(), lr=lr_agent)

        # 健壮读取 gen 配置，提供默认值，避免 KeyError
        gen_cfg = cfg.get("gen", {}) if isinstance(cfg.get("gen", {}), dict) else {}
        self.gen_cfg = gen_cfg

        T_steps   = int(gen_cfg.get("steps", 1000))           # 默认 1000
        cond_dim  = int(gen_cfg.get("cond_dim", 64))          # 默认 64
        schedule  = str(gen_cfg.get("schedule", "linear"))    # 默认 linear
        n_clients = int(cfg.get("data", {}).get("n_clients", 8))

        print(f"[Client {cid}] gen.cfg -> steps={T_steps}, cond_dim={cond_dim}, schedule={schedule}", flush=True)

        self.gen = SimpleDDPM(
            self.x_dim, self.n_classes, n_clients,
            T=T_steps,
            cond_dim=cond_dim,
            device=self.device,
            schedule=schedule
        ).to(self.device)
        self.opt_gen = Adam(self.gen.parameters(), lr=lr_gen)

        # ---- 其它状态 ----
        self.cid_tensor = None
        self._synth_buffer = None    # (Xs_cpu, Ys_cpu) for raw mix-in
        self.dump_synth = bool(self.gen_cfg.get("dump_synth", False))

        # ---- 类别频次/权重（可用于采样与损失加权）----
        from collections import Counter
        cnt = Counter()
        for _, y in self.train_loader:
            cnt.update(y.tolist())
        self.class_freq = torch.tensor([cnt.get(i, 0) for i in range(self.n_classes)],
                                       dtype=torch.float32) + 1.0
        cw = (self.class_freq.sum() / self.class_freq).to(self.device)
        self.class_weights = cw / cw.mean()

        # ---- (训练)Focal Loss 与 (评测)Logit Adjustment 配置 ----
        tr_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
        ev_cfg = cfg.get("eval",  {}) if isinstance(cfg.get("eval",  {}), dict) else {}
        self.use_focal    = bool(tr_cfg.get("use_focal", True))
        self.focal_gamma  = float(tr_cfg.get("focal_gamma", 2.0))
        self.use_logit_adjust = bool(ev_cfg.get("logit_adjust", True))
        tau = float(ev_cfg.get("la_tau", 1.0))
        eps = 1e-8
        prior = (self.class_freq / self.class_freq.sum()).clamp_min(eps)  # π_y
        # 经典 LA：logits_adj = logits - τ * log π_y（仅推理用）
        self.logit_bias = (tau * torch.log(prior)).to(self.device)

        # 词表一致性检查（可选提示）
        if cfg["data"].get("known_classes"):
            if len(cfg["data"]["known_classes"]) != self.n_classes:
                print(f"[WARN][Client {cid}] n_classes({self.n_classes}) != len(known_classes)"
                      f"({len(cfg['data']['known_classes'])}). 请确保 make_loaders 按 known_classes 编码。",
                      flush=True)

        # ---- 异构管理器 ----
        self.hetero = HeteroManager(device=self.device, n_classes=self.n_classes,
                                x_dim=self.x_dim, tlist_len=3, lr=lr_agent,
                                class_weights=self.class_weights)
        self.hetero.attach_flow(self.agent)

        # ---- 日志（CSV）----
        self.log_path = os.path.join(self.log_dir, f"client_{cid}.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "epoch", "step", "loss_cls", "acc", "loss_gen"])

        self.eval_log_path = os.path.join(self.log_dir, f"client_{cid}_eval.csv")
        if not os.path.exists(self.eval_log_path):
            with open(self.eval_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "split", "loss", "acc", "precision_macro", "recall_macro", "f1_macro"])

        self.gate_log_path = os.path.join(self.log_dir, f"client_{cid}_gate.csv")
        if not os.path.exists(self.gate_log_path):
            with open(self.gate_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "schema", "ks", "mmd", "ok", "injected_n"])

        # —— 路由/异构评测日志 —— #
        self.router_log_path = os.path.join(self.log_dir, f"client_{cid}_router.csv")
        if not os.path.exists(self.router_log_path):
            with open(self.router_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "n_flow", "n_temporal", "n_graph",
                            "acc_flow", "acc_temporal", "acc_graph"])
        self.e2e_log_path = os.path.join(self.log_dir, f"client_{cid}_e2e_eval.csv")
        if not os.path.exists(self.e2e_log_path):
            with open(self.e2e_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["round", "split", "loss", "acc", "precision_macro", "recall_macro", "f1_macro"])

        # —— 生成-上传给服务器（可关） —— #
        self.share_to_server = bool(self.cfg.get("comm", {}).get("synth_to_server", True))
        self.share_max_per_schema = int(self.cfg.get("comm", {}).get("synth_max_per_schema", 5000))

        # —— 曲线记录器 —— #
        self.curve_logger = CurveLogger(cid=self.cid, out_dir=self.out_dir)

    # --------- 工具函数 ---------
    def _lazy_cid_tensor(self, n):
        if self.cid_tensor is None or self.cid_tensor.numel() != n:
            self.cid_tensor = torch.full((n,), self.cid, device=self.device, dtype=torch.long)
        return self.cid_tensor

    def _send_synth_to_server(self, round_id: int, schema: str, X_np, y_np):
        if not self.share_to_server:
            return
        n = len(X_np)
        if n == 0:
            return
        if n > self.share_max_per_schema:
            idx = np.random.choice(n, size=self.share_max_per_schema, replace=False)
            X_np = X_np[idx]; y_np = y_np[idx]
        blob = pack_arrays(X=X_np, y=y_np)
        try:
            self.queues["to_srv"].put({
                "round": round_id,
                "type": "synth",
                "payload": {
                    "schema": schema,
                    "blob": blob,
                    "meta": {
                        "client": int(self.cid),
                        "x_dim": int(self.x_dim),
                        "n_classes": int(self.n_classes)
                    }
                }
            }, block=False)
        except Exception:
            pass

    def inject_synth(self, xs_np, ys_np):
        """把通过门控的合成样本保存进缓冲区（含正确标签），供下一轮 raw 视图混入"""
        if xs_np is None or len(xs_np) == 0:
            return
        Xs = torch.tensor(xs_np, dtype=torch.float32)
        Ys = torch.tensor(ys_np, dtype=torch.long)
        self._synth_buffer = (Xs.detach().cpu(), Ys.detach().cpu())

    def maybe_mix_batch(self, xb, yb, mix_ratio: float):
        """把缓冲区里的合成样本按比例混到一个 batch 的前 k 条（raw 视图）"""
        if self._synth_buffer is None:
            return xb, yb
        Xs, Ys = self._synth_buffer
        k = int(len(xb) * max(0.0, min(1.0, mix_ratio)))
        k = min(k, len(Xs))
        if k > 0:
            xb[:k] = Xs[:k].to(xb.device)
            yb[:k] = Ys[:k].to(yb.device)
        return xb, yb

    # 1) 更宽松的映射：未知一律回落到 flow
    def _schema_to_bucket(self, schema_name: str) -> str | None:
        s = str(schema_name).lower()
        if s in ("raw", "denoised", "score_ms"):
            return "flow"
        if s.startswith("temporal"):
            return "temporal"
        if s.startswith("graph"):
            return "graph"
        # 以前这里是 return None，导致完全不注入
        return "flow"

    # 2) 维度检查：对 flow 放宽为 “二维并且 D==x_dim 就行”
    def _can_inject(self, schema_name: str, Xv: np.ndarray) -> bool:
        s = str(schema_name).lower()
        if s in ("raw", "denoised", "score_ms"):  # 都当成 flow 的 2D 特征
            return (Xv.ndim == 2) and (Xv.shape[1] == int(self.x_dim))
        # 其它还是照 expected_dims 严格检查
        if s.startswith("temporal") or s.startswith("graph"):
            exp = getattr(self.hetero, "expected_dims", None)
            if isinstance(exp, dict) and s.split(":")[0] in exp:  # 允许像 temporal:win3 这种前缀
                return Xv.ndim == 2 and (Xv.shape[1] == int(exp[s.split(':')[0]]))
        return False


    def _eval_split(self, loader):
        """对给定数据集做一次评估，返回 (loss, acc, p_macro, r_macro, f1_macro, cm_np)"""
        self.agent.eval()
        n_cls = self.n_classes
        total_loss, total_num = 0.0, 0
        cm = torch.zeros((n_cls, n_cls), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for xb, yb in loader:
                xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device)
                yb = yb.long().to(self.device)
                logits = self.agent(xb)

                # 评测损失仍用原始 logits；预测用于统计时应用 LA
                loss = F.cross_entropy(logits, yb, reduction="sum", weight=self.class_weights)
                total_loss += loss.item()
                total_num  += yb.size(0)

                if getattr(self, "use_logit_adjust", False):
                    logits_for_pred = logits - self.logit_bias
                else:
                    logits_for_pred = logits
                pred = logits_for_pred.argmax(dim=1)

                for t, p in zip(yb, pred):
                    cm[t, p] += 1

        loss = total_loss / max(1, total_num)
        acc  = cm.trace().item() / max(1, cm.sum().item())

        eps = 1e-9
        tp = torch.diag(cm).float()
        fp = cm.sum(0).float() - tp
        fn = cm.sum(1).float() - tp
        precision = (tp / (tp + fp + eps)).mean().item()
        recall    = (tp / (tp + fn + eps)).mean().item()
        f1_macro  = (2 * precision * recall / (precision + recall + eps)) if (precision+recall) > 0 else 0.0
        return loss, acc, precision, recall, float(f1_macro), cm.detach().cpu().numpy()

    def _save_round_artifacts(self, r, split, cm_np, gate_rows=None, dispatch_cnt=None, gen_diag=None):
        """保存每轮产物（混淆矩阵/门控/分发/扩散诊断）"""
        # 1) 混淆矩阵
        cm_dir = Path(self.out_dir) / "cm" / f"client_{self.cid:02d}"
        cm_dir.mkdir(parents=True, exist_ok=True)
        np.save(cm_dir / f"cm_round_{r}_{split}.npy", cm_np)

        # 可选：画热力图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(cm_np, origin="upper")
            ax.set_title(f"Client {self.cid} - {split} - round {r}")
            fig.colorbar(im, ax=ax)
            fig.savefig(cm_dir / f"cm_round_{r}_{split}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        # 2) 门控快照
        if gate_rows:
            gate_snap = Path(self.out_dir) / "gate" / f"client_{self.cid:02d}"
            gate_snap.mkdir(parents=True, exist_ok=True)
            (gate_snap / f"r{r:04d}_gate.json").write_text(
                json.dumps(gate_rows, ensure_ascii=False, indent=2)
            )

        # 3) 分发计数
        if dispatch_cnt:
            disp_dir = Path(self.out_dir) / "dispatch" / f"client_{self.cid:02d}"
            disp_dir.mkdir(parents=True, exist_ok=True)
            (disp_dir / f"r{r:04d}_dispatch.json").write_text(
                json.dumps(dispatch_cnt, ensure_ascii=False, indent=2)
            )

        # 4) 扩散对比诊断
        if gen_diag:
            gdir = Path(self.out_dir) / "diffusion_diag" / f"client_{self.cid:02d}"
            gdir.mkdir(parents=True, exist_ok=True)
            (gdir / f"r{r:04d}_diag.json").write_text(
                json.dumps(gen_diag, ensure_ascii=False, indent=2)
            )

    # --------------------------- 训练主循环 --------------------------- #
    def run(self):
        R = int(self.cfg["train"]["rounds"])
        E = int(self.cfg["train"]["local_epochs"])

        # 从 self.gen_cfg 抽取，给默认值（避免 KeyError）
        mix_ratio   = float(self.gen_cfg.get("mix_ratio", 0.0))
        het_epochs  = int(self.gen_cfg.get("hetero_epochs", 1))
        synth_n_def = int(self.gen_cfg.get("synth_n", 512))
        alpha       = float(self.gen_cfg.get("alpha", 1.0))            # 类感知采样指数
        quota_pc    = int(self.gen_cfg.get("quota_per_class", 0))      # 每类定额（0 关闭）

        print(f"[Client {self.cid}] start rounds={R}", flush=True)

        step_global = 0
        for r in range(1, R + 1):
            # === 逐轮训练均值累加器 ===
            round_cls_loss_sum = 0.0
            round_gen_loss_sum = 0.0
            round_train_steps  = 0

            # 进入训练模式
            self.agent.train()
            self.gen.train()

            # == 本地训练 ==
            for e in range(E):
                for xb, yb in self.train_loader:
                    # 清洗 + 标签 long
                    xb2, yb2 = xb.clone(), yb.clone()
                    xb2 = torch.nan_to_num(xb2, nan=0.0, posinf=1e6, neginf=-1e6)
                    yb2 = yb2.long()

                    # 按 mix_ratio 混入（raw 视图）
                    if mix_ratio > 0.0:
                        xb2, yb2 = self.maybe_mix_batch(xb2, yb2, mix_ratio)

                    xb2, yb2 = xb2.to(self.device), yb2.to(self.device)

                    # 判别器步（分类器）：可选 Focal Loss
                    self.opt_agent.zero_grad(set_to_none=True)
                    logits = self.agent(xb2)
                    if self.use_focal:
                        ce = F.cross_entropy(logits, yb2, reduction="none",
                                             weight=self.class_weights)
                        pt = torch.softmax(logits, dim=1).gather(1, yb2.view(-1, 1)).squeeze(1)
                        loss_cls = ((1 - pt) ** self.focal_gamma * ce).mean()
                    else:
                        loss_cls = F.cross_entropy(logits, yb2, weight=self.class_weights)
                    loss_cls.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                    self.opt_agent.step()
                    acc = (logits.argmax(dim=1) == yb2).float().mean().detach()

                    # 生成器步（扩散）
                    cid_t = self._lazy_cid_tensor(len(yb2))
                    loss_gen = self.gen.loss(xb2, yb2, cid_t)
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0)
                    self.opt_gen.step()

                    # 记录日志（逐步 & 逐轮累计）
                    step_global += 1
                    with open(self.log_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([r, e, step_global,
                                    _to_float(loss_cls), _to_float(acc), _to_float(loss_gen)])
                    round_cls_loss_sum += float(loss_cls.detach().cpu().item())
                    round_gen_loss_sum += float(loss_gen.detach().cpu().item())
                    round_train_steps  += 1

            # == 上传生成器 score 的 CPU 权重 ==
            score_state_cpu = {k: v.detach().cpu() for k, v in self.gen.score.state_dict().items()}
            self.queues["to_srv"].put({
                "round": r, "type": "weights",
                "payload": {
                    "score_state": score_state_cpu,
                    "meta": {"x_dim": self.x_dim, "n_classes": self.n_classes}
                }
            })

            # == 等待服务器广播 ==
            msg = self.queues[f"to_cli_{self.cid}"].get()
            assert msg["type"] == "broadcast" and msg["round"] == r
            self.gen.score.load_state_dict(msg["payload"]["score_state"])

            # 这些变量用于 _save_round_artifacts
            gate_rows = None
            dispatch_cnt = None
            gen_diag = None

            # == 若需合成（多结构视图）==
            if msg["payload"].get("synth") is not None:

                # —— 类感知采样（支持每类定额）—— #
                with torch.no_grad():
                    freq = self.class_freq.to(self.device)
                    p = (1.0 / (freq ** max(0.0, alpha)))
                    p = p / p.sum()
                if quota_pc > 0:
                    y_list = [torch.full((quota_pc,), c, device=self.device, dtype=torch.long)
                              for c in range(self.n_classes)]
                    y_cls = torch.cat(y_list, dim=0)
                    n = int(y_cls.numel())
                else:
                    n = int(synth_n_def)
                    y_cls = torch.multinomial(p, num_samples=n, replacement=True)
                y_cli = torch.full((n,), self.cid, device=self.device, dtype=torch.long)

                # 1) 生成 raw 合成样本
                with torch.no_grad():
                    Xs_t = self.gen.sample(n, self.x_dim, y_cls, y_cli)  # [n, x_dim]
                Xs_t = torch.nan_to_num(Xs_t, nan=0.0, posinf=1e6, neginf=-1e6)
                Xs = Xs_t.detach().cpu().numpy()
                Ys = y_cls.detach().cpu().numpy()

                # 2) 真实数据对照（第一批）
                xb_ref, yb_ref = None, None
                for xb_ref_, yb_ref_ in self.train_loader:
                    xb_ref = xb_ref_; yb_ref = yb_ref_
                    break

                # 3) 生成“视图”（raw / temporal_* / graph_*） —— 注意：后续有维度检查
                Xr_views = self.gen.make_views(
                    xb_ref.to(self.device),
                    yb_ref.to(self.device).long(),
                    torch.full((len(yb_ref),), self.cid, device=self.device, dtype=torch.long),
                    t_list=(50, 250, 450)
                )  # dict[str, np.ndarray]

                synth_views = self.gen.make_views(
                    torch.tensor(Xs, dtype=torch.float32, device=self.device),
                    torch.tensor(Ys, dtype=torch.long, device=self.device),
                    torch.full((len(Ys),), self.cid, device=self.device, dtype=torch.long),
                    t_list=(50, 250, 450)
                )

                # ---- 读取 gate 基线阈值 + raw 兜底开关 ----
                gate_cfg = self.cfg.get("gate", {}) if isinstance(self.cfg.get("gate", {}), dict) else {}
                ks_base   = float(gate_cfg.get("ks", 0.2))
                mmd_base  = float(gate_cfg.get("mmd", 0.2))
                accept_raw_always = bool(gate_cfg.get("accept_raw_always", True))

                # 调试：第一轮/每5轮打印键名，便于对齐 schema 前缀
                if r == 1 or (r % 5 == 0):
                    print(f"[Client {self.cid}] synth_views keys -> {list(synth_views.keys())}", flush=True)
                    print(f"[Client {self.cid}] real_views  keys -> {list(Xr_views.keys())}", flush=True)

                # 4) 门控 + 异构注入（带维度检查 + 动态阈值“先松后紧”）
                dispatch_cnt = {"flow": 0, "temporal": 0, "graph": 0}
                gate_rows = []

                # 少数类比例估计，用于放宽阈值
                median_freq = torch.median(self.class_freq).item()
                is_minor = (self.class_freq < median_freq).to(torch.bool).cpu().numpy()
                minor_ratio = float(is_minor[Ys].mean()) if len(Ys) > 0 else 0.0
                scale = 1.0 - 0.4 * minor_ratio   # 最多放宽 40%
                ks_thr  = ks_base  * max(0.2, scale)
                mmd_thr = mmd_base * max(0.2, scale)

                for schema_name, Xv in synth_views.items():
                    Xr_use = Xr_views.get(schema_name, None)
                    if Xr_use is None or len(Xr_use) == 0 or len(Xv) == 0:
                        ks_val = None; mmd_val = None; ok = False; inj_n = 0
                    else:
                        log, ok = fidelity_gate(Xv, Xr_use, ks_thr=ks_thr, mmd_thr=mmd_thr)
                        ks_val = log.get("ks_val"); mmd_val = log.get("mmd_val")

                        # ★ 兜底：raw 一律放行（用于先跑通 flow 桶）
                        if schema_name == "raw" and accept_raw_always:
                            ok = True

                        inj_n = 0
                        if ok:
                            bucket = self._schema_to_bucket(schema_name)
                            # 仅当“维度安全通过”时才注入异构，以免拼接报错
                            if bucket is not None and self._can_inject(schema_name, Xv):
                                self.hetero.inject(bucket, Xv, Ys)
                                dispatch_cnt[bucket] += int(len(Xv))
                                inj_n = len(Xv)
                                # raw 同时注入主代理的混入缓冲
                                if schema_name == "raw" and mix_ratio > 0.0:
                                    self.inject_synth(Xv, Ys)
                                # 可选：上报服务器
                                self._send_synth_to_server(r, schema_name, Xv, Ys)
                            else:
                                # 安全起见：仅 raw -> flow 默认允许（只用于主代理混入）
                                if schema_name == "raw" and mix_ratio > 0.0:
                                    self.inject_synth(Xv, Ys)

                    gate_rows.append((schema_name, ks_val, mmd_val, int(ok), inj_n))

                # 5) 异构代理近线小训练 & 小评测（若 HeteroManager 支持）
                het_log = self.hetero.train_one_round(
                    batch_size=int(self.cfg["train"]["batch_size"]),
                    epochs=het_epochs
                )
                if hasattr(self.hetero, "eval_on_buffers"):
                    het_eval = self.hetero.eval_on_buffers()
                else:
                    het_eval = {"flow": {"acc": 0.0}, "temporal": {"acc": 0.0}, "graph": {"acc": 0.0}}

                acc_flow     = float(het_eval.get("flow",     {}).get("acc", 0.0))
                acc_temporal = float(het_eval.get("temporal", {}).get("acc", 0.0))
                acc_graph    = float(het_eval.get("graph",    {}).get("acc", 0.0))

                # router.csv（每轮一行）
                with open(self.router_log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([r,
                                dispatch_cnt.get("flow", 0),
                                dispatch_cnt.get("temporal", 0),
                                dispatch_cnt.get("graph", 0),
                                f"{acc_flow:.6f}", f"{acc_temporal:.6f}", f"{acc_graph:.6f}"])

                gen_diag = {
                    "gate": [{"schema": n, "ks": kv, "mmd": mv, "ok": bool(okv), "n": int(nn)}
                             for (n, kv, mv, okv, nn) in gate_rows],
                    "dispatch": dispatch_cnt
                }

                print(f"[Client {self.cid}] R{r:03d} | dispatch={dispatch_cnt} | "
                      f"hetero acc(flow/tmp/gnn)={acc_flow:.3f}/{acc_temporal:.3f}/{acc_graph:.3f}",
                      flush=True)

                # 调试：第一轮/每5轮观测 dispatch 结构
                if r == 1 or (r % 5 == 0):
                    print(f"[Client {self.cid}] R{r:03d} | dispatch(keys)={list(dispatch_cnt.keys())} -> {dispatch_cnt}",
                          flush=True)

                # 写 gate.csv 追加多视图结果
                with open(self.gate_log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    for (name, ks_v, mmd_v, ok_v, inj_n) in gate_rows:
                        w.writerow([r, name, ks_v, mmd_v, ok_v, inj_n])

            # == 每轮评估（验证集）==
            val_loss, val_acc, p_m, r_m, f1_m, cm = self._eval_split(self.val_loader)
            with open(self.eval_log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([r, "val", f"{val_loss:.6f}", f"{val_acc:.6f}",
                            f"{p_m:.6f}", f"{r_m:.6f}", f"{f1_m:.6f}"])
            print(f"[Client {self.cid}] R{r:03d} | val: loss={val_loss:.4f} "
                  f"acc={val_acc:.4f} f1_macro={f1_m:.4f}", flush=True)

            # —— 保存本轮产物 —— #
            self._save_round_artifacts(
                r, split="val", cm_np=cm,
                gate_rows=gate_rows,
                dispatch_cnt=dispatch_cnt,
                gen_diag=gen_diag
            )
            # —— 端到端（路由+各代理）验证 —— #
            # —— 端到端（路由+各代理）验证 —— #
            e2e_loss, e2e_acc, e2e_p, e2e_r, e2e_f1, cm_e2e = self._eval_end2end(self.val_loader)

            # 发送到服务器用于“全局指标”汇总（loss 用总和，样本数 n 用混淆矩阵求和）
            n_e2e = int(cm_e2e.sum())
            try:
                self.queues["to_srv"].put({
                    "round": r,
                    "type": "metrics",
                    "payload": {
                        "split": "val",
                        "cm": cm_e2e,                          # numpy array OK（队列里会被pickle）
                        "loss_sum": float(e2e_loss) * n_e2e,   # 累积交叉熵
                        "n": n_e2e
                    }
                }, block=False)
            except Exception:
                pass
            # 1) 写本地 e2e 日志
            with open(self.e2e_log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([r, "val", f"{e2e_loss:.6f}", f"{e2e_acc:.6f}",
                            f"{e2e_p:.6f}", f"{e2e_r:.6f}", f"{e2e_f1:.6f}"])

            # 2) 把 E2E 结果发送给服务器做全局聚合
            try:
                self.queues["to_srv"].put({
                    "round": r,
                    "type": "e2e_metrics",
                    "payload": {
                        "client": int(self.cid),
                        "n": int(len(self.val_loader.dataset)),   # 用于加权
                        "loss": float(e2e_loss),
                        "acc":  float(e2e_acc),
                        "precision": float(e2e_p),
                        "recall":    float(e2e_r),
                        "f1":   float(e2e_f1)
                        # 如需更精细的聚合，可再传 "cm": cm_e2e.tolist(), "n_classes": int(self.n_classes)
                    }
                }, block=False)
            except Exception:
                pass

            # 3) 可选：静音每个客户端的 E2E 打印（只保留服务器的全局打印）
            if not bool(self.cfg.get("log", {}).get("quiet_client_e2e", True)):
                print(f"[Client {self.cid}] R{r:03d} | E2E(val): loss={e2e_loss:.4f} acc={e2e_acc:.4f} f1_macro={e2e_f1:.4f}", flush=True)

            # 4)（保持原有保存混淆矩阵的行为）
            self._save_round_artifacts(r, split="val_e2e", cm_np=cm_e2e)

            # 若你想把 E2E 的混淆矩阵图也单独存一份：
            self._save_round_artifacts(r, split="val_e2e", cm_np=cm_e2e)

            # —— 曲线记录 —— #
            mean_train_cls = round_cls_loss_sum / max(1, round_train_steps)
            mean_train_gen = round_gen_loss_sum / max(1, round_train_steps)
            val_pack = {"loss": val_loss, "acc": val_acc, "precision": p_m, "recall": r_m, "f1": f1_m}
            hetero_pack = None
            if 'acc_flow' in locals():
                hetero_pack = {
                    "acc_flow": float(acc_flow),
                    "acc_temporal": float(acc_temporal),
                    "acc_graph": float(acc_graph),
                }
            self.curve_logger.add_round(
                round_id=r,
                train_cls_loss=mean_train_cls,
                train_gen_loss=mean_train_gen,
                train_steps=round_train_steps,
                val=val_pack,
                test=None,
                dispatch=dispatch_cnt,
                hetero=hetero_pack
            )
            self.curve_logger.save_png_curves(out_dir=self.out_dir)

            # 可选：每 K 轮做一次测试集评估
            K = int(self.cfg.get("eval", {}).get("test_every", 0))
            if K and (r % K == 0):
                te_loss, te_acc, p_m, r_m, f1_m, cm_te = self._eval_split(self.test_loader)
                with open(self.eval_log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([r, "test", f"{te_loss:.6f}", f"{te_acc:.6f}",
                                f"{p_m:.6f}", f"{r_m:.6f}", f"{f1_m:.6f}"])
                self._save_round_artifacts(r, split="test", cm_np=cm_te)

        print(f"[Client {self.cid}] finished.", flush=True)


def _to_float(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)
