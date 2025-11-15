# utils/curve_logger.py
import os, json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class RoundCurveLogger:
    """
    逐轮记录训练/验证标量，并自动画图落盘（PNG）。
    写入：
      - {log_dir}/curves/curves_client_{cid}.jsonl   （逐轮一行，容错强）
      - {log_dir}/curves/summary_client_{cid}.csv    （便于表格/Excel）
    出图：
      - {out_dir}/plots/client_{cid}_val_acc.png
      - {out_dir}/plots/client_{cid}_val_f1.png
      - {out_dir}/plots/client_{cid}_train_losses.png
      - {out_dir}/plots/client_{cid}_dispatch.png    （按需）
      - {out_dir}/plots/client_{cid}_hetero_acc.png  （按需）
    """
    def __init__(self, log_dir: str, client_id: int):
        self.log_dir = Path(log_dir)
        self.cid = int(client_id)
        self.curve_dir = self.log_dir / "curves"
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.curve_dir / f"curves_client_{self.cid}.jsonl"
        self.csv   = self.curve_dir / f"summary_client_{self.cid}.csv"
        # 若CSV不存在，写表头
        if not self.csv.exists():
            with open(self.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "round",
                    "train_cls_loss","train_gen_loss","train_steps",
                    "val_loss","val_acc","val_precision","val_recall","val_f1",
                    "test_loss","test_acc","test_precision","test_recall","test_f1",
                    "dispatch_flow","dispatch_temporal","dispatch_graph",
                    "acc_flow","acc_temporal","acc_graph"
                ])

    # ---- 工具：读取全部 JSONL 为 list(dict) ----
    def _read_all(self):
        recs = []
        if self.jsonl.exists():
            with open(self.jsonl, "r") as f:
                for line in f:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
        return recs

    def add_round(self,
                  round_id: int,
                  train_cls_loss=None, train_gen_loss=None, train_steps=None,
                  val=None,    # dict: {"loss","acc","precision","recall","f1"}
                  test=None,   # 同上或 None
                  dispatch=None,  # dict: {"flow","temporal","graph"}
                  hetero=None     # dict: {"acc_flow","acc_temporal","acc_graph"}
                  ):
        rec = {"round": int(round_id)}
        if train_cls_loss is not None: rec["train_cls_loss"] = float(train_cls_loss)
        if train_gen_loss is not None: rec["train_gen_loss"] = float(train_gen_loss)
        if train_steps     is not None: rec["train_steps"]     = int(train_steps)
        if isinstance(val, dict):
            rec["val"] = {
                "loss": float(val.get("loss", 0.0)),
                "acc": float(val.get("acc", 0.0)),
                "precision": float(val.get("precision", 0.0)),
                "recall": float(val.get("recall", 0.0)),
                "f1": float(val.get("f1", 0.0)),
            }
        if isinstance(test, dict):
            rec["test"] = {
                "loss": float(test.get("loss", 0.0)),
                "acc": float(test.get("acc", 0.0)),
                "precision": float(test.get("precision", 0.0)),
                "recall": float(test.get("recall", 0.0)),
                "f1": float(test.get("f1", 0.0)),
            }
        if isinstance(dispatch, dict):
            rec["dispatch"] = {
                "flow": int(dispatch.get("flow", 0)),
                "temporal": int(dispatch.get("temporal", 0)),
                "graph": int(dispatch.get("graph", 0)),
            }
        if isinstance(hetero, dict):
            rec["hetero"] = {
                "acc_flow": float(hetero.get("acc_flow", 0.0)),
                "acc_temporal": float(hetero.get("acc_temporal", 0.0)),
                "acc_graph": float(hetero.get("acc_graph", 0.0)),
            }

        # 追加到 JSONL
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 同步写 CSV（便于 Excel 复盘）
        with open(self.csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                rec["round"],
                rec.get("train_cls_loss",""), rec.get("train_gen_loss",""), rec.get("train_steps",""),
                (rec.get("val",{}) or {}).get("loss",""),
                (rec.get("val",{}) or {}).get("acc",""),
                (rec.get("val",{}) or {}).get("precision",""),
                (rec.get("val",{}) or {}).get("recall",""),
                (rec.get("val",{}) or {}).get("f1",""),
                (rec.get("test",{}) or {}).get("loss",""),
                (rec.get("test",{}) or {}).get("acc",""),
                (rec.get("test",{}) or {}).get("precision",""),
                (rec.get("test",{}) or {}).get("recall",""),
                (rec.get("test",{}) or {}).get("f1",""),
                (rec.get("dispatch",{}) or {}).get("flow",""),
                (rec.get("dispatch",{}) or {}).get("temporal",""),
                (rec.get("dispatch",{}) or {}).get("graph",""),
                (rec.get("hetero",{}) or {}).get("acc_flow",""),
                (rec.get("hetero",{}) or {}).get("acc_temporal",""),
                (rec.get("hetero",{}) or {}).get("acc_graph",""),
            ])

    # ---- 出图：每次调用都会重读 JSONL 并重画 PNG ----
    def save_png_curves(self, out_dir: str):
        out_dir = Path(out_dir) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        recs = self._read_all()
        if not recs:
            return

        rounds = [r.get("round", None) for r in recs]

        # 1) val_acc
        val_acc = [ (r.get("val",{}) or {}).get("acc", None) for r in recs ]
        if any(a is not None for a in val_acc):
            plt.figure()
            plt.plot(rounds, val_acc, marker="o")
            plt.xlabel("Round"); plt.ylabel("Val Acc")
            plt.title(f"Client {self.cid} - Validation Accuracy")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir/f"client_{self.cid}_val_acc.png", dpi=160, bbox_inches="tight")
            plt.close()

        # 2) val_f1
        val_f1 = [ (r.get("val",{}) or {}).get("f1", None) for r in recs ]
        if any(a is not None for a in val_f1):
            plt.figure()
            plt.plot(rounds, val_f1, marker="o")
            plt.xlabel("Round"); plt.ylabel("Val F1 (macro)")
            plt.title(f"Client {self.cid} - Validation F1 (macro)")
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir/f"client_{self.cid}_val_f1.png", dpi=160, bbox_inches="tight")
            plt.close()

        # 3) 训练损失
        tr_cls = [ r.get("train_cls_loss", None) for r in recs ]
        tr_gen = [ r.get("train_gen_loss", None) for r in recs ]
        if any(v is not None for v in tr_cls) or any(v is not None for v in tr_gen):
            plt.figure()
            if any(v is not None for v in tr_cls):
                plt.plot(rounds, tr_cls, marker="o", label="train_cls_loss")
            if any(v is not None for v in tr_gen):
                plt.plot(rounds, tr_gen, marker="o", label="train_gen_loss")
            plt.xlabel("Round"); plt.ylabel("Loss")
            plt.title(f"Client {self.cid} - Training Loss (per-round mean)")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.savefig(out_dir/f"client_{self.cid}_train_losses.png", dpi=160, bbox_inches="tight")
            plt.close()

        # 4) 分发计数（可选）
        disp_flow = [ (r.get("dispatch",{}) or {}).get("flow", None) for r in recs ]
        disp_temp = [ (r.get("dispatch",{}) or {}).get("temporal", None) for r in recs ]
        disp_graph= [ (r.get("dispatch",{}) or {}).get("graph", None) for r in recs ]
        if any(v not in (None,"") for v in disp_flow+disp_temp+disp_graph):
            plt.figure()
            if any(v not in (None,"") for v in disp_flow):
                plt.plot(rounds, disp_flow, marker="o", label="flow")
            if any(v not in (None,"") for v in disp_temp):
                plt.plot(rounds, disp_temp, marker="o", label="temporal")
            if any(v not in (None,"") for v in disp_graph):
                plt.plot(rounds, disp_graph, marker="o", label="graph")
            plt.xlabel("Round"); plt.ylabel("Dispatched samples")
            plt.title(f"Client {self.cid} - Dispatch Counts per Round")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.savefig(out_dir/f"client_{self.cid}_dispatch.png", dpi=160, bbox_inches="tight")
            plt.close()

        # 5) 异构代理子集 acc（可选）
        acc_flow = [ (r.get("hetero",{}) or {}).get("acc_flow", None) for r in recs ]
        acc_temp = [ (r.get("hetero",{}) or {}).get("acc_temporal", None) for r in recs ]
        acc_graph= [ (r.get("hetero",{}) or {}).get("acc_graph", None) for r in recs ]
        if any(v not in (None,"") for v in acc_flow+acc_temp+acc_graph):
            plt.figure()
            if any(v not in (None,"") for v in acc_flow):
                plt.plot(rounds, acc_flow, marker="o", label="flow")
            if any(v not in (None,"") for v in acc_temp):
                plt.plot(rounds, acc_temp, marker="o", label="temporal")
            if any(v not in (None,"") for v in acc_graph):
                plt.plot(rounds, acc_graph, marker="o", label="graph")
            plt.xlabel("Round"); plt.ylabel("Hetero Acc on cached subsets")
            plt.title(f"Client {self.cid} - Hetero Agents Acc")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.savefig(out_dir/f"client_{self.cid}_hetero_acc.png", dpi=160, bbox_inches="tight")
            plt.close()
