# tools/paper_export.py
import os, re, json, argparse, glob, math, collections
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _client_id_from_path(p: str) -> int:
    m = re.search(r"client_(\d+)", p)
    return int(m.group(1)) if m else -1

def load_train_curves(logdir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(logdir, "client_*[0-9].csv")):
        cid = _client_id_from_path(fp)
        try:
            df = pd.read_csv(fp)
            df["client"] = cid
            rows.append(df)
        except Exception as e:
            print(f"[skip] {fp}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_eval(logdir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(logdir, "client_*_eval.csv")):
        cid = _client_id_from_path(fp)
        try:
            df = pd.read_csv(fp)
            df["client"] = cid
            # 字段确保是 float
            for k in ["loss", "acc", "precision_macro", "recall_macro", "f1_macro"]:
                df[k] = pd.to_numeric(df[k], errors="coerce")
            rows.append(df)
        except Exception as e:
            print(f"[skip] {fp}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_gate(logdir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(logdir, "client_*_gate.csv")):
        cid = _client_id_from_path(fp)
        try:
            df = pd.read_csv(fp)
            df["client"] = cid
            rows.append(df)
        except Exception as e:
            print(f"[skip] {fp}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_events(logdir: str) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(logdir, "client_*.events.jsonl")):
        cid = _client_id_from_path(fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        obj["client"] = cid
                        rows.append(obj)
                    except:
                        pass
        except Exception as e:
            print(f"[skip] {fp}: {e}")
    return pd.DataFrame(rows)

def summarize_curves(df: pd.DataFrame, outdir: str):
    if df.empty: 
        return
    # 全局按 step 的均值
    g = df.groupby("step").agg({"loss_cls":"mean","acc":"mean","loss_gen":"mean"}).reset_index()
    g.to_csv(os.path.join(outdir, "curves_overview.csv"), index=False)

    # 画图：loss_cls/acc/loss_gen
    plt.figure()
    for cid, sub in df.groupby("client"):
        plt.plot(sub["step"], sub["loss_cls"], alpha=0.3, label=f"C{cid}")
    plt.plot(g["step"], g["loss_cls"], linewidth=2.5, label="mean",)
    plt.xlabel("step"); plt.ylabel("loss_cls"); plt.title("Classifier Loss vs. Step")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "curve_loss_cls.png")); plt.close()

    plt.figure()
    for cid, sub in df.groupby("client"):
        plt.plot(sub["step"], sub["acc"], alpha=0.3, label=f"C{cid}")
    plt.plot(g["step"], g["acc"], linewidth=2.5, label="mean",)
    plt.xlabel("step"); plt.ylabel("acc"); plt.title("Accuracy vs. Step")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "curve_acc.png")); plt.close()

    plt.figure()
    for cid, sub in df.groupby("client"):
        plt.plot(sub["step"], sub["loss_gen"], alpha=0.3, label=f"C{cid}")
    plt.plot(g["step"], g["loss_gen"], linewidth=2.5, label="mean",)
    plt.xlabel("step"); plt.ylabel("loss_gen"); plt.title("Generator Loss vs. Step")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "curve_loss_gen.png")); plt.close()

def summarize_eval(df: pd.DataFrame, outdir: str):
    if df.empty: 
        return
    # 每轮/每split，计算均值与std
    m = df.groupby(["round","split"]).agg(
        loss_mean=("loss","mean"), loss_std=("loss","std"),
        acc_mean=("acc","mean"), acc_std=("acc","std"),
        f1_mean=("f1_macro","mean"), f1_std=("f1_macro","std"),
        p_mean=("precision_macro","mean"), r_mean=("recall_macro","mean")
    ).reset_index()
    m.to_csv(os.path.join(outdir, "eval_summary.csv"), index=False)

    # 最后一轮 test 的汇总（论文常用）
    last_round = df["round"].max()
    last_test = df[(df["round"]==last_round) & (df["split"]=="test")]
    if not last_test.empty:
        lt = last_test.agg({"loss":"mean","acc":"mean","f1_macro":"mean","precision_macro":"mean","recall_macro":"mean"}).to_dict()
        with open(os.path.join(outdir, "final_test_summary.json"), "w") as f:
            json.dump({"round": int(last_round), **{k:float(v) for k,v in lt.items()}}, f, indent=2)

    # 画 F1 vs round（val）
    val = m[m["split"]=="val"]
    plt.figure()
    plt.plot(val["round"], val["f1_mean"])
    plt.fill_between(val["round"], val["f1_mean"]-val["f1_std"].fillna(0),
                     val["f1_mean"]+val["f1_std"].fillna(0), alpha=0.2)
    plt.xlabel("round"); plt.ylabel("F1 (macro)"); plt.title("Validation F1 vs. Round")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "val_f1_vs_round.png")); plt.close()

def summarize_gate(df: pd.DataFrame, outdir: str):
    if df.empty:
        return
    df["ok"] = pd.to_numeric(df["ok"], errors="coerce").fillna(0).astype(int)
    df["injected_n"] = pd.to_numeric(df["injected_n"], errors="coerce").fillna(0).astype(int)

    # schema 总体通过率与总注入数
    g1 = df.groupby("schema").agg(
        pass_rate=("ok","mean"),
        injected_total=("injected_n","sum")
    ).reset_index().sort_values("injected_total", ascending=False)
    g1.to_csv(os.path.join(outdir, "gate_summary.csv"), index=False)

    # 每轮每 schema 注入量（堆叠图）
    g2 = df.groupby(["round","schema"])["injected_n"].sum().reset_index()
    pivot = g2.pivot(index="round", columns="schema", values="injected_n").fillna(0)
    pivot.to_csv(os.path.join(outdir, "inject_counts_by_schema.csv"))
    pivot.plot(kind="area", stacked=True, alpha=0.8)
    plt.xlabel("round"); plt.ylabel("# injected"); plt.title("Injected per Schema per Round")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "inject_stacked_area.png")); plt.close()

def summarize_events(ev: pd.DataFrame, outdir: str):
    if ev.empty: 
        return
    # 路由选择频次
    router = ev[ev["type"]=="router"].copy()
    rows = []
    for _, r in router.iterrows():
        sel = (r.get("payload") or {}).get("selected", [])
        if isinstance(sel, list):
            for s in sel:
                rows.append({"client": r["client"], "schema": s})
    if rows:
        df_sel = pd.DataFrame(rows)
        g = df_sel.groupby("schema").size().reset_index(name="count").sort_values("count", ascending=False)
        g.to_csv(os.path.join(outdir, "router_summary.csv"), index=False)
        # 图
        plt.figure()
        plt.bar(g["schema"], g["count"])
        plt.xlabel("schema"); plt.ylabel("count"); plt.title("Router Selections")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "router_bar.png")); plt.close()

    # 异构代理近线训练指标（来自 hetero 事件）
    hetero = ev[ev["type"]=="hetero"].copy()
    rows = []
    for _, r in hetero.iterrows():
        round_id = (r.get("payload") or {}).get("round", None)
        stats = (r.get("payload") or {}).get("stats", {})
        for k, v in stats.items():
            rows.append({
                "round": round_id, "schema": k,
                "loss": v.get("loss", None),
                "acc": v.get("acc", None),
                "n": v.get("n", None),
                "client": r["client"],
            })
    if rows:
        dfh = pd.DataFrame(rows)
        dfh.to_csv(os.path.join(outdir, "hetero_summary.csv"), index=False)

        g = dfh.groupby(["round","schema"]).agg(acc_mean=("acc","mean")).reset_index()
        for schema, sub in g.groupby("schema"):
            plt.figure()
            plt.plot(sub["round"], sub["acc_mean"])
            plt.xlabel("round"); plt.ylabel("acc"); plt.title(f"Hetero {schema} Nearline Acc")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, f"hetero_{schema}_acc.png")); plt.close()

def summarize_synth_counts(outs_dir: str, outdir: str):
    if not outs_dir or not os.path.isdir(outs_dir):
        return
    rows = []
    for client_dir in sorted(glob.glob(os.path.join(outs_dir, "client_*"))):
        cid = _client_id_from_path(client_dir)
        for round_dir in sorted(glob.glob(os.path.join(client_dir, "round_*"))):
            m = re.search(r"round_(\d+)", round_dir)
            rid = int(m.group(1)) if m else -1
            for schema_dir in glob.glob(os.path.join(round_dir, "*")):
                schema = os.path.basename(schema_dir)
                x_path = os.path.join(schema_dir, "X_synth.npy")
                if os.path.isfile(x_path):
                    try:
                        import numpy as np
                        n = int(np.load(x_path).shape[0])
                    except Exception:
                        n = None
                    rows.append({"client": cid, "round": rid, "schema": schema, "n": n})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, "synth_counts.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="./logs")
    ap.add_argument("--outs", type=str, default="./outs/synth")
    ap.add_argument("--out", type=str, default="./paper_export")
    args = ap.parse_args()

    _ensure_dir(args.out)

    df_train = load_train_curves(args.logdir)
    df_eval  = load_eval(args.logdir)
    df_gate  = load_gate(args.logdir)
    df_ev    = load_events(args.logdir)

    summarize_curves(df_train, args.out)
    summarize_eval(df_eval, args.out)
    summarize_gate(df_gate, args.out)
    summarize_events(df_ev, args.out)
    summarize_synth_counts(args.outs, args.out)

    print(f"[OK] exported to: {args.out}")

if __name__ == "__main__":
    main()
