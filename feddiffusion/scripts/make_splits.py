# scripts/make_splits.py
import os, argparse, json
import numpy as np, pandas as pd
from collections import defaultdict

def dirichlet_partition(labels, n_clients=6, alpha=0.3, rare_labels=None, rare_clients=2):
    idx_by_cls = defaultdict(list)
    for i, y in enumerate(labels): idx_by_cls[y].append(i)
    client_indices = [[] for _ in range(n_clients)]
    for c, idxs in idx_by_cls.items():
        p = np.random.dirichlet([alpha]*n_clients)
        parts = (np.cumsum(p)*len(idxs)).astype(int)[:-1]
        splits = np.split(np.random.permutation(idxs), parts)
        for k in range(n_clients):
            client_indices[k].extend(splits[k].tolist())
    # 稀缺类只落到少数客户端
    if rare_labels:
        keep = set(np.random.choice(range(n_clients), size=rare_clients, replace=False))
        for k in range(n_clients):
            client_indices[k] = [i for i in client_indices[k] if labels[i] not in rare_labels or k in keep]
    return client_indices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--root", type=str, default="data/fed_splits")
    ap.add_argument("--label_col", type=str, default="Label")
    ap.add_argument("--n_clients", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--rare", type=str, default="")  # 逗号分隔
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    df = pd.read_csv(args.csv)
    labels = df[args.label_col].astype(str).values
    rare = set([s for s in args.rare.split(",") if s]) if args.rare else None

    parts = dirichlet_partition(labels, args.n_clients, args.alpha, rare_labels=rare)
    for cid, idxs in enumerate(parts):
        sub = df.iloc[idxs].sample(frac=1.0, random_state=42)
        n = len(sub); n_train = int(n*0.7); n_val = int(n*0.15)
        os.makedirs(f"{args.root}/client_{cid:02d}", exist_ok=True)
        sub.iloc[:n_train].to_csv(f"{args.root}/client_{cid:02d}/train.csv", index=False)
        sub.iloc[n_train:n_train+n_val].to_csv(f"{args.root}/client_{cid:02d}/val.csv", index=False)
        sub.iloc[n_train+n_val:].to_csv(f"{args.root}/client_{cid:02d}/test.csv", index=False)
    print(f"done: splits in {args.root}")

if __name__ == "__main__":
    main()
