#!/usr/bin/env python
import numpy as np
from pathlib import Path

# ======= 配置部分 =======
ROOT = Path("/home/yyc/data/feddiffusion")   # 换成你 run.py 里传给 Client 的 root_dir
ROUND = 50                                    # 想看第几轮就写第几轮
N_CLIENTS = 6                                 # 客户端总数（0~5）

# 用端到端的混淆矩阵；如想看主分类器，把 val_e2e 换成 val
FNAME = "cm_round_{round}_val_e2e.npy".format(round=ROUND)

cm_list = []
dims = []

# 先把所有 client 的 cm 读出来，记录维度
for cid in range(N_CLIENTS):
    cm_path = ROOT / "outs" / "cm" / f"client_{cid:02d}" / FNAME
    if not cm_path.exists():
        print(f"[WARN] client_{cid:02d} missing {cm_path.name}, skip.")
        continue

    cm = np.load(cm_path)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        print(f"[WARN] client_{cid:02d} cm shape {cm.shape} is not square, skip.")
        continue

    cm_list.append((cid, cm.astype(np.int64)))
    dims.append(cm.shape[0])

if not cm_list:
    print(f"[ERROR] no confusion matrices found for round {ROUND}.")
    raise SystemExit(1)

max_dim = max(dims)
print(f"[INFO] max_dim = {max_dim}, client dims = {dims}")

# ======= 把所有 cm padding 到 (max_dim, max_dim) 再求和 =======
cm_sum = np.zeros((max_dim, max_dim), dtype=np.int64)

for cid, cm in cm_list:
    d = cm.shape[0]
    if d < max_dim:
        # 在右侧和下方补 0，使得左上角对齐
        pad_width = ((0, max_dim - d), (0, max_dim - d))
        cm_padded = np.pad(cm, pad_width, mode="constant", constant_values=0)
    else:
        cm_padded = cm
    cm_sum += cm_padded

cm = cm_sum
total = cm.sum()
print(f"[INFO] total samples across all clients (val_e2e) = {int(total)}")

# ======= 只对“真的出现过的类别”算指标（避免全 0 类别干扰）=======
eps = 1e-9
row_sum = cm.sum(axis=1)
col_sum = cm.sum(axis=0)
used_mask = (row_sum + col_sum) > 0

if used_mask.sum() == 0:
    print("[ERROR] no active classes found in confusion matrix.")
    raise SystemExit(1)

cm_used = cm[np.ix_(used_mask, used_mask)]

tp = np.diag(cm_used).astype(np.float64)
fp = cm_used.sum(axis=0).astype(np.float64) - tp
fn = cm_used.sum(axis=1).astype(np.float64) - tp

acc = tp.sum() / max(1.0, cm_used.sum())
precision = np.mean(tp / (tp + fp + eps))
recall    = np.mean(tp / (tp + fn + eps))
f1_macro  = 0.0 if (precision + recall) < eps else (2 * precision * recall / (precision + recall))

print(
    f"[Global] R{ROUND:03d} | E2E(val): "
    f"acc={acc:.4f} precision_macro={precision:.4f} "
    f"recall_macro={recall:.4f} f1_macro={f1_macro:.4f} "
    f"(n_classes_used={int(used_mask.sum())}, total_n={int(cm_used.sum())})"
)
