# utils/dataset.py
import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_path, label_col="Label", standardize=True, known_classes=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)

        # 1) 干掉 NaN/空标签
        mask = df[label_col].notna()
        if (~mask).sum() > 0:
            bad = sorted(set(df.loc[~mask, label_col].astype(str)))
            print(f"[WARN] drop {(~mask).sum()} rows with NaN/empty labels: {bad}")
        df = df[mask].reset_index(drop=True)

        # 2) 标签转字符串用于对齐
        y_str = df[label_col].astype(str).values

        # 3) 若提供 known_classes：过滤并映射；否则就用本地唯一值当词表
        if known_classes is None:
            classes = sorted(pd.unique(y_str).tolist())
        else:
            classes = list(known_classes)
            in_vocab = np.isin(y_str, classes)
            if (~in_vocab).sum() > 0:
                unk = sorted(set(y_str[~in_vocab].tolist()))
                print(f"[WARN] drop {(~in_vocab).sum()} rows with unknown labels: {unk}")
            y_str = y_str[in_vocab]
            df = df.loc[in_vocab].reset_index(drop=True)

        cls2id = {c: i for i, c in enumerate(classes)}
        y = np.array([cls2id[s] for s in y_str], dtype=np.int64)

        # 4) 特征：去标签列、float32、基础标准化
        X = df.drop(columns=[label_col]).astype(np.float32).values
        if standardize and len(X) > 0:
            mu = np.nanmean(X, axis=0, keepdims=True)
            std = np.nanstd(X, axis=0, keepdims=True) + 1e-6
            X = (X - mu) / std

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.classes = classes
        self.classes_ = classes     # 兼容老字段
        self.x_dim = X.shape[1] if X.ndim == 2 else 0

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_loaders(*, train_csv, val_csv, test_csv, label_col, batch_size, known_classes=None):
    """
    只接受关键字参数，避免位置参数错传。
    返回：tr_loader, va_loader, te_loader, classes, x_dim
    """
    # 先用训练集确定类别词表（若未显式提供）
    tr = CSVDataset(train_csv, label_col=label_col, standardize=True,
                    known_classes=known_classes)
    classes = tr.classes   # 训练集词表
    x_dim = tr.x_dim

    # 用训练集词表去约束 val/test，保证标签索引一致
    va = CSVDataset(val_csv,  label_col=label_col, standardize=True, known_classes=classes)
    te = CSVDataset(test_csv, label_col=label_col, standardize=True, known_classes=classes)

    # DataLoaders
    # 训练集如果非常小，drop_last=True 可能导致没有 batch；如有需要可改成 False
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  drop_last=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, drop_last=False)

    return tr_loader, va_loader, te_loader, classes, x_dim
