import argparse, os, math, json, random
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Utilities
# ---------------------------

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def quantile_clip(X: np.ndarray, low=0.01, high=0.99):
    lo = np.quantile(X, low, axis=0)
    hi = np.quantile(X, high, axis=0)
    return np.clip(X, lo, hi), lo, hi

def quantile_mask(X: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    m1 = (X >= lo).all(axis=1)
    m2 = (X <= hi).all(axis=1)
    return (m1 & m2)

# ---------------------------
# Diffusion schedule
# ---------------------------

@dataclass
class DDPMCfg:
    steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02

class Diffusion:
    def __init__(self, cfg: DDPMCfg, device):
        self.device = device
        self.cfg = cfg
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """ x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps """
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

    def p_sample(self, model, x_t, t, cond=None):
        """ one reverse step """
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1)
        alphas_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)

        eps_theta = model(x_t, t, cond)
        # x0_pred via predicted noise
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t + 1e-8)
        # mean of posterior q(x_{t-1} | x_t, x0)
        mean = torch.sqrt(self.alphas[t-1].view(-1,1)) * x0_pred + \
               torch.sqrt(1 - self.alphas[t-1].view(-1,1)) * eps_theta if (t>0).all() else x0_pred
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(self.betas[t]).view(-1,1)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, model, n, xdim, cond=None):
        device = self.device
        x_t = torch.randn(n, xdim, device=device)
        for t in reversed(range(self.cfg.steps)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, tt, cond)
        return x_t

# ---------------------------
# Time embedding & Model
# ---------------------------

class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(1) / 1000.0 * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb

class MLP(nn.Module):
    def __init__(self, xdim, cond_dim=0, hidden=256, tdim=128):
        super().__init__()
        self.tproj = nn.Sequential(SinusoidalTime(tdim), nn.Linear(tdim, hidden), nn.SiLU())
        in_dim = xdim + (cond_dim if cond_dim>0 else 0) + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, xdim)
        )
        self.cond_dim = cond_dim
        self.hidden = hidden
    def forward(self, x, t, cond=None):
        tfeat = self.tproj(t)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        x = torch.cat([x, tfeat], dim=1)
        return self.net(x)

# ---------------------------
# Data Loading
# ---------------------------

def load_csv_for_class(csv_path, label_col, minor_class):
    df = pd.read_csv(csv_path)
    assert label_col in df.columns, f"{label_col} not in CSV columns"
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])
    # 只用数值列
    X = X.select_dtypes(include=[np.number]).copy()
    # 取少数类数据
    X_minor = X[y == str(minor_class)].to_numpy()
    X_all = X.to_numpy()
    cols = X.columns.tolist()
    return X_all, X_minor, y, cols

def build_dataloaders(X_minor: np.ndarray, batch_size=512, device="cpu"):
    tensor = torch.tensor(X_minor, dtype=torch.float32, device=device)
    ds = TensorDataset(tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl

# ---------------------------
# kNN density filter
# ---------------------------

def knn_density_mask(X_ref: np.ndarray, X_gen: np.ndarray, k=10, q=0.20):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_ref)
    dist, _ = nbrs.kneighbors(X_gen)
    avg = dist.mean(axis=1)
    thr = np.quantile(avg, 1 - q)  # 过滤最稀薄的 q 部分
    return avg <= thr

# ---------------------------
# Training loop
# ---------------------------

def train_diffusion(model, diff, dl, epochs=10, steps=200, device="cpu"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        for (xb,) in dl:
            b = xb.size(0)
            t = torch.randint(0, steps, (b,), device=device, dtype=torch.long)
            x_t, noise = diff.q_sample(xb, t)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
        # 可选：打印
        # print(f"[ep {ep+1}/{epochs}] loss={loss.item():.4f}")

# ---------------------------
# Main
# ---------------------------

def main():
    set_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="train csv (only numeric features + label col)")
    ap.add_argument("--label-col", type=str, required=True)
    ap.add_argument("--minor-class", type=str, required=True, help="the class to augment (string match)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gen-num", type=int, default=10000)
    ap.add_argument("--out-aug", type=str, default="augmented_train.csv")
    ap.add_argument("--save-model", type=str, default="diffusion_minor.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) 载入数据
    X_all, X_minor, y_all, cols = load_csv_for_class(args.csv, args.label_col, args.minor_class)
    if X_minor.shape[0] < 500:
        print(f"[warn] Minor class only has {X_minor.shape[0]} samples; consider merging days or classes.")

    # 2) 标准化（在“少数类数据”上拟合；也可对全体特征拟合，这里选对少数类拟合，生成分布更贴近少数类）
    scaler = StandardScaler()
    X_minor_z = scaler.fit_transform(X_minor)
    X_all_z = scaler.transform(X_all)

    # 3) 分位数裁剪（拟合阈值，用于后续合成样本过滤）
    _, lo, hi = quantile_clip(X_minor_z, 0.01, 0.99)

    # 4) 训练扩散模型（无条件版；如需条件，可添加 one-hot 条件向量）
    device = torch.device(args.device)
    xdim = X_minor_z.shape[1]
    model = MLP(xdim=xdim, cond_dim=0, hidden=args.hidden).to(device)
    diff = Diffusion(DDPMCfg(steps=args.steps), device=device)

    dl = build_dataloaders(X_minor_z, batch_size=args.batch_size, device=device)
    train_diffusion(model, diff, dl, epochs=args.epochs, steps=args.steps, device=device)

    # 5) 采样生成
    model.eval()
    with torch.no_grad():
        Xg_z = diff.sample(model, n=args.gen_num, xdim=xdim).cpu().numpy()

    # 6) 质量门控：分位数裁剪 + kNN 密度过滤
    mask_q = quantile_mask(Xg_z, lo, hi)
    Xg_q = Xg_z[mask_q]
    if Xg_q.shape[0] == 0:
        print("[warn] All generated samples filtered by quantile mask; relax thresholds.")
        Xg_q = Xg_z

    # kNN 密度：参照真实少数类
    try:
        mask_knn = knn_density_mask(X_minor_z, Xg_q, k=10, q=0.20)
        Xg_keep = Xg_q[mask_knn]
    except Exception as e:
        print("[warn] kNN filter failed (likely too few samples); skipping.", e)
        Xg_keep = Xg_q

    # 反标准化到原特征尺度
    Xg = scaler.inverse_transform(Xg_keep)

    # 7) 拼接回训练数据（注意：仅用于训练集，不应合并到验证/测试）
    df_train = pd.read_csv(args.csv)
    yname = args.label_col

    # 只保留数值列顺序与训练一致
    X_cols = df_train.drop(columns=[yname]).select_dtypes(include=[np.number]).columns.tolist()
    df_gen = pd.DataFrame(Xg, columns=X_cols)
    df_gen[yname] = str(args.minor_class)

    # 将生成样本与原训练集（原表）按数值列对齐后拼接
    # 非数值列若需要，可另行还原或补充，这里仅用于“特征表训练”的常见流程
    base = df_train[X_cols + [yname]].copy()
    df_out = pd.concat([base, df_gen], axis=0, ignore_index=True)

    # 8) 保存
    df_out.to_csv(args.out_aug, index=False)
    torch.save({"model": model.state_dict(),
                "cfg": vars(args),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "cols": X_cols}, args.save_model)

    print(f"[done] Generated {len(df_gen)} samples for class='{args.minor_class}'.")
    print(f"[save] Augmented train CSV -> {args.out_aug}")
    print(f"[save] Diffusion model ckpt -> {args.save_model}")
    print(f"[info] Columns used: {len(X_cols)} numeric features.")

if __name__ == "__main__":
    main()
