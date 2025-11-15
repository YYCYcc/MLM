# gate/quality_gate.py
import numpy as np

def _clean_matrix(X):
    """将任意输入转成 float32 的二维数组，并滤掉含 NaN/Inf 的行。"""
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        return None
    # 仅保留每行都有限的样本
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    # 依然可能为空
    return X

def fidelity_gate(Xs, Xr, ks_thr=0.9, mmd_thr=0.3, max_eval=2048, random_state=2024):
    """
    对合成样本 Xs 与真实样本 Xr 做快速一致性校验。
    - 先做强力清洗：去掉 NaN/Inf 行；若为空直接返回 reason。
    - 样本过多时做子采样，避免过慢。
    - ks/mmd 任意一个未达标则返回 False。
    """
    rng = np.random.default_rng(random_state)

    Xs = _clean_matrix(Xs)
    Xr = _clean_matrix(Xr)

    if Xs is None or Xr is None or len(Xs) == 0 or len(Xr) == 0:
        return {"ks": False, "ks_val": None, "mmd": False, "mmd_val": None,
                "reason": "empty_or_nan_batch"}, False

    # 子采样，控制计算量
    if len(Xs) > max_eval:
        Xs = Xs[rng.choice(len(Xs), size=max_eval, replace=False)]
    if len(Xr) > max_eval:
        Xr = Xr[rng.choice(len(Xr), size=max_eval, replace=False)]

    # 简化的 KS：按每列的 KS 统计取均值（或中位数）
    try:
        from scipy.stats import ks_2samp
        ks_vals = []
        D = min(Xs.shape[1], Xr.shape[1])
        for j in range(D):
            v = ks_2samp(Xs[:, j], Xr[:, j], alternative="two-sided", mode="auto").statistic
            ks_vals.append(float(v))
        ks_val = float(np.mean(ks_vals)) if ks_vals else None
    except Exception:
        ks_val = None  # 没装 scipy 也能继续

    # 简化的 MMD（RBF 核）
    try:
        def _rbf(X, Y, gamma=None):
            if gamma is None:
                # 用中位数启发式估一个带宽
                XY = np.vstack([X, Y])
                # 避免 O(n^2) 的完整距离矩阵，取一个小子集估计
                sub = XY[rng.choice(len(XY), size=min(1024, len(XY)), replace=False)]
                pdist = np.sum((sub[:, None, :] - sub[None, :, :])**2, axis=-1)
                med2 = np.median(pdist[pdist > 0])
                gamma = 1.0 / (2.0 * (med2 + 1e-6))
            XX = np.exp(-gamma * ((X[:, None, :] - X[None, :, :])**2).sum(-1))
            YY = np.exp(-gamma * ((Y[:, None, :] - Y[None, :, :])**2).sum(-1))
            XY = np.exp(-gamma * ((X[:, None, :] - Y[None, :, :])**2).sum(-1))
            return XX, YY, XY

        XX, YY, XY = _rbf(Xs, Xr)
        mmd_val = float(XX.mean() + YY.mean() - 2.0 * XY.mean())
    except Exception:
        mmd_val = None

    ok_ks  = (ks_val is not None)  and (ks_val  <= ks_thr)  # KS 距离越小越相近
    ok_mmd = (mmd_val is not None) and (mmd_val <= mmd_thr) # MMD 越小越相近
    ok = (ok_ks and ok_mmd) if (ks_val is not None and mmd_val is not None) else False

    log = {"ks": ok_ks, "ks_val": ks_val, "mmd": ok_mmd, "mmd_val": mmd_val}
    return log, ok
