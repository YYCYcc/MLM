# utils/packager.py
import numpy as np

def temporal_from_tabular(X: np.ndarray, y: np.ndarray, win: int = 8, stride: int = 4):
    """
    将 [N, D] 的样本序列按时间顺序打成滑窗 [B, T, D]。
    假设 X 已按时间排序（同类/同主机可以先行分组再调用）。
    标签用“窗口内众数”或窗口最后一个样本的标签（这里取最后一个）。
    """
    assert X.ndim == 2
    N, D = X.shape
    outs, ys = [], []
    for s in range(0, max(0, N - win + 1), stride):
        seg = X[s:s + win]
        if len(seg) < win:
            break
        outs.append(seg)
        ys.append(y[s + win - 1])
    if not outs:
        return np.zeros((0, win, D), np.float32), np.zeros((0,), np.int64)
    return np.asarray(outs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def freq_from_temporal(X_seq: np.ndarray, k_keep: int = 16):
    """
    [B, T, D] -> 沿时间维做 rFFT，取前 k_keep 幅值，输出 [B, D, K]
    """
    assert X_seq.ndim == 3
    B, T, D = X_seq.shape
    fft = np.fft.rfft(X_seq, axis=1)       # [B, T_fft, D]
    mag = np.abs(fft)[:, :k_keep, :]       # [B, K, D]
    return np.transpose(mag, (0, 2, 1)).astype(np.float32)  # [B, D, K]


def graph_from_tabular(X: np.ndarray, y: np.ndarray, max_nodes=2000, k=8, chunk=1000):
    """
    将 [N,D] 切块构建 kNN 图，每块生成一个 pyg Data：
    - 节点特征: x
    - 图级标签: y_graph（窗口内众数/最后一个标签；这里用众数）
    - 边: 对每个点连向 k 个最近邻（用余弦相似）
    返回: List[pyg.data.Data]；未安装 pyg 时抛出 RuntimeError。
    """
    try:
        import torch
        from torch_geometric.data import Data
    except Exception:
        raise RuntimeError("graph_from_tabular 需要 torch-geometric")

    def knn_cosine_block(Xb, kk):
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-8)
        sim = Xb @ Xb.T
        np.fill_diagonal(sim, -np.inf)
        idx = np.argpartition(-sim, kth=min(kk, sim.shape[1]-1), axis=1)[:, :kk]  # [Nb, k]
        rows = np.repeat(np.arange(len(Xb))[:, None], idx.shape[1], axis=1).reshape(-1)
        cols = idx.reshape(-1)
        edge_index = np.stack([rows, cols], axis=0)  # [2, E]
        return edge_index

    outs = []
    N = len(X)
    if N == 0:
        return outs
    step = int(chunk)
    for s in range(0, N, step):
        Xe = X[s:s+step]
        Ye = y[s:s+step]
        if len(Xe) == 0:
            continue
        Xe = Xe[:max_nodes]; Ye = Ye[:max_nodes]
        edge_index = knn_cosine_block(Xe, k)
        y_graph = np.bincount(Ye).argmax()
        data = Data(
            x=torch.tensor(Xe, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor([int(y_graph)], dtype=torch.long)
        )
        outs.append(data)
    return outs


def flatten_for_gate(X):
    """
    将任意视图矩阵/张量展平至 [B, F] 供 KS/MMD 快速门控。
    支持 [B, D], [B, T, D], [B, D, K] 等。
    """
    X = np.asarray(X)
    if X.ndim == 2:
        return X.astype(np.float32)
    B = X.shape[0]
    return X.reshape(B, -1).astype(np.float32)
