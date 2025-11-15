# utils/packager.py
import numpy as np
import torch

# ============ 时序窗口打包 ============
def temporal_from_tabular(X_np: np.ndarray,
                          y_np: np.ndarray,
                          win: int = 8,
                          stride: int = 4,
                          label_strategy: str = "last"):
    """
    把按时间排序的单流表格特征打成 [B, T, D] 的滑窗序列。
    - X_np: [N, D]
    - y_np: [N]
    - 返回: X_seq [M, T, D], y_seq [M]
    """
    assert X_np.ndim == 2 and y_np.ndim == 1
    N, D = X_np.shape
    if N < win:
        return np.empty((0, win, D), dtype=np.float32), np.empty((0,), dtype=np.int64)

    Xs, Ys = [], []
    for st in range(0, N - win + 1, stride):
        ed = st + win
        Xw = X_np[st:ed]
        yw = y_np[st:ed]
        if label_strategy == "last":
            y = int(yw[-1])
        elif label_strategy == "max":
            y = int(np.bincount(yw).argmax())
        else:
            y = int(yw[-1])
        Xs.append(Xw)
        Ys.append(y)
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ys, dtype=np.int64)


# ============ 图打包（基于 kNN） ============
def _knn_graph(x: torch.Tensor, k: int = 8):
    """
    基于余弦相似度的简易 kNN 图（无向），用于没有元数据的场景。
    x: [N, D]  返回 edge_index [2, E]
    """
    x = torch.nn.functional.normalize(x, dim=1)
    sims = x @ x.t()                     # [N, N]
    N = x.size(0)
    # 去除自环
    sims.fill_diagonal_(-1.0)
    # top-k 邻居
    idx = sims.topk(k=k, dim=1).indices  # [N, k]
    row = torch.arange(N, device=x.device).view(-1, 1).expand_as(idx)
    edge_index = torch.stack([row.reshape(-1), idx.reshape(-1)], dim=0)
    # 无向化
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # 去重
    edge_index = edge_index[:, edge_index[0] <= edge_index[1]]
    return edge_index


def graph_from_tabular(X_np: np.ndarray,
                       y_np: np.ndarray,
                       max_nodes: int = 2000,
                       k: int = 8,
                       chunk: int = 1000):
    """
    把单流表格特征打成若干 pyg.Data 图（分块避免过大）。
    这里把“每个块内的流”视为一张图（图级标签=块内多数标签）。
    返回: List[pyg.data.Data]
    """
    try:
        from torch_geometric.data import Data
    except Exception as e:
        raise RuntimeError("graph_from_tabular 需要 torch_geometric，请先安装。") from e

    N, D = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    datas = []
    st = 0
    while st < N:
        ed = min(N, st + min(max_nodes, chunk))
        x = X_t[st:ed]
        y_blk = y_t[st:ed]
        edge_index = _knn_graph(x, k=k)
        # 图级标签：多数票
        y_graph = torch.mode(y_blk).values.unsqueeze(0)  # [1]
        data = Data(x=x, edge_index=edge_index, y=y_graph)
        datas.append(data)
        st = ed
    return datas


# ============ flatten 工具（用于门控） ============
def flatten_for_gate(X_seq: np.ndarray) -> np.ndarray:
    """把 [B, T, D] 展平成 [B, T*D]，用于 fidelity_gate 的 2D 输入"""
    if X_seq.ndim != 3:
        raise ValueError("expect [B, T, D]")
    B, T, D = X_seq.shape
    return X_seq.reshape(B, T * D).astype(np.float32)
