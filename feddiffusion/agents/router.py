# agents/router.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureRouter(nn.Module):
    """
    将原始向量 X（[N, x_dim]）按样本分发到 {flow, temporal, graph} 三个桶之一。
    - mode="heuristic": 使用简单可解释的规则（推荐先用）
    - mode="learned"  : 使用一个小 MLP 作为 gating（后续可训练）
    cfg.router 示例：
    {
      "mode": "heuristic",
      "temporal_cols": [20,21,22,23,24],          # 可空
      "graph_cols": {"src": 0, "dst": 1},         # 可空
      "thresholds": {"temporal_var": 0.15, "degree": 5, "sparsity": 0.8}
    }
    """
    def __init__(self, cfg: dict, x_dim: int, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        rcfg = cfg or {}
        self.mode = str(rcfg.get("mode", "heuristic")).lower()

        # 规则相关配置
        self.temporal_cols = list(rcfg.get("temporal_cols", []))
        self.graph_cols = rcfg.get("graph_cols", {}) or {}
        thr = rcfg.get("thresholds", {}) or {}
        self.t_thr = float(thr.get("temporal_var", 0.15))  # 方差阈值
        self.d_thr = int(thr.get("degree", 5))             # 批内度数阈值
        self.s_thr = float(thr.get("sparsity", 0.85))      # 稀疏度阈值（>则偏 Flow）

        # learned gating（可选）
        if self.mode == "learned":
            h = max(64, x_dim // 2)
            self.gate = nn.Sequential(
                nn.Linear(x_dim, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, 3)  # 0:flow, 1:temporal, 2:graph
            )
        else:
            self.gate = None

        self.to(self.device)

    @torch.no_grad()
    def route(self, X):
        """
        输入：
          X: np.ndarray 或 torch.Tensor，形状 [N, x_dim]
        输出：
          buckets: dict[str -> np.ndarray of idx]
            {"flow": idxs0, "temporal": idxs1, "graph": idxs2}
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)
        N = len(X_np)
        if N == 0:
            return {"flow": np.array([], int), "temporal": np.array([], int), "graph": np.array([], int)}

        if self.mode == "learned":
            return self._route_learned(X_np)
        else:
            return self._route_heuristic(X_np)

    def _route_learned(self, X_np: np.ndarray):
        x = torch.from_numpy(X_np).to(self.device)
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).cpu().numpy()
        flow_idx = np.where(pred == 0)[0]
        temp_idx = np.where(pred == 1)[0]
        graph_idx= np.where(pred == 2)[0]
        return {"flow": flow_idx, "temporal": temp_idx, "graph": graph_idx}

    def _route_heuristic(self, X_np: np.ndarray):
        N, D = X_np.shape
        # 1) 稀疏度：非零占比；越稀疏越倾向 Flow（规则可按需改）
        nnz = (np.abs(X_np) > 1e-8).sum(axis=1)
        sparsity = 1.0 - (nnz / max(1, D))  # 1.0 表示全 0，0 表示全非 0

        # 2) temporal 倾向：指定 temporal_cols 的跨列方差均值
        temp_score = np.zeros(N, dtype=np.float32)
        if self.temporal_cols:
            cols = np.array(self.temporal_cols, dtype=int)
            cols = cols[(cols >= 0) & (cols < D)]
            if len(cols) > 1:
                sub = X_np[:, cols]  # [N, T]
                # 跨列方差：捕捉“随时间维的波动”
                temp_score = np.var(sub, axis=1).astype(np.float32)

        # 3) graph 倾向：按批内的 src/dst “度数”近似。出现次数越多，越偏 graph。
        graph_score = np.zeros(N, dtype=np.float32)
        if "src" in self.graph_cols and "dst" in self.graph_cols:
            si = int(self.graph_cols["src"]); di = int(self.graph_cols["dst"])
            if 0 <= si < D and 0 <= di < D:
                src = X_np[:, si].astype(np.int64, copy=False)
                dst = X_np[:, di].astype(np.int64, copy=False)
                # 批内计数（离线简单近似）：次数作为度数
                uniq, cnt = np.unique(np.concatenate([src, dst]), return_counts=True)
                deg_map = {u: c for u, c in zip(uniq, cnt)}
                deg = np.vectorize(lambda u, v: deg_map.get(u, 0) + deg_map.get(v, 0))(src, dst)
                graph_score = deg.astype(np.float32)

        # 4) 规则决策（可按优先级覆盖）：
        #   - 若 graph_score >= d_thr → graph
        #   - 否则若 temp_score >= t_thr → temporal
        #   - 否则若 sparsity >= s_thr → flow
        #   - 其余 → flow（默认兜底）
        graph_mask = graph_score >= float(self.d_thr)
        temp_mask  = (~graph_mask) & (temp_score >= float(self.t_thr))
        flow_mask  = (~graph_mask) & (~temp_mask)

        flow_idx = np.where(flow_mask)[0]
        temp_idx = np.where(temp_mask)[0]
        graph_idx= np.where(graph_mask)[0]

        return {"flow": flow_idx, "temporal": temp_idx, "graph": graph_idx}
