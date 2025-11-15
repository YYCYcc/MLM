# agents/hetero.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .flow_mlp import FlowMLP

# 可选：如果你还没提供这些文件，会自动回退到 MLP
try:
    from .temporal_tcn import TemporalTCN  # 期望输入 [N, T, F]（如你的实现不同，请在模型内部转置）
except Exception:
    TemporalTCN = None

try:
    from .graph_gnn import GraphGNN  # 若你实现了图模型，这里会用；否则回退到 MLP
except Exception:
    GraphGNN = None


# ------------- 小工具 -------------
def _bucket_name(schema: str) -> str:
    s = str(schema).lower()
    if s.startswith("temporal"):
        return "temporal"
    if s.startswith("graph"):
        return "graph"
    if s in ("raw", "flow"):
        return "flow"
    return "flow"

def _to_tensor(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)

def _flatten2d(x: np.ndarray) -> np.ndarray:
    """将任意 [N, ...] 展平成 [N, D]（保持 N 不变）"""
    if x.ndim == 2:
        return x
    n = x.shape[0]
    return x.reshape(n, -1)

def _pad_trunc_lastdim(x: np.ndarray, target_d: int) -> np.ndarray:
    """二维 [N, D] 在最后一维对齐到 target_d：D>target 截断；D<target 右侧零填充。"""
    assert x.ndim == 2, f"expect 2D, got {x.shape}"
    n, d = x.shape
    if d == target_d:
        return x
    if d > target_d:
        return x[:, :target_d]
    pad = np.zeros((n, target_d - d), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


# ------------- 主类 -------------
class HeteroManager:
    """
    管理 flow / temporal / graph 三类异构代理：
      - attach_flow(main_agent): 绑定主分类器作为 flow 分支（共享权重）
      - inject(bucket, X_np, y_np): 按桶注入缓存
      - train_one_round(...): 在缓存子集做一小轮近线训练（temporal/graph）
      - eval_on_buffers(): 在缓存子集上快速评测 acc
      - predict_bucket(bucket, X_tensor): 单桶前向（供 E2E 置信度路由使用）
    同时暴露 expected_dims 供上层做维度安全检查；内部亦做了对齐，避免拼接报错。
    """

    # ---------- 构造 ----------
    def __init__(self, device, n_classes, x_dim, tlist_len=3, lr=1e-3, class_weights=None):
        self.device = torch.device(device)
        self.n_classes = int(n_classes)
        self.x_dim = int(x_dim)
        self.tlist_len = int(tlist_len)
        self.lr = float(lr)

        # 期望维度表（上层 client._can_inject 会用到“精确 schema 名”）
        # 预注册：raw、temporal_50/250/450、graph_50/250/450（维度默认 x_dim * tlist_len = 204）
        self.expected_dims = {
            "raw": self.x_dim,
            "temporal": self.x_dim * self.tlist_len,
            "graph":    self.x_dim * self.tlist_len,
            "temporal_50":  self.x_dim * self.tlist_len,
            "temporal_250": self.x_dim * self.tlist_len,
            "temporal_450": self.x_dim * self.tlist_len,
            "graph_50":     self.x_dim * self.tlist_len,
            "graph_250":    self.x_dim * self.tlist_len,
            "graph_450":    self.x_dim * self.tlist_len,
        }

        # flow（与主分类器共享）
        self.flow: nn.Module | None = None
        self.opt_flow = None  # flow 通常由外部训练，这里可不再单独优化
        self._flow_in_dim = None

        # temporal
        self.temporal: nn.Module | None = None
        self.opt_temporal = None
        self._temporal_use_tcn = False
        self._temporal_shape = None  # (T, F) 或 2D in_dim
        self._flow_shared = False
        self._flow_in_dim = None
        # graph
        self.graph: nn.Module | None = None
        self.opt_graph = None
        self._graph_in_dim = None

        # 类别权重
        self.class_weights = class_weights.to(self.device) if torch.is_tensor(class_weights) else None

        # 缓冲：各桶 list[(X_np, y_np), ...]
        self._cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "flow": [], "temporal": [], "graph": []
        }

    # ---------- 期望维度登记（可选） ----------
    def register_schema_dim(self, schema: str, dim: int):
        """当你见到新的视图名时，调用登记其期望维度，方便上层做严格注入判断。"""
        self.expected_dims[str(schema)] = int(dim)

    # ---------- 绑定主分类器 ----------
    def attach_flow(self, main_agent: nn.Module):
        self.flow = main_agent
        self._flow_shared = True
        # 设一下当前输入维度，避免 _ensure_model 认为没初始化
        self._flow_in_dim = getattr(self, "x_dim", None)
        # 给共享模型也配一个优化器，便于近线小训练（如果你不想改主模型，可把这行去掉）
        self.opt_flow = torch.optim.Adam(self.flow.parameters(), lr=self.lr)

    # ---------- 推理（单桶） ----------
    @torch.no_grad()
    def predict_bucket(self, bucket: str, X: torch.Tensor) -> torch.Tensor:
        """
        对某个桶的数据 X 前向（X 已是该桶所需视图形状/维度）。
        返回 logits，形状 [N, n_classes]。
        """
        bucket = bucket.lower()
        if bucket == "flow":
            if self.flow is None:
                raise RuntimeError("HeteroManager.flow not attached. Call attach_flow(main_agent) first.")
            return self.flow(X)
        elif bucket == "temporal":
            if self.temporal is None:
                # 没有 temporal 模型时：回退为恒等 MLP（按当前输入维度）
                in_dim = X.shape[1] if X.ndim == 2 else (X.shape[1] * X.shape[2])
                self.temporal = FlowMLP(int(in_dim), self.n_classes).to(self.device)
            return self.temporal(X)
        elif bucket == "graph":
            if self.graph is None:
                in_dim = X.shape[1] if X.ndim == 2 else (X.shape[1] * X.shape[2])
                self.graph = FlowMLP(int(in_dim), self.n_classes).to(self.device)
            return self.graph(X)
        else:
            raise ValueError(f"unknown bucket: {bucket}")

    # ---------- 模型状态 ----------
    def eval(self):
        for m in (self.flow, self.temporal, self.graph):
            if hasattr(m, "eval") and (m is not None):
                m.eval()

    # ---------- 注入 ----------
    def inject(self, bucket_or_schema: str, X_np, y_np):
        """
        上层通常传入 bucket（'flow'/'temporal'/'graph'）。
        若传入 schema 名（如 'temporal_250'），也能自动归一到对应桶。
        """
        bucket = _bucket_name(bucket_or_schema)
        if not isinstance(X_np, np.ndarray) or not isinstance(y_np, np.ndarray):
            return
        if X_np.shape[0] == 0 or y_np.shape[0] == 0:
            return
        n = min(len(X_np), len(y_np.reshape(-1)))
        if n <= 0:
            return
        self._cache[bucket].append((X_np[:n], y_np[:n].reshape(-1)))

    # ---------- 近线小训练 ----------
    def train_one_round(self, batch_size=256, epochs=1, focal=False, gamma=2.0):
        """
        对各桶的缓存做一小轮训练。
        - flow：默认**不重复训练**（它已由主分类器在主循环里训练）。如需训练，可放开注释。
        - temporal/graph：使用 TCN/GNN（若提供），否则回退到 MLP。
        """
        log = {}
        for bucket in ("flow", "temporal", "graph"):
            X_np, y_np = self._gather_and_align(bucket)
            if X_np is None:
                log[bucket] = {"loss": 0.0, "acc": 0.0, "n": 0}
                continue

            model, opt = self._ensure_model(bucket, X_np)
            # flow 分支通常与主分类器共享，这里不再重复训练，避免干扰主优化器
            if bucket == "flow":
                # 如需训练 flow，请把下面两行注释删掉，改成与 temporal/graph 同步训练
                model.eval()
                with torch.no_grad():
                    x_t = self._prep_input_for(bucket, X_np)
                    y_t = _to_tensor(y_np, dtype=torch.long, device=self.device)
                    logits = model(x_t)
                    pred = logits.argmax(dim=1)
                    acc = float((pred == y_t).float().mean().detach().cpu())
                    log[bucket] = {"loss": 0.0, "acc": acc, "n": int(y_t.numel())}
                continue

            # 训练 temporal / graph
            model.train()
            y_t = _to_tensor(y_np, dtype=torch.long, device=self.device)
            x_t = self._prep_input_for(bucket, X_np)

            ds = TensorDataset(x_t, y_t)
            dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

            total_loss, correct, total = 0.0, 0, 0
            for _ in range(max(1, int(epochs))):
                for xb, yb in dl:
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    if focal:
                        ce = F.cross_entropy(logits, yb, reduction="none",
                                             weight=self.class_weights)
                        pt = torch.softmax(logits, dim=1).gather(1, yb.view(-1, 1)).squeeze(1)
                        loss = ((1 - pt) ** gamma * ce).mean()
                    else:
                        loss = F.cross_entropy(logits, yb, weight=self.class_weights)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                    total_loss += float(loss.detach().cpu())
                    pred = logits.argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    total   += int(yb.size(0))

            if total == 0:
                log[bucket] = {"loss": 0.0, "acc": 0.0, "n": 0}
            else:
                log[bucket] = {"loss": total_loss / max(1, len(dl)),
                               "acc": correct / total, "n": total}

        # 可选：训练后清空缓存，避免持续膨胀
        for k in self._cache:
            self._cache[k].clear()

        return log

    # ---------- 在缓存子集上评测 ----------
    def eval_on_buffers(self):
        res = {}
        for bucket in ("flow", "temporal", "graph"):
            X_np, y_np = self._gather_and_align(bucket)
            if X_np is None:
                res[bucket] = {"acc": 0.0, "n": 0}
                continue

            model, _ = self._ensure_model(bucket, X_np)
            model.eval()
            with torch.no_grad():
                x_t = self._prep_input_for(bucket, X_np)
                y_t = _to_tensor(y_np, dtype=torch.long, device=self.device)
                logits = model(x_t)
                pred = logits.argmax(dim=1)
                acc = float((pred == y_t).float().mean().detach().cpu())
                res[bucket] = {"acc": acc, "n": int(y_t.numel())}
        return res

    # ---------- 内部：聚合并对齐 ----------
    def _gather_and_align(self, bucket: str):
        """
        聚合该桶缓存，并**对齐形状**：
          - temporal：若全部 3D 且 (T,F) 一致 => 直接 3D 拼接；否则回退 2D 对齐
          - flow / graph：展平为 2D，然后按该桶“最大 D”做右侧零填充或截断
        返回对齐后的 (X_np, y_np)；若无数据返回 (None, None)。
        """
        buf = self._cache.get(bucket, [])
        if not buf:
            return None, None

        Xs, Ys = [], []
        for X, y in buf:
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                continue
            if len(X) == 0 or len(y) == 0:
                continue
            n = min(len(X), len(y))
            Xs.append(X[:n])
            Ys.append(y[:n].reshape(-1))

        if not Xs:
            return None, None

        if bucket == "temporal":
            all_3d = all(x.ndim == 3 for x in Xs)
            if all_3d:
                t_f_set = {(x.shape[1], x.shape[2]) for x in Xs}
                if len(t_f_set) == 1:
                    X_np = np.concatenate(Xs, axis=0)
                    y_np = np.concatenate(Ys, axis=0)
                    return X_np, y_np
                # 否则回退 2D

        # flow / graph 或 temporal 的回退：统一 2D + 对齐
        Xs_2d = [_flatten2d(x) for x in Xs]
        max_d = max(x.shape[1] for x in Xs_2d)
        Xs_aligned = [_pad_trunc_lastdim(x, max_d) for x in Xs_2d]
        X_np = np.concatenate(Xs_aligned, axis=0)
        y_np = np.concatenate(Ys, axis=0)
        return X_np, y_np

    # ---------- 内部：按形状确保/重建模型 ----------
    def _ensure_model(self, bucket: str, X_np: np.ndarray):
        """
        根据当前对齐后的 X_np 形状，确保该桶模型存在且输入维度匹配。
        - flow：X_np 为 2D => FlowMLP(in_dim)；若已 attach_flow，则直接使用共享模型
        - temporal：
            * 3D -> 若有 TemporalTCN，用它；否则回退 2D MLP
            * 2D -> FlowMLP(in_dim)
        - graph：默认 2D FlowMLP(in_dim)；如你实现了 GraphGNN，可在此替换
        """
        bucket = bucket.lower()
        if bucket == "flow":
            in_dim = int(_flatten2d(X_np).shape[1])
            # 如果已经共享了主模型，就不要再重建；直接复用
            if self._flow_shared and (self.flow is not None):
                # 确保记录的 in_dim 更新一下，防止后续误触发重建
                self._flow_in_dim = in_dim
                return self.flow, self.opt_flow
            # 否则按老逻辑创建/重建
            if (self.flow is None) or (self._flow_in_dim != in_dim):
                self.flow = FlowMLP(in_dim, self.n_classes).to(self.device)
                self.opt_flow = torch.optim.Adam(self.flow.parameters(), lr=self.lr)
                self._flow_in_dim = in_dim
            return self.flow, self.opt_flow
        if bucket == "temporal":
            if X_np.ndim == 3 and TemporalTCN is not None:
                T, Fdim = int(X_np.shape[1]), int(X_np.shape[2])
                need_rebuild = (
                    self.temporal is None or
                    (not self._temporal_use_tcn) or
                    (self._temporal_shape != (T, Fdim))
                )
                if need_rebuild:
                    self.temporal = TemporalTCN(in_channels=Fdim, n_classes=self.n_classes).to(self.device)
                    self.opt_temporal = torch.optim.Adam(self.temporal.parameters(), lr=self.lr)
                    self._temporal_use_tcn = True
                    self._temporal_shape = (T, Fdim)
                return self.temporal, self.opt_temporal
            # 回退 2D
            X2 = _flatten2d(X_np)
            in_dim = int(X2.shape[1])
            need_rebuild = (
                self.temporal is None or
                self._temporal_use_tcn or
                (self._temporal_shape != in_dim)
            )
            if need_rebuild:
                self.temporal = FlowMLP(in_dim, self.n_classes).to(self.device)
                self.opt_temporal = torch.optim.Adam(self.temporal.parameters(), lr=self.lr)
                self._temporal_use_tcn = False
                self._temporal_shape = in_dim
            return self.temporal, self.opt_temporal

        if bucket == "graph":
            # 这里默认展平 2D；若你已实现 GraphGNN，可替换为图构建逻辑
            X2 = _flatten2d(X_np)
            in_dim = int(X2.shape[1])
            if GraphGNN is not None and isinstance(self.graph, GraphGNN):
                return self.graph, self.opt_graph
            if (self.graph is None) or (self._graph_in_dim != in_dim) or (GraphGNN is not None and not isinstance(self.graph, GraphGNN)):
                # 回退到 MLP（或你可在此改为 GraphGNN 的构建）
                self.graph = FlowMLP(in_dim, self.n_classes).to(self.device)
                self.opt_graph = torch.optim.Adam(self.graph.parameters(), lr=self.lr)
                self._graph_in_dim = in_dim
            return self.graph, self.opt_graph

        # 默认回 flow
        return self._ensure_model("flow", X_np)

    # ---------- 内部：准备张量输入 ----------
    def _prep_input_for(self, bucket: str, X_np: np.ndarray) -> torch.Tensor:
        """将 numpy => torch.Tensor，并适配该桶模型的输入形状。"""
        bucket = bucket.lower()
        if bucket == "temporal" and (X_np.ndim == 3) and self._temporal_use_tcn:
            # 你的 TemporalTCN 若期望 [N, C, L]，请在 TemporalTCN 内部自行转置
            return _to_tensor(X_np, dtype=torch.float32, device=self.device)
        # 其余一律展平 2D
        X2 = _flatten2d(X_np)
        return _to_tensor(X2, dtype=torch.float32, device=self.device)
