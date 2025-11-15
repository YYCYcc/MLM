# agents/__init__.py
"""
Expose all agent classes and provide a small registry/factory.
"""

from .flow_mlp import FlowMLP

# 这些可能是可选模块；用 try/except 以避免在未放置文件时 ImportError 直接崩溃
try:
    from .temporal_tcn import TemporalTCN
except Exception:
    TemporalTCN = None  # 占位，便于在 registry 里做存在性检查

try:
    from .graph_gnn import GraphGNN
except Exception:
    GraphGNN = None

# 异构管理器（集中路由/训练/评测）
try:
    from .hetero import HeteroManager
except Exception:
    HeteroManager = None

# —— 注册表：名字 → 类 —— #
AGENT_REGISTRY = {
    "flow": FlowMLP,
    **({"temporal": TemporalTCN} if TemporalTCN is not None else {}),
    **({"graph": GraphGNN}       if GraphGNN   is not None else {}),
}

def create_agent(name: str, **kwargs):
    """
    Small factory: create an agent by name.
    Example:
        flow = create_agent("flow", in_dim=68, n_classes=7)
    """
    key = name.lower()
    if key not in AGENT_REGISTRY:
        raise KeyError(
            f"[agents] Unknown agent '{name}'. "
            f"Available: {sorted(AGENT_REGISTRY.keys())} "
            f"(did you add temporal_tcn.py / graph_gnn.py?)"
        )
    cls = AGENT_REGISTRY[key]
    return cls(**kwargs)

__all__ = [
    "FlowMLP",
    "TemporalTCN",
    "GraphGNN",
    "HeteroManager",
    "AGENT_REGISTRY",
    "create_agent",
]
