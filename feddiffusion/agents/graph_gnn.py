# agents/graph_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


class GraphGCN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        if not _HAS_PYG:
            raise RuntimeError("GraphGCN 需要 torch-geometric，请先安装。")
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)] +
                                   [GCNConv(hidden, hidden) for _ in range(layers - 1)])
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, data):
        h = data.x
        for conv in self.convs:
            h = conv(h, data.edge_index)
            h = F.relu(h, inplace=True)
            h = self.do(h)
        g = global_mean_pool(h, data.batch)
        return self.head(g)

    def step(self, batch, opt, class_weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y.long(), weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        acc = (logits.argmax(1) == batch.y).float().mean().detach()
        return loss.detach(), acc
