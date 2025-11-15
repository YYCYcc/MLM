# agents/tab_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class _FeatureTokenizer(nn.Module):
    def __init__(self, d_feat:int, d_model:int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.randn(d_feat, d_model) * 0.02)

    def forward(self, x):  # x:[B,D]
        B, D = x.shape
        x = x.view(B, D, 1)
        tok = self.proj(x) + self.pos.view(1, D, -1)
        return tok


class TabTransformer(nn.Module):
    def __init__(self, d_feat:int, n_classes:int, d_model=128, nhead=4, depth=3, dropout=0.1):
        super().__init__()
        self.tok = _FeatureTokenizer(d_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model * 4, dropout=dropout,
                                               activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):  # [B,D]
        h = self.tok(x)     # [B,D,dm]
        h = self.enc(h)     # [B,D,dm]
        h = h.mean(dim=1)   # [B,dm]
        return self.head(h)

    def step(self, xb, yb, opt, class_weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = self(xb)
        loss = F.cross_entropy(logits, yb, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        acc = (logits.argmax(1) == yb).float().mean().detach()
        return loss.detach(), acc
