import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

class TemporalTransformer(nn.Module):
    def __init__(self, d_feat:int, n_classes:int, d_model=128, nhead=4, depth=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model*4, dropout=dropout,
                                               activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
        self.pos  = nn.Parameter(torch.randn(512, d_model)*0.02)  # 支持 T<=512
    def forward(self, x_btd):          # [B,T,D]
        B,T,_ = x_btd.shape
        h = self.inp(x_btd) + self.pos[:T].unsqueeze(0)  # [B,T,dm]
        h = self.enc(h)
        h = h.mean(dim=1)                                  # 时序池化
        return self.head(h)
    def step(self, xb_btd, yb, opt, class_weights=None)->Tuple[torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = self(xb_btd)
        loss = F.cross_entropy(logits, yb, weight=class_weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),1.0); opt.step()
        acc = (logits.argmax(1)==yb).float().mean().detach()
        return loss.detach(), acc
