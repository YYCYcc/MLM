# agents/temporal_tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x:[B,C,T]
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y, inplace=True)
        y = self.drop(y)
        return y + self.proj(x)


class TemporalTCN(nn.Module):
    """
    输入 [B,T,D]，内部转为 [B,D,T] 做空洞卷积
    """
    def __init__(self, d_feat:int, n_classes:int, hid=128, depth=3, p=0.1):
        super().__init__()
        layers = []
        in_ch = d_feat
        for i in range(depth):
            layers += [_TCNBlock(in_ch, hid, d=2**i, p=p)]
            in_ch = hid
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hid, n_classes)

    def forward(self, x_btd):  # [B,T,D]
        x = x_btd.transpose(1, 2)      # [B,D,T]
        h = self.net(x)                # [B,H,T]
        h = h.mean(dim=-1)             # [B,H]
        return self.head(h)

    def step(self, xb_btd, yb, opt, class_weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = self(xb_btd)
        loss = F.cross_entropy(logits, yb, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        acc = (logits.argmax(1) == yb).float().mean().detach()
        return loss.detach(), acc
