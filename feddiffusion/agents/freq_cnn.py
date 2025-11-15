# agents/freq_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FreqCNN(nn.Module):
    """
    输入 [B, D, K]，沿频率轴做 1D 卷积；D=特征数，K=保留的频率 bins
    """
    def __init__(self, d_feat:int, n_classes:int, k_bins:int=16, width=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = d_feat
        for _ in range(depth):
            layers += [nn.Conv1d(in_ch, width, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            in_ch = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x_bdk):  # [B,D,K]
        h = self.net(x_bdk)    # [B,C,K]
        h = h.mean(dim=-1)     # [B,C]
        return self.head(h)

    def step(self, xb_bdk, yb, opt, class_weights=None) -> Tuple[torch.Tensor, torch.Tensor]:
        opt.zero_grad(set_to_none=True)
        logits = self(xb_bdk)
        loss = F.cross_entropy(logits, yb, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        acc = (logits.argmax(1) == yb).float().mean().detach()
        return loss.detach(), acc
