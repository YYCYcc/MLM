# agents/flow_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMLP(nn.Module):
    def __init__(self, x_dim, n_classes, width=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], x_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    @torch.no_grad()
    def accuracy(self, logits, y):
        return (logits.argmax(dim=1) == y).float().mean()

    def step(self, xb, yb, opt, class_weights=None, focal=False, gamma=2.0):
        """
        训练一步，返回 (loss, acc)
        - class_weights: Tensor[n_classes] 或 None
        - focal: 是否使用 focal loss
        - gamma: focal loss 的 gamma
        """
        opt.zero_grad(set_to_none=True)
        logits = self(xb)

        if focal:
            # per-sample CE
            ce = F.cross_entropy(logits, yb, weight=class_weights, reduction="none")
            with torch.no_grad():
                pt = torch.softmax(logits, dim=1).gather(1, yb.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
            loss = ((1.0 - pt) ** gamma * ce).mean()
        else:
            loss = F.cross_entropy(logits, yb, weight=class_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

        acc = self.accuracy(logits, yb)
        return loss.detach(), acc.detach()
