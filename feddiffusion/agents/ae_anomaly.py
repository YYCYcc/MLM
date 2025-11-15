# agents/ae_anomaly.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEAnomaly(nn.Module):
    """
    无监督重构式异常检测；用于近线时对“看似正常”数据做微调，并定期重估阈值 tau
    """
    def __init__(self, d_feat:int, bottleneck:int=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_feat, 256), nn.ReLU(True),
                                 nn.Linear(256, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, 256), nn.ReLU(True),
                                 nn.Linear(256, d_feat))
        self.register_buffer("tau", torch.tensor(0.5))

    def forward(self, x):
        z = self.enc(x)
        xh = self.dec(z)
        return xh

    def recon_error(self, x):
        with torch.no_grad():
            xh = self.forward(x)
            return ((x - xh) ** 2).mean(dim=1)

    def step_unsup(self, xb, opt) -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        xh = self.forward(xb)
        loss = F.mse_loss(xh, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()
        return loss.detach()

    def update_threshold(self, loader, quantile: float = 0.995, device=None):
        errs = []
        for xb, _ in loader:
            if device:
                xb = xb.to(device)
            e = self.recon_error(xb).cpu()
            errs.append(e)
        if errs:
            errs = torch.cat(errs, 0)
            tau = torch.quantile(errs, q=quantile).item()
            self.tau.data = torch.tensor(tau)
        return float(self.tau.item())

    def predict_label(self, xb):
        e = self.recon_error(xb)
        return (e > self.tau).long()
