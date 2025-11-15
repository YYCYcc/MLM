# gen/tabddpm.py  —— 方案B：带时间步嵌入的 Tabular DDPM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CondEmbed(nn.Module):
    def __init__(self, n_classes, n_clients, d=32):
        super().__init__()
        self.e_cls = nn.Embedding(n_classes, d // 2)
        self.e_cli = nn.Embedding(n_clients, d - d // 2)

    def forward(self, y_cls, y_cli):
        return torch.cat([self.e_cls(y_cls), self.e_cli(y_cli)], dim=-1)


class ScoreMLP(nn.Module):
    """
    方案B：score 网络吃 (x, cond, t_emb)
    in_dim = x_dim + cond_dim + t_dim
    """
    def __init__(self, x_dim, cond_dim=32, t_dim=64, width=256, depth=3):
        super().__init__()
        self.t_dim = int(t_dim)
        in_dim = int(x_dim + cond_dim + self.t_dim)

        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, cond, t_emb):
        """
        x:     [B, x_dim]
        cond:  [B, cond_dim]
        t_emb: [B, t_dim]
        """
        if t_emb is None:
            raise ValueError("ScoreMLP.forward 需要 t_emb（方案B）。")
        h = torch.cat([x, cond, t_emb], dim=-1)
        return self.net(h)  # 预测 epsilon


class SimpleDDPM(nn.Module):
    def __init__(
        self,
        x_dim,
        n_classes,
        n_clients,
        T=500,
        cond_dim=32,
        t_dim=64,
        device="cpu",
        schedule="linear",  # or "cosine"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.T = int(T)
        self.t_dim = int(t_dim)

        # 条件嵌入 & score 网络（带时间步嵌入）
        self.embed = CondEmbed(n_classes, n_clients, cond_dim).to(self.device)
        self.score = ScoreMLP(x_dim, cond_dim, t_dim=self.t_dim).to(self.device)

        # 预设扩散日程
        betas, alphas, alpha_bars = self._make_schedule(self.T, schedule=schedule)
        self.register_buffer("betas", betas)           # [T]
        self.register_buffer("alphas", alphas)         # [T]
        self.register_buffer("alpha_bars", alpha_bars) # [T]

    # ---------- schedules ----------
    @staticmethod
    def _cosine_alpha_cumprod(T, s=0.008):
        """
        cosine schedule (Nichol & Dhariwal, 2021)
        返回 a_bar[t], t=0..T-1
        """
        steps = T + 1
        t = torch.linspace(0, T, steps)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        f = f / f[0]
        a_bar = f[1:] / f[:-1]
        a_bar = torch.cumprod(a_bar, 0)
        return a_bar

    def _make_schedule(self, T, schedule="linear"):
        if schedule == "cosine":
            a_bar = self._cosine_alpha_cumprod(T)
            a_bar = torch.clamp(a_bar, 1e-5, 1.0)
            alphas = torch.empty(T)
            alphas[0] = a_bar[0]
            alphas[1:] = a_bar[1:] / a_bar[:-1]
            betas = 1.0 - alphas
        else:
            # linear
            betas = torch.linspace(1e-4, 2e-2, T, dtype=torch.float32)
            alphas = 1.0 - betas
            a_bar = torch.cumprod(alphas, dim=0)
        return betas.float(), alphas.float(), a_bar.float()

    # ---------- utils ----------
    @staticmethod
    def _view_like(coeff, x):
        if not isinstance(coeff, torch.Tensor):
            coeff = torch.as_tensor(coeff, device=x.device, dtype=x.dtype)
        if coeff.ndim == 0:
            coeff = coeff.expand(x.size(0))
        view_shape = (x.size(0),) + (1,) * (x.dim() - 1)
        return coeff.view(view_shape)

    def _timestep_embedding(self, t: torch.Tensor, dim: int = 64):
        if t.ndim == 0:
            t = t[None]
        t = t.to(self.device).float()
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(1, (half - 1)))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    # ---------- forward bits ----------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x0.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(x0.size(0))

        a_bar_t = self.alpha_bars[t]
        x = self._view_like(torch.sqrt(a_bar_t), x0) * x0 \
            + self._view_like(torch.sqrt(1.0 - a_bar_t), x0) * noise
        return x, noise

    def loss(self, x0, y_cls, y_cli):
        B = x0.size(0)
        y_cls = y_cls.to(x0.device, dtype=torch.long)
        y_cli = y_cli.to(x0.device, dtype=torch.long)

        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        x_t, eps = self.q_sample(x0, t)
        cond = self.embed(y_cls, y_cli)
        t_emb = self._timestep_embedding(t, dim=self.t_dim)
        eps_pred = self.score(x_t, cond, t_emb)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_t.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(x_t.size(0))

        beta_t = self._view_like(self.betas[t], x_t)
        alpha_t = self._view_like(self.alphas[t], x_t)
        a_bar_t = self._view_like(self.alpha_bars[t], x_t)

        t_emb = self._timestep_embedding(t, dim=self.t_dim)
        eps = self.score(x_t, cond, t_emb)

        mean = (x_t - (beta_t / torch.sqrt(1.0 - a_bar_t)) * eps) / torch.sqrt(alpha_t + 1e-8)

        z = torch.randn_like(x_t)
        mask = (t > 0).float().view((x_t.size(0),) + (1,) * (x_t.dim() - 1))
        x_prev = mean + mask * (torch.sqrt(beta_t) * z)
        return x_prev

    @torch.no_grad()
    def denoise_to_x0(self, x_t, t, y_cls, y_cli):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_t.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(x_t.size(0))

        cond = self.embed(y_cls, y_cli)
        t_emb = self._timestep_embedding(t, dim=self.t_dim)
        eps = self.score(x_t, cond, t_emb)
        a_bar_t = self._view_like(self.alpha_bars[t], x_t)
        x0_hat = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
        return x0_hat

    @torch.no_grad()
    def sample(self, n, x_dim, y_cls, y_cli):
        y_cls = y_cls.to(self.device, dtype=torch.long)
        y_cli = y_cli.to(self.device, dtype=torch.long)

        x = torch.randn(n, x_dim, device=self.device)
        cond = self.embed(y_cls, y_cli)
        for tt in reversed(range(self.T)):
            t = torch.full((n,), tt, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

    @torch.no_grad()
    def make_views(self, x_in, y_cls, y_cli, t_list=(50, 250, 450)):
        if not isinstance(x_in, torch.Tensor):
            x = torch.tensor(x_in, dtype=torch.float32, device=self.device)
        else:
            x = x_in.to(self.device, dtype=torch.float32)

        y_cls = y_cls.to(self.device, dtype=torch.long)
        y_cli = y_cli.to(self.device, dtype=torch.long)

        B, D = x.size(0), x.size(1)
        cond = self.embed(y_cls, y_cli)

        raw = x.detach().cpu().numpy()

        t_mid = t_list[len(t_list) // 2] if len(t_list) else max(0, self.T // 2)
        x_t_mid, _ = self.q_sample(x, t_mid)
        den = self.denoise_to_x0(x_t_mid, t_mid, y_cls, y_cli).detach().cpu().numpy()

        eps_list = []
        for tt in t_list:
            x_t, _ = self.q_sample(x, tt)
            t = torch.full((B,), tt, device=self.device, dtype=torch.long)
            t_emb = self._timestep_embedding(t, dim=self.t_dim)
            e = self.score(x_t, cond, t_emb)
            eps_list.append(e)
        score_ms = torch.cat(eps_list, dim=-1).detach().cpu().numpy()  # [B, D * len(t_list)]

        return {"raw": raw, "denoised": den, "score_ms": score_ms}
