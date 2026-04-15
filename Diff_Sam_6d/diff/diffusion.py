"""极简 DDPM 标量/向量扩散工具（用于 pose 向量去噪）。"""

from __future__ import annotations

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def make_alphas_cumprod(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps."""
    ab = alphas_cumprod[t].view(-1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def ddpm_posterior_variance(betas: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Var[x_{t-1}|x_t] 系数，与 Ho DDPM 一致。"""
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
    )
    return betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20)


@torch.no_grad()
def sample_ddpm_eps(
    eps_model: torch.nn.Module,
    cond: torch.Tensor,
    *,
    betas: torch.Tensor,
    device: torch.device,
    obj_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    从标准高斯出发，逐步反推 x_0。eps_model(x_t, t, cond[, obj_id]) 预测 epsilon。
    cond: (B, C)；返回 x_0: (B, D)，与训练时 x0 同空间。
    """
    betas = betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod = make_alphas_cumprod(betas)
    posterior_variance = ddpm_posterior_variance(betas, alphas_cumprod)
    sqrt_recip_alphas = torch.rsqrt(alphas.clamp(min=1e-20))
    sqrt_one_minus_acp = torch.sqrt((1.0 - alphas_cumprod).clamp(min=1e-20))

    B = cond.shape[0]
    D = eps_model.pose_dim  # type: ignore[attr-defined]
    x = torch.randn(B, D, device=device, dtype=torch.float32)
    T = betas.shape[0]
    cond = cond.to(device)
    if obj_id is not None:
        obj_id = obj_id.to(device)

    for step in reversed(range(T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        if obj_id is not None:
            eps_theta = eps_model(x, t, cond, obj_id)
        else:
            eps_theta = eps_model(x, t, cond)
        mean = sqrt_recip_alphas[step] * (
            x - betas[step] * eps_theta / sqrt_one_minus_acp[step]
        )
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance[step].clamp(min=0)) * noise
        else:
            x = mean
    return x


def sinusoid_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) long, 映射到 (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device, dtype=torch.float32))
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
    return emb
