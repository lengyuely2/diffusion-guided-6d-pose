"""条件噪声网络 eps_theta(x_t, t, cond)：占位实现，后续可换 U-Net / SE(3) 头。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion import sinusoid_time_embedding


class EpsMLP(nn.Module):
    def __init__(
        self,
        pose_dim: int = 12,
        cond_dim: int = 192,
        time_dim: int = 32,
        hidden: int = 256,
        obj_emb_dim: int = 0,
        num_obj_ids: int = 22,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.cond_dim = cond_dim
        self.obj_emb_dim = int(obj_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        extra = self.obj_emb_dim
        if extra > 0:
            self.obj_emb = nn.Embedding(int(num_obj_ids), self.obj_emb_dim)
        else:
            self.obj_emb = None
        in_dim = pose_dim + time_dim + cond_dim + extra
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, pose_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        obj_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        te = sinusoid_time_embedding(t, self.time_mlp[0].in_features)
        te = self.time_mlp(te)
        parts: list[torch.Tensor] = [x_t, te, cond]
        if self.obj_emb is not None:
            if obj_id is None:
                raise ValueError("obj_emb 已启用，需要传入 obj_id (B,) long")
            oid = obj_id.long().clamp(0, self.obj_emb.num_embeddings - 1)
            parts.append(self.obj_emb(oid))
        h = torch.cat(parts, dim=-1)
        return self.net(h)
