#!/usr/bin/env python3
"""快速检查 diff 包导入与一次前向是否可运行。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def main() -> int:
    from diff.diffusion import linear_beta_schedule, q_sample, make_alphas_cumprod
    from diff.geometry import pose_to_x12, pose_to_vec9
    from diff.dataset import total_cond_dim
    from diff.model import EpsMLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, D = 2, 12
    cd = total_cond_dim(include_cam_k=True)
    m = EpsMLP(pose_dim=D, cond_dim=cd, time_dim=32, hidden=64).to(device)
    x = torch.randn(B, D, device=device)
    t = torch.randint(0, 10, (B,), device=device)
    c = torch.randn(B, cd, device=device)
    y = m(x, t, c)
    assert y.shape == (B, D)
    betas = linear_beta_schedule(10).to(device)
    acp = make_alphas_cumprod(betas)
    noise = torch.randn_like(x)
    xt = q_sample(x, t, noise, acp)
    assert xt.shape == x.shape
    import numpy as np

    R = np.eye(3, dtype=np.float64)
    t_mm = np.zeros(3, dtype=np.float64)
    _ = pose_to_x12(R, t_mm)
    _ = pose_to_vec9(R, t_mm)
    print("smoke_test ok", device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
