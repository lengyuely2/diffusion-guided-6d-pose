"""BOP 位姿与简单向量互转（毫米 / 无单位旋转）。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def R9_to_matrix(r9: list[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(r9, dtype=np.float64).reshape(3, 3)
    return a


def pose_to_x12(R_m2c: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
    """12D 向量: R 9 + t(mm)/500 做粗略尺度归一。"""
    R = np.asarray(R_m2c, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_mm, dtype=np.float64).reshape(3)
    return np.concatenate([R.reshape(-1), t / 500.0], axis=0).astype(np.float32)


def x12_to_pose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(12)
    R = x[:9].reshape(3, 3)
    t_mm = x[9:12] * 500.0
    return R, t_mm


def project_to_SO3(R: np.ndarray) -> np.ndarray:
    """将 3×3 矩阵投影到最近的旋转矩阵（SVD），det=+1。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rr = U @ Vt
    if np.linalg.det(Rr) < 0:
        U2 = U.copy()
        U2[:, -1] *= -1.0
        Rr = U2 @ Vt
    return Rr.astype(np.float64)


def rotation_geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """SO(3) 上 R_pred^T R_gt 的测地线角（度）；输入可先非正交，会先投影。"""
    Rp = project_to_SO3(R_pred)
    Rg = project_to_SO3(R_gt)
    R_rel = Rp.T @ Rg
    tr = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


# --- Zhou et al. 6D 旋转 + 平移 /500 → 9 维 ---


def matrix_to_6d_np(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def sixd_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """6D → R∈SO(3)（Gram–Schmidt），numpy。"""
    d6 = np.asarray(d6, dtype=np.float64).reshape(6)
    a1, a2 = d6[:3], d6[3:6]
    n1 = np.linalg.norm(a1) + 1e-12
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2) + 1e-12
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float64)


def pose_to_vec9(R_m2c: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
    """9D: 6D(R) + t(mm)/500。"""
    R = np.asarray(R_m2c, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_mm, dtype=np.float64).reshape(3)
    d6 = matrix_to_6d_np(R)
    return np.concatenate([d6, (t / 500.0).astype(np.float32)], axis=0)


def vec9_to_pose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(9)
    R = sixd_to_matrix_np(x[:6])
    t_mm = x[6:9] * 500.0
    return R, t_mm


def matrix_to_6d_torch(R: torch.Tensor) -> torch.Tensor:
    """R: (B,3,3) → (B,6)"""
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)


def sixd_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """d6: (B,6) → R (B,3,3)，可反传。"""
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    dot = (b1 * a2).sum(-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_geodesic_rad_torch(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """批量测地线角（弧度），R1,R2: (B,3,3)。"""
    R = torch.matmul(R1.transpose(-1, -2), R2)
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """DDPM ε 预测 → x0 估计。"""
    ab = alphas_cumprod[t].view(-1, 1)
    return (x_t - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab.clamp(min=1e-8))
