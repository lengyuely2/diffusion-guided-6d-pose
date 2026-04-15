"""YCB-V models_info 轴对齐包围盒 → 相机坐标系 → 图像投影与绘制。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .paths import BOP_YCBV


def models_info_path() -> Path:
    return BOP_YCBV / "models" / "models_info.json"


@lru_cache(maxsize=1)
def load_models_info() -> dict[str, dict]:
    p = models_info_path()
    return json.loads(p.read_text(encoding="utf-8"))


def object_corners_mm(obj_id: int) -> np.ndarray:
    """物体坐标系下 8 个角点 (mm)，来自 models_info 轴对齐盒。"""
    info = load_models_info()[str(int(obj_id))]
    min_x, min_y, min_z = info["min_x"], info["min_y"], info["min_z"]
    sx, sy, sz = info["size_x"], info["size_y"], info["size_z"]
    pts = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        bx, by, bz = (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1
        pts[i] = [min_x + bx * sx, min_y + by * sy, min_z + bz * sz]
    return pts


def box_edges_indices() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(8):
        for b in range(3):
            j = i ^ (1 << b)
            if i < j:
                edges.append((i, j))
    return edges


def transform_points_m2c(R: np.ndarray, t_mm: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_mm, dtype=np.float64).reshape(3)
    return (pts_m @ R.T) + t.reshape(1, 3)


def project_points(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pts_cam: (N,3) → uv (N,2), z (N,)"""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    p = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    x = p[:, 0]
    y = p[:, 1]
    z = np.clip(p[:, 2], 1e-6, None)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=-1), z


def draw_box3d_edges(
    img_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t_mm: np.ndarray,
    obj_id: int,
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    corners_m = object_corners_mm(obj_id)
    corners_c = transform_points_m2c(R, t_mm, corners_m)
    uv, z = project_points(K, corners_c)
    H, W = img_bgr.shape[:2]
    edges = box_edges_indices()
    for a, b in edges:
        if z[a] <= 0 or z[b] <= 0:
            continue
        pa = (int(round(uv[a, 0])), int(round(uv[a, 1])))
        pb = (int(round(uv[b, 0])), int(round(uv[b, 1])))
        if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
            cv2.line(img_bgr, pa, pb, color, thickness, lineType=cv2.LINE_AA)


def overlay_gt_pred_boxes(
    rgb: np.ndarray,
    K: np.ndarray,
    obj_id: int,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    *,
    color_gt: tuple[int, int, int] = (0, 200, 0),
    color_pred: tuple[int, int, int] = (0, 80, 255),
    thickness: int = 2,
) -> np.ndarray:
    """RGB → BGR 绘制 GT(绿) / Pred(橙红)，返回 RGB uint8。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    draw_box3d_edges(bgr, K, R_gt, t_gt, obj_id, color=color_gt, thickness=thickness)
    draw_box3d_edges(bgr, K, R_pred, t_pred, obj_id, color=color_pred, thickness=thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
