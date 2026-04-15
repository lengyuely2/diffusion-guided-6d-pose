"""定位 ISM 输出的 npz，并与 GT 做同类 mask IoU 匹配。"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import cv2
import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0.5, b > 0.5).sum()
    union = np.logical_or(a > 0.5, b > 0.5).sum()
    return float(inter / union) if union > 0 else 0.0


def find_npz(pred_dir: Path, scene_id: int, frame_id: int) -> Path | None:
    pred_dir = Path(pred_dir)
    for name in (
        f"scene{int(scene_id):06d}_frame{int(frame_id):06d}.npz",
        f"scene{int(scene_id):06d}_frame{int(frame_id)}.npz",
    ):
        p = pred_dir / name
        if p.is_file():
            return p
    cands = sorted(pred_dir.glob(f"scene*{scene_id:06d}*frame*{frame_id}*.npz"))
    cands = [p for p in cands if "_runtime" not in p.name]
    return cands[-1] if cands else None


def load_ism_npz(path: Path) -> dict[str, np.ndarray] | None:
    """
    读取 ISM 的 npz。若文件截断/损坏（常见于推理进程被 kill），返回 None 并由上层回退到 GT mask。
    """
    try:
        return dict(np.load(str(path), allow_pickle=True))
    except (EOFError, OSError, zipfile.BadZipFile, ValueError) as e:
        logging.warning("损坏或截断的 ISM npz，已跳过: %s (%s)", path, e)
        return None


def match_same_class_iou(
    gt_masks: list[np.ndarray],
    gt_obj_ids: list[int],
    category_id: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[list[int], list[float]]:
    """
    对每个 GT 实例，在预测中取 category_id == gt_obj_id 且 IoU 最大的那条。
    返回 best_pred_idx 列表（-1 表示无同类预测），以及对应 IoU。
    """
    if segmentation.ndim == 2:
        segmentation = segmentation[np.newaxis, ...]
    n = len(category_id)
    H, W = gt_masks[0].shape
    best_idx: list[int] = []
    best_iou: list[float] = []
    for gm, oid in zip(gt_masks, gt_obj_ids):
        bi, bu = -1, 0.0
        for j in range(n):
            if int(category_id[j]) != int(oid):
                continue
            pm = segmentation[j].astype(np.float32)
            if pm.shape != (H, W):
                pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
            iou = mask_iou(gm, pm)
            if iou > bu:
                bu, bi = iou, j
        best_idx.append(bi)
        best_iou.append(bu)
    return best_idx, best_iou
