"""ISM npz 与 BOP GT 的 IoU、错类高重叠统计（与 diagnose_sample 日志一致）。"""

from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from diff.ism_bridge import mask_iou


def category_histogram_str(category_id: np.ndarray) -> str:
    c = Counter(int(x) for x in np.asarray(category_id).reshape(-1))
    parts = [f"{k}×{c[k]}" for k in sorted(c.keys())]
    return ", ".join(parts)


def _resize_mask(pm: np.ndarray, H: int, W: int) -> np.ndarray:
    pm = np.asarray(pm, dtype=np.float32)
    if pm.shape != (H, W):
        pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
    return pm


def match_same_class_iou_full(
    gt_masks: list[np.ndarray],
    gt_obj_ids: list[int],
    category_id: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[list[int], list[float], list[float]]:
    """
    同类最大 IoU；另返回「任意类」最大 IoU（用于 bestany）。
    返回 best_same_idx, iou_same, iou_best_any.
    """
    seg = segmentation
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    n = len(category_id)
    H, W = gt_masks[0].shape
    best_idx: list[int] = []
    best_iou: list[float] = []
    best_any: list[float] = []
    for gm, oid in zip(gt_masks, gt_obj_ids):
        bi, bu = -1, 0.0
        b_any = 0.0
        for j in range(n):
            pm = _resize_mask(seg[j], H, W)
            iou = mask_iou(gm, pm)
            b_any = max(b_any, iou)
            if int(category_id[j]) != int(oid):
                continue
            if iou > bu:
                bu, bi = iou, j
        best_idx.append(bi)
        best_iou.append(bu)
        best_any.append(b_any)
    return best_idx, best_iou, best_any


def wrong_class_high_overlap_count(
    gt_masks: list[np.ndarray],
    gt_obj_ids: list[int],
    category_id: np.ndarray,
    segmentation: np.ndarray,
    *,
    iou_thresh: float = 0.5,
) -> int:
    """每个 GT：若存在 category≠obj_id 的 pred 与 GT IoU≥thresh，则计 1。"""
    seg = segmentation
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    n = len(category_id)
    H, W = gt_masks[0].shape
    cnt = 0
    for gm, oid in zip(gt_masks, gt_obj_ids):
        best_wrong = 0.0
        for j in range(n):
            if int(category_id[j]) == int(oid):
                continue
            pm = _resize_mask(seg[j], H, W)
            best_wrong = max(best_wrong, mask_iou(gm, pm))
        if best_wrong >= iou_thresh:
            cnt += 1
    return cnt


def per_instance_records(
    gt_masks: list[np.ndarray],
    gt_obj_ids: list[int],
    category_id: np.ndarray,
    segmentation: np.ndarray,
    scores: np.ndarray | None,
    obj_id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    best_idx, ious, _ = match_same_class_iou_full(gt_masks, gt_obj_ids, category_id, segmentation)
    seg = segmentation
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    out: list[dict[str, Any]] = []
    for i, (iou, bi) in enumerate(zip(ious, best_idx)):
        oid = int(gt_obj_ids[i])
        rec: dict[str, Any] = {
            "gt_instance_index": i,
            "obj_id": oid,
            "obj_name": obj_id_to_name.get(oid, str(oid)),
            "best_pred_index": int(bi),
            "mask_iou": round(float(iou), 4),
        }
        if scores is not None and bi >= 0 and bi < len(scores):
            rec["pred_score"] = float(scores[bi])
        out.append(rec)
    return out
