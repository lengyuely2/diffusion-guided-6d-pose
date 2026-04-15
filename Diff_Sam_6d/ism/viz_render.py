"""GT / Pred mask 叠加图：compare、bestany、gt_only、metrics 文本图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from diff.ism_bridge import mask_iou


def _blend_mask_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    m = (mask > 0.5).astype(np.float32)
    c = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    for c_i in range(3):
        bgr[:, :, c_i] = bgr[:, :, c_i] * (1 - alpha * m) + c[:, :, c_i] * (alpha * m)
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _edge_overlay(rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.drawContours(out_bgr, contours, -1, bgr, thickness=2)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def render_compare(
    rgb: np.ndarray,
    gt_masks: list[np.ndarray],
    matched_pred_masks: list[np.ndarray | None],
) -> np.ndarray:
    """浅绿 GT + 洋红边 matched 同类 pred。"""
    out = rgb.astype(np.float32)
    H, W = rgb.shape[:2]
    green = np.array([180.0, 255.0, 180.0], dtype=np.float32)
    for gm in gt_masks:
        m = cv2.resize(gm, (W, H), interpolation=cv2.INTER_NEAREST) if gm.shape != (H, W) else gm
        w = (m > 0.5).astype(np.float32)[..., np.newaxis]
        out = out * (1.0 - 0.35 * w) + green * (0.35 * w)
    out = np.clip(out, 0, 255).astype(np.uint8)
    for pm in matched_pred_masks:
        if pm is None:
            continue
        m = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST) if pm.shape != (H, W) else pm
        out = _edge_overlay(out, m, (255, 0, 255))
    return out


def render_gt_only(rgb: np.ndarray, gt_masks: list[np.ndarray]) -> np.ndarray:
    """多实例 GT 用不同绿色深浅叠加。"""
    H, W = rgb.shape[:2]
    base = rgb.copy().astype(np.float32)
    for k, gm in enumerate(gt_masks):
        m = cv2.resize(gm, (W, H), interpolation=cv2.INTER_NEAREST) if gm.shape != (H, W) else gm
        tint = np.array([40 + k * 30, 200 - k * 15, 60], dtype=np.float32)
        w = (m > 0.5).astype(np.float32)[..., np.newaxis]
        base = base * (1 - 0.35 * w) + tint * (0.35 * w)
    return np.clip(base, 0, 255).astype(np.uint8)


def render_bestany(
    rgb: np.ndarray,
    gt_masks: list[np.ndarray],
    category_id: np.ndarray,
    segmentation: np.ndarray,
) -> np.ndarray:
    """每个 GT 上叠加「任意类」IoU 最大的 pred（青色系边）。"""
    seg = segmentation
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    n = len(category_id)
    H, W = rgb.shape[:2]
    out = rgb.copy()
    for gm in gt_masks:
        g = cv2.resize(gm, (W, H), interpolation=cv2.INTER_NEAREST) if gm.shape != (H, W) else gm
        best_j, best_iou = -1, -1.0
        for j in range(n):
            pm = seg[j].astype(np.float32)
            if pm.shape != (H, W):
                pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
            iou = mask_iou(g, pm)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            pm = seg[best_j].astype(np.float32)
            if pm.shape != (H, W):
                pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
            out = _edge_overlay(out, pm, (0, 255, 255))
    return out


def render_metrics_panel(text_lines: list[str], size: tuple[int, int] = (720, 400)) -> np.ndarray:
    w, h = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    y = 28
    for line in text_lines[:25]:
        cv2.putText(img, line[:120], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_vis_bundle(
    out_dir: Path,
    stem: str,
    rgb: np.ndarray,
    gt_masks: list[np.ndarray],
    category_id: np.ndarray,
    segmentation: np.ndarray,
    scores: np.ndarray | None,
    best_same_idx: list[int],
    metrics_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = segmentation
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]
    H, W = rgb.shape[:2]
    matched: list[np.ndarray | None] = []
    for bi in best_same_idx:
        if bi < 0:
            matched.append(None)
            continue
        pm = seg[bi].astype(np.float32)
        if pm.shape != (H, W):
            pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
        matched.append(pm)

    compare = render_compare(rgb, gt_masks, matched)
    bestany = render_bestany(rgb, gt_masks, category_id, segmentation)
    gtonly = render_gt_only(rgb, gt_masks)
    panel = render_metrics_panel(metrics_lines)

    cv2.imwrite(str(out_dir / f"{stem}_compare.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_bestany.png"), cv2.cvtColor(bestany, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_gt_only.png"), cv2.cvtColor(gtonly, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_metrics.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def build_metrics_lines(
    scene_id: int,
    frame_id: int,
    mean_iou: float,
    min_iou: float,
    n_wrong: int,
    n_gt: int,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        f"scene={scene_id:06d} frame={frame_id:06d}",
        f"mean_IoU(matched)={mean_iou:.4f}  min={min_iou:.4f}",
        f"wrong_class_high_iou(>=0.5): {n_wrong}/{n_gt}",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    return lines
