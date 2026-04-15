#!/usr/bin/env python3
"""
单帧：GT mask vs ISM npz（同类 IoU），可选写 gt_pred_metrics JSON 与四张可视化图。

用法：
  conda run -n diffsam python -m ism.visualize_gt_compare --scene 54 --frame 9 --out-dir ism/output/single_vis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="若设置则写 JSON + PNG")
    ap.add_argument("--wrong-iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    from diff.bop_frame import load_gt_instances, load_rgb
    from diff.ism_bridge import find_npz, load_ism_npz
    from ism.metrics_viz import (
        match_same_class_iou_full,
        per_instance_records,
        wrong_class_high_overlap_count,
    )
    from ism.paths import BOP_YCBV, official_ism_predictions_dir
    from ism.viz_render import build_metrics_lines, save_vis_bundle
    from ism.ycbv_obj_names import YCBV_OBJ_NAMES

    pred = args.pred_dir or official_ism_predictions_dir()
    npz = find_npz(pred, args.scene, args.frame)
    if npz is None:
        print(f"未找到 npz: scene={args.scene} frame={args.frame} in {pred}", file=sys.stderr)
        return 1
    rgb = load_rgb(args.split, args.scene, args.frame)
    gts = load_gt_instances(args.split, args.scene, args.frame)
    gmasks = [x.mask for x in gts]
    gids = [x.obj_id for x in gts]
    data = load_ism_npz(npz)
    cat = data["category_id"]
    seg = data["segmentation"]
    import numpy as np

    scores = data.get("score")
    scores_np = np.asarray(scores) if scores is not None else None

    best_idx, ious, _ = match_same_class_iou_full(gmasks, gids, cat, seg)
    n_gt = len(gts)
    mean_iou = float(np.mean(ious)) if n_gt else 0.0
    min_iou = float(np.min(ious)) if n_gt else 0.0
    n_wrong = wrong_class_high_overlap_count(
        gmasks, gids, cat, seg, iou_thresh=args.wrong_iou_thresh
    )

    obj_names = {int(k): v for k, v in YCBV_OBJ_NAMES.items()}
    print(f"npz={npz}  rgb={rgb.shape}  n_gt={len(gts)}  n_pred={len(cat)}")
    for i, (g, iou) in enumerate(zip(gts, ious)):
        print(f"  inst{i} obj_id={g.obj_id}  IoU(same-class)={iou:.4f}")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"scene{args.scene:06d}_frame{args.frame:06d}"
        rgb_path = BOP_YCBV / args.split / f"{args.scene:06d}" / "rgb" / f"{args.frame:06d}.png"
        per_inst = per_instance_records(gmasks, gids, cat, seg, scores_np, obj_names)
        payload = {
            "split": args.split,
            "scene_id": int(args.scene),
            "frame_id": int(args.frame),
            "rgb": str(rgb_path),
            "npz": str(npz),
            "legend": {
                "gt": "浅绿/白边 = BOP mask_visib + scene_gt",
                "pred": "洋红填充/边 = 同类 obj_id 下 IoU 最大的 ISM mask",
            },
            "per_instance": per_inst,
            "mean_iou_matched": round(mean_iou, 4),
        }
        jp = args.out_dir / f"gt_pred_metrics_{stem}.json"
        jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = build_metrics_lines(args.scene, args.frame, mean_iou, min_iou, n_wrong, n_gt)
        save_vis_bundle(args.out_dir, stem, rgb, gmasks, cat, seg, scores_np, best_idx, lines)
        print(f"wrote {jp} and {stem}_*.png -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
