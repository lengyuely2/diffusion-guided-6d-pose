#!/usr/bin/env python3
"""
从 test meta 随机抽查若干帧，对照 ISM npz 与 BOP GT：
打印 pred 数量、category 直方图、同类 mean/min IoU、高重叠错类实例数；
可选仅对「问题帧」导出 compare / bestany / gt_only / metrics PNG 与 gt_pred_metrics JSON。

示例：
  conda run -n diffsam python -m ism.diagnose_sample --n 500 --seed 42 --vis \\
    --vis-mode problem --vis-max 120 --out-dir ism/output/vis_sample_seed42_n500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.bop_frame import load_gt_instances, load_rgb
from diff.ism_bridge import find_npz, load_ism_npz
from ism.metrics_viz import (
    category_histogram_str,
    match_same_class_iou_full,
    per_instance_records,
    wrong_class_high_overlap_count,
)
from ism.paths import BOP_YCBV, official_ism_predictions_dir
from ism.viz_render import build_metrics_lines, save_vis_bundle
from ism.ycbv_obj_names import YCBV_OBJ_NAMES


def _meta_pairs(split: str) -> list[tuple[int, int]]:
    meta_path = BOP_YCBV / f"{split}_metaData.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    scenes = [int(s) for s in data["scene_id"]]
    frames = [int(f) for f in data["frame_id"]]
    return list(zip(scenes, frames))


def frame_stem(scene_id: int, frame_id: int) -> str:
    return f"scene{int(scene_id):06d}_frame{int(frame_id):06d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-n", "--num-samples", type=int, default=50, dest="n", help="抽查帧数（不放回）")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="JSON/可视化输出目录")
    ap.add_argument("--vis", action="store_true", help="导出 PNG + 每帧 metrics JSON")
    ap.add_argument(
        "--vis-mode",
        choices=("problem", "all"),
        default="problem",
        help="problem: 仅错类高重叠>0 时出图；all: 每帧都出（仍受 --vis-max 限制）",
    )
    ap.add_argument("--vis-max", type=int, default=120, help="最多保存多少组可视化")
    ap.add_argument("--wrong-iou-thresh", type=float, default=0.5, help="错类高重叠 IoU 阈值")
    ap.add_argument("--log-file", type=Path, default=None, help="同时 tee 到该文件")
    args = ap.parse_args()

    pred_dir = args.pred_dir or official_ism_predictions_dir()
    pairs = _meta_pairs(args.split)
    n_total = len(pairs)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, n_total)
    pick = rng.choice(n_total, size=n, replace=False)

    out_dir = args.out_dir
    if args.vis and out_dir is None:
        out_dir = Path(f"ism/output/vis_sample_seed{args.seed}_n{n}")

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, s: str) -> None:
            for f in self.files:
                f.write(s)
                f.flush()

        def flush(self) -> None:
            for f in self.files:
                f.flush()

    log_out = [sys.stdout]
    log_f = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = args.log_file.open("w", encoding="utf-8")
        log_out.append(log_f)
    tee = Tee(*log_out)

    def log(msg: str = "") -> None:
        tee.write(msg + "\n")

    log(f"split={args.split}  抽查 {n} 帧  seed={args.seed}  (meta 总帧数 {n_total})")
    if args.vis:
        log("=" * 72)
        log(f"[render-vis] 输出目录: {out_dir}  模式={'仅问题帧' if args.vis_mode == 'problem' else '全部'}  vis-max={args.vis_max}")
        log("=" * 72)

    vis_saved = 0
    obj_names = {int(k): v for k, v in YCBV_OBJ_NAMES.items()}

    for k, meta_idx in enumerate(int(x) for x in pick):
        sid, fid = pairs[meta_idx]
        idx_label = k + 1
        npz = find_npz(pred_dir, sid, fid)
        log(f"\n### [{idx_label}/{n}]  meta_idx={meta_idx}  scene={sid:06d}  frame={fid:06d}\n")
        if npz is None:
            log(f"npz 缺失: pred_dir={pred_dir}")
            continue
        stem = frame_stem(sid, fid)
        data = load_ism_npz(npz)
        cat = np.asarray(data["category_id"])
        n_pred = len(cat)
        log(f"npz: {npz}")
        log(f"pred count: {n_pred}")
        log(f"category_id in npz: {category_histogram_str(cat)}")

        rgb = load_rgb(args.split, sid, fid)
        gts = load_gt_instances(args.split, sid, fid)
        gmasks = [x.mask for x in gts]
        gids = [x.obj_id for x in gts]
        seg = data["segmentation"]
        scores = data.get("score")
        scores_np = np.asarray(scores) if scores is not None else None

        best_idx, ious, _ = match_same_class_iou_full(gmasks, gids, cat, seg)
        n_gt = len(gts)
        if n_gt == 0:
            log("  (no GT instances)")
            continue
        mean_iou = float(np.mean(ious))
        min_iou = float(np.min(ious))
        n_wrong = wrong_class_high_overlap_count(
            gmasks, gids, cat, seg, iou_thresh=args.wrong_iou_thresh
        )
        log(
            f"  → mean_IoU(同类)={mean_iou:.4f}  min={min_iou:.4f}  "
            f"高重叠错类实例数={n_wrong}/{n_gt}"
        )

        is_problem = n_wrong > 0
        do_vis = args.vis and out_dir is not None and vis_saved < args.vis_max
        if do_vis:
            if args.vis_mode == "all":
                should_vis = True
            else:
                should_vis = is_problem
            if should_vis:
                rgb_path = Path(BOP_YCBV) / args.split / f"{sid:06d}" / "rgb" / f"{fid:06d}.png"
                per_inst = per_instance_records(gmasks, gids, cat, seg, scores_np, obj_names)
                payload = {
                    "split": args.split,
                    "scene_id": int(sid),
                    "frame_id": int(fid),
                    "rgb": str(rgb_path),
                    "npz": str(npz),
                    "legend": {
                        "gt": "浅绿/白边 = BOP mask_visib + scene_gt",
                        "pred": "洋红填充/边 = 同类 obj_id 下 IoU 最大的 ISM mask",
                    },
                    "per_instance": per_inst,
                    "mean_iou_matched": round(mean_iou, 4),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / f"gt_pred_metrics_{stem}.json"
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

                lines = build_metrics_lines(sid, fid, mean_iou, min_iou, n_wrong, n_gt)
                save_vis_bundle(out_dir, stem, rgb, gmasks, cat, seg, scores_np, best_idx, lines)
                vis_saved += 1
                log(
                    f"  [vis] {stem}  compare + bestany + gt_only + metrics  -> {out_dir}/"
                )

    if log_f is not None:
        log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
