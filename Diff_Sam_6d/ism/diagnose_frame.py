#!/usr/bin/env python3
"""检查单帧：RGB、scene_camera、scene_gt、mask_visib 与可选 ISM npz 是否存在。"""

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
    args = ap.parse_args()

    from diff.ism_bridge import find_npz, load_ism_npz
    from ism.paths import BOP_YCBV, official_ism_predictions_dir

    sdir = BOP_YCBV / args.split / f"{args.scene:06d}"
    rgb = sdir / "rgb" / f"{args.frame:06d}.png"
    print("scene_dir:", sdir)
    print("rgb exists:", rgb.is_file(), rgb)
    cam_path = sdir / "scene_camera.json"
    if cam_path.is_file():
        data = json.loads(cam_path.read_text(encoding="utf-8"))
        ok = str(args.frame) in data
        print("scene_camera.json: has frame", str(args.frame), "=", ok)
    else:
        print("no scene_camera.json", file=sys.stderr)
    gt_path = sdir / "scene_gt.json"
    if gt_path.is_file():
        sg = json.loads(gt_path.read_text(encoding="utf-8"))
        print("scene_gt.json: has frame", str(args.frame) in sg, "n_gt=", len(sg.get(str(args.frame), [])))
    pred = args.pred_dir or official_ism_predictions_dir()
    npz = find_npz(pred, args.scene, args.frame)
    print("npz:", npz)
    if npz is not None:
        z = load_ism_npz(npz)
        print("npz keys:", list(z.keys()), "n_pred:", len(z["category_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
