#!/usr/bin/env python3
"""
对 test split 全量帧：最大可见实例上跑 DDPM 采样，统计 trans/rot/score，写 CSV + JSON。
score = rot_weight * rot_deg + trans_mm（与已有 summary 一致）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.bop_frame import largest_instance, load_cam_K, load_gt_instances, load_rgb
from diff.dataset import build_pose_cond_vector, crop_mask_for_gt_instance, infer_cond_setup_from_ckpt
from diff.diffusion import linear_beta_schedule, sample_ddpm_eps
from diff.geometry import pose_to_vec9, rotation_geodesic_deg, vec9_to_pose, x12_to_pose
from diff.model import EpsMLP
from diff.paths import official_ism_predictions_dir


def _meta_pairs_from_split(split: str) -> list[tuple[int, int]]:
    from diff.paths import BOP_YCBV

    meta_path = BOP_YCBV / f"{split}_metaData.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    scenes = [int(s) for s in data["scene_id"]]
    frames = [int(f) for f in data["frame_id"]]
    return list(zip(scenes, frames))


def _load_ckpt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--require-ism", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rot-weight", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None, help="调试用，截断帧数")
    ap.add_argument("--seed-base", type=int, default=1315423911)
    ap.add_argument("--out-csv", type=Path, default=Path("diff/output/full_test_eval_per_frame.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("diff/output/full_test_eval_summary.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = _load_ckpt(args.checkpoint)
    T = int(ckpt.get("timesteps", 100))
    pose_dim = int(ckpt.get("pose_dim", 12))
    pose_repr = str(ckpt.get("pose_repr", "x12" if pose_dim == 12 else "vec9"))
    setup = infer_cond_setup_from_ckpt(ckpt)
    obj_emb_dim = setup.obj_emb_dim
    model = EpsMLP(
        pose_dim=pose_dim,
        cond_dim=setup.cond_dim,
        time_dim=32,
        hidden=256,
        obj_emb_dim=obj_emb_dim,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    betas = linear_beta_schedule(T)

    pred_dir = args.pred_dir or official_ism_predictions_dir()
    pairs = _meta_pairs_from_split(args.split)
    if args.max_frames is not None:
        pairs = pairs[: args.max_frames]

    if args.require_ism:
        from diff.ism_bridge import find_npz

        filt: list[tuple[int, int]] = []
        for sid, fid in pairs:
            if find_npz(pred_dir, sid, fid) is not None:
                filt.append((sid, fid))
        pairs = filt

    n_meta = len(pairs)
    rows: list[dict] = []
    t0 = time.perf_counter()

    try:
        from tqdm import tqdm

        it = tqdm(pairs, desc="eval", mininterval=1.0)
    except ImportError:
        it = pairs

    for sid, fid in it:
        seed = int((args.seed_base + sid * 100003 + fid * 9176) & 0x7FFFFFFF)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        rgb = load_rgb(args.split, sid, fid)
        instances = load_gt_instances(args.split, sid, fid)
        li = largest_instance(instances)
        inst = instances[li]
        K = load_cam_K(args.split, sid, fid) if setup.include_cam_k else None
        cm = crop_mask_for_gt_instance(
            rgb,
            instances,
            li,
            pred_dir,
            sid,
            fid,
            use_ism_crop=setup.include_ism_crop,
        )
        cond_np = build_pose_cond_vector(
            rgb,
            K,
            include_cam_k=setup.include_cam_k,
            include_obj_id_scalar=setup.include_obj_id_scalar,
            obj_id=int(inst.obj_id) if setup.include_obj_id_scalar else None,
            include_ism_crop=setup.include_ism_crop,
            crop_mask=cm if setup.include_ism_crop else None,
        )
        cond = torch.from_numpy(cond_np).unsqueeze(0).float()
        oid = None
        if obj_emb_dim > 0:
            oid = torch.tensor([max(1, min(int(inst.obj_id), 21))], dtype=torch.long)

        with torch.no_grad():
            x0_hat = sample_ddpm_eps(model, cond, betas=betas, device=device, obj_id=oid)
        x_np = x0_hat.cpu().numpy().reshape(-1)

        if pose_repr == "vec9" or pose_dim == 9:
            R_p, t_p = vec9_to_pose(x_np)
            x0_gt = pose_to_vec9(inst.R_m2c, inst.t_mm)
        else:
            R_p, t_p = x12_to_pose(x_np)
            x0_gt = inst.pose_x12.copy()

        trans_mm = float(np.linalg.norm(t_p - inst.t_mm))
        rot_deg = rotation_geodesic_deg(R_p, inst.R_m2c)
        x12_l2 = float(np.linalg.norm(x_np - x0_gt))
        score = args.rot_weight * rot_deg + trans_mm

        rows.append(
            {
                "scene_id": sid,
                "frame_id": fid,
                "inst_i": li,
                "obj_id": inst.obj_id,
                "trans_mm": trans_mm,
                "rot_deg": rot_deg,
                "x12_l2": x12_l2,
                "score": score,
            }
        )

    wall = time.perf_counter() - t0
    n_eval = len(rows)
    trans_arr = np.array([r["trans_mm"] for r in rows], dtype=np.float64)
    rot_arr = np.array([r["rot_deg"] for r in rows], dtype=np.float64)
    score_arr = np.array([r["score"] for r in rows], dtype=np.float64)
    l2_arr = np.array([r["x12_l2"] for r in rows], dtype=np.float64)

    def _stats(a: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    def _arg_extr(a: np.ndarray, worst: bool) -> int:
        return int(a.argmax() if worst else a.argmin())

    ibs, iws = _arg_extr(score_arr, False), _arg_extr(score_arr, True)
    ibt, iwt = _arg_extr(trans_arr, False), _arg_extr(trans_arr, True)
    ibr, iwr = _arg_extr(rot_arr, False), _arg_extr(rot_arr, True)

    def _pick(i: int) -> dict:
        r = rows[i]
        return {
            "scene_id": r["scene_id"],
            "frame_id": r["frame_id"],
            "inst_i": r["inst_i"],
            "obj_id": r["obj_id"],
            "trans_mm": r["trans_mm"],
            "rot_deg": r["rot_deg"],
            "x12_l2": r["x12_l2"],
            "score": r["score"],
        }

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "n_meta": n_meta,
        "n_evaluated": n_eval,
        "n_skipped": 0,
        "rot_weight": args.rot_weight,
        "wall_time_sec": wall,
        "trans_mm": _stats(trans_arr),
        "rot_deg": _stats(rot_arr),
        "score": _stats(score_arr),
        "x12_l2": _stats(l2_arr),
        "best_by_score": _pick(ibs),
        "worst_by_score": _pick(iws),
        "best_by_trans": _pick(ibt),
        "worst_by_trans": _pick(iwt),
        "best_by_rot": _pick(ibr),
        "worst_by_rot": _pick(iwr),
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scene_id", "frame_id", "inst_i", "obj_id", "trans_mm", "rot_deg", "x12_l2", "score"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out_csv} ({n_eval} rows)", flush=True)
    print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())