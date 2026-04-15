#!/usr/bin/env python3
"""
随机抽若干 test 帧，算 score；取最好/最差各 top-k，画 3D 框叠加图与 summary.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.bop_box3d import overlay_gt_pred_boxes
from diff.bop_frame import largest_instance, load_cam_K, load_gt_instances, load_rgb
from diff.dataset import build_pose_cond_vector, crop_mask_for_gt_instance, infer_cond_setup_from_ckpt
from diff.diffusion import linear_beta_schedule, sample_ddpm_eps
from diff.geometry import pose_to_vec9, rotation_geodesic_deg, vec9_to_pose, x12_to_pose
from diff.model import EpsMLP
from diff.paths import official_ism_predictions_dir


def _meta_pairs(split: str) -> list[tuple[int, int]]:
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
    ap.add_argument("--n-requested", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rot-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("diff/output/pose_eval_vis_box3d"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = _load_ckpt(args.checkpoint)
    T = int(ckpt.get("timesteps", 100))
    pose_dim = int(ckpt.get("pose_dim", 12))
    pose_repr = str(ckpt.get("pose_repr", "x12" if pose_dim == 12 else "vec9"))
    setup = infer_cond_setup_from_ckpt(ckpt)
    model = EpsMLP(
        pose_dim=pose_dim,
        cond_dim=setup.cond_dim,
        time_dim=32,
        hidden=256,
        obj_emb_dim=setup.obj_emb_dim,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    betas = linear_beta_schedule(T)

    pred_dir = args.pred_dir or official_ism_predictions_dir()
    pairs = _meta_pairs(args.split)
    if args.require_ism:
        from diff.ism_bridge import find_npz

        pairs = [(s, f) for s, f in pairs if find_npz(pred_dir, s, f) is not None]

    rng = np.random.default_rng(args.seed)
    if len(pairs) == 0:
        print("无可用帧。", file=sys.stderr)
        return 1
    n_req = min(args.n_requested, len(pairs))
    idxs = rng.choice(len(pairs), size=n_req, replace=False)
    sampled = [pairs[i] for i in idxs]

    results: list[dict] = []
    for sid, fid in sampled:
        sample_seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)

        rgb = load_rgb(args.split, sid, fid)
        K_vis = load_cam_K(args.split, sid, fid)
        K_cond = K_vis if setup.include_cam_k else None
        instances = load_gt_instances(args.split, sid, fid)
        li = largest_instance(instances)
        inst = instances[li]
        cm = crop_mask_for_gt_instance(
            rgb, instances, li, pred_dir, sid, fid, use_ism_crop=setup.include_ism_crop
        )
        cond = torch.from_numpy(
            build_pose_cond_vector(
                rgb,
                K_cond,
                include_cam_k=setup.include_cam_k,
                include_obj_id_scalar=setup.include_obj_id_scalar,
                obj_id=int(inst.obj_id) if setup.include_obj_id_scalar else None,
                include_ism_crop=setup.include_ism_crop,
                crop_mask=cm if setup.include_ism_crop else None,
            )
        ).unsqueeze(0).float()
        oid = None
        if setup.obj_emb_dim > 0:
            oid = torch.tensor([max(1, min(int(inst.obj_id), 21))], dtype=torch.long)

        with torch.no_grad():
            x0_hat = sample_ddpm_eps(model, cond, betas=betas, device=device, obj_id=oid)
        x_np = x0_hat.cpu().numpy().reshape(-1)

        if pose_repr == "vec9" or pose_dim == 9:
            R_p, t_p = vec9_to_pose(x_np)
        else:
            R_p, t_p = x12_to_pose(x_np)

        trans_mm = float(np.linalg.norm(t_p - inst.t_mm))
        rot_deg = rotation_geodesic_deg(R_p, inst.R_m2c)
        if pose_repr == "vec9" or pose_dim == 9:
            x0_gt = pose_to_vec9(inst.R_m2c, inst.t_mm)
        else:
            x0_gt = inst.pose_x12.copy()
        x12_l2 = float(np.linalg.norm(x_np - x0_gt))
        score = args.rot_weight * rot_deg + trans_mm

        results.append(
            {
                "scene_id": int(sid),
                "frame_id": int(fid),
                "inst_i": int(li),
                "obj_id": int(inst.obj_id),
                "trans_mm": trans_mm,
                "rot_deg": rot_deg,
                "x12_l2": x12_l2,
                "score": score,
                "rot_weight": args.rot_weight,
                "sample_seed": sample_seed,
                "R_pred": R_p,
                "t_pred": t_p,
                "R_gt": inst.R_m2c,
                "t_gt": inst.t_mm,
                "rgb": rgb,
                "K": K_vis,
            }
        )

    scores = np.array([r["score"] for r in results], dtype=np.float64)
    order = np.argsort(scores)
    k = max(1, args.top_k)
    best_idx = order[:k]
    worst_idx = order[-k:][::-1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    def _save(tag: str, rank: int, r: dict) -> None:
        name = f"{tag}_{rank:02d}_scene{r['scene_id']:06d}_frame{r['frame_id']:06d}.png"
        out = overlay_gt_pred_boxes(
            r["rgb"],
            r["K"],
            r["obj_id"],
            r["R_gt"],
            r["t_gt"],
            r["R_pred"],
            r["t_pred"],
        )
        cv2.imwrite(str(args.out_dir / name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    for i, j in enumerate(best_idx):
        _save("best", i + 1, results[j])
    for i, j in enumerate(worst_idx):
        _save("worst", i + 1, results[j])

    def _strip(d: dict) -> dict:
        out: dict = {}
        for k, v in d.items():
            if k in ("R_pred", "t_pred", "R_gt", "t_gt", "rgb", "K"):
                continue
            if isinstance(v, np.generic):
                out[k] = v.item()
            else:
                out[k] = v
        return out

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "n_requested": n_req,
        "n_valid": len(results),
        "seed": int(args.seed),
        "rot_weight": float(args.rot_weight),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "best": [_strip(results[j]) for j in best_idx],
        "worst": [_strip(results[j]) for j in worst_idx],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir} (summary + {2 * k} pngs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
