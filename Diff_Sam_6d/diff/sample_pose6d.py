#!/usr/bin/env python3
"""加载 vec9 checkpoint，DDPM 采样并与 GT 对比。"""

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

from diff.bop_frame import largest_instance, load_cam_K, load_gt_instances, load_rgb
from diff.dataset import build_pose_cond_vector, crop_mask_for_gt_instance, infer_cond_setup_from_ckpt
from diff.diffusion import linear_beta_schedule, sample_ddpm_eps
from diff.geometry import pose_to_vec9, rotation_geodesic_deg, vec9_to_pose
from diff.model import EpsMLP
from diff.paths import BOP_YCBV, official_ism_predictions_dir


def _meta_pairs(split: str) -> list[tuple[int, int]]:
    meta_path = BOP_YCBV / f"{split}_metaData.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    scenes = [int(s) for s in data["scene_id"]]
    frames = [int(f) for f in data["frame_id"]]
    return list(zip(scenes, frames))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--random-frame", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stochastic-runs", type=int, default=4)
    ap.add_argument("--pred-dir", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    T = int(ckpt.get("timesteps", 100))
    pd = int(ckpt.get("pose_dim", 9))
    if pd != 9:
        print("此脚本用于 pose_dim=9 (vec9)", file=sys.stderr)
        return 1
    setup = infer_cond_setup_from_ckpt(ckpt)
    model = EpsMLP(
        pose_dim=9,
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

    rng = np.random.default_rng(args.seed)
    if args.random_frame:
        pairs = _meta_pairs(args.split)
        sid, fid = pairs[int(rng.integers(0, len(pairs)))]
    else:
        if args.scene is None or args.frame is None:
            print("请指定 --scene/--frame 或 --random-frame", file=sys.stderr)
            return 1
        sid, fid = args.scene, args.frame

    rgb = load_rgb(args.split, sid, fid)
    instances = load_gt_instances(args.split, sid, fid)
    li = largest_instance(instances)
    inst = instances[li]
    x0_gt = torch.from_numpy(pose_to_vec9(inst.R_m2c, inst.t_mm)).unsqueeze(0).float()
    K = load_cam_K(args.split, sid, fid) if setup.include_cam_k else None
    cm = crop_mask_for_gt_instance(
        rgb, instances, li, pred_dir, sid, fid, use_ism_crop=setup.include_ism_crop
    )
    cond = torch.from_numpy(
        build_pose_cond_vector(
            rgb,
            K,
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

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    errs: list[float] = []
    x_last = None
    for _ in range(max(1, args.stochastic_runs)):
        x0_hat = sample_ddpm_eps(model, cond, betas=betas, device=device, obj_id=oid)
        x_last = x0_hat
        e = torch.linalg.norm(x0_hat.cpu() - x0_gt, dim=-1).item()
        errs.append(e)

    x_np = x_last.cpu().numpy().reshape(-1)
    R_p, t_p = vec9_to_pose(x_np)
    R_g, t_g = inst.R_m2c, inst.t_mm
    trans_err = float(np.linalg.norm(t_p - t_g))
    rot_err = rotation_geodesic_deg(R_p, R_g)

    print(f"scene={sid} frame={fid}  inst={li} obj_id={inst.obj_id}")
    print(f"checkpoint={args.checkpoint}  timesteps={T}  device={device}")
    print(
        f"vec9 L2 vs GT: mean={float(np.mean(errs)):.4f}  std={float(np.std(errs)):.4f}  "
        f"(n={len(errs)} runs)"
    )
    print(f"translation |pred-gt| (mm): {trans_err:.2f}")
    print(f"rotation angle (deg): {rot_err:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
