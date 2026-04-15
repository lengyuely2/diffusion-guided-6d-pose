#!/usr/bin/env python3
"""
扩散噪声预测训练：MSE(eps_hat, eps)，条件为下采样 RGB + 归一化内参（默认 196 维；--no-cam-k 为 192 维旧设定）。

  conda run -n diffsam python diff/train.py --require-ism --steps 20000 --batch-size 32 --num-workers 4 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.dataset import YcbvIsmPoseDataset, infer_cond_setup_from_ckpt, total_cond_dim
from diff.diffusion import linear_beta_schedule, make_alphas_cumprod, q_sample
from diff.logutil import setup_tee_log
from diff.model import EpsMLP
from diff.paths import official_ism_predictions_dir


def _load_ckpt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        default="test",
        help="BOP ycbv 下 {split}_metaData.json，如 test、train_real",
    )
    ap.add_argument("--pred-dir", type=Path, default=None, help="ISM npz 目录，默认官方 log")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--require-ism", action="store_true", help="仅保留已有 npz 的帧")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--timesteps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save", type=Path, default=Path("diff/output/eps_mlp.pt"))
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--no-cam-k",
        action="store_true",
        help="条件不含相机内参（192 维，与旧 checkpoint 兼容）",
    )
    ap.add_argument(
        "--cond-obj-id",
        action="store_true",
        help="条件末尾追加归一化 obj_id（+1 维）；与 --obj-emb-dim 互斥",
    )
    ap.add_argument("--ism-crop-cond", action="store_true", help="ISM 对齐裁剪 RGB（+192 维）")
    ap.add_argument(
        "--obj-emb-dim",
        type=int,
        default=0,
        help=">0 时 nn.Embedding 物体 id；与 --cond-obj-id 互斥",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="额外写日志到该文件（行缓冲；nohup+conda run 时建议加）",
    )
    args = ap.parse_args()
    if args.log_file is not None:
        setup_tee_log(args.log_file)
        print(f"[train] tee log -> {args.log_file.resolve()}", flush=True)

    if args.cond_obj_id and args.obj_emb_dim > 0:
        print("不能同时使用 --cond-obj-id 与 --obj-emb-dim", file=sys.stderr, flush=True)
        sys.exit(1)

    ckpt0 = None
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"找不到 checkpoint: {args.resume}", file=sys.stderr, flush=True)
            sys.exit(1)
        ckpt0 = _load_ckpt(args.resume)
        setup = infer_cond_setup_from_ckpt(ckpt0)
        include_cam_k = setup.include_cam_k
        include_obj_id_scalar = setup.include_obj_id_scalar
        include_ism_crop = setup.include_ism_crop
        obj_emb_dim = setup.obj_emb_dim
        cond_dim = setup.cond_dim
    else:
        include_cam_k = not args.no_cam_k
        include_obj_id_scalar = args.cond_obj_id
        include_ism_crop = args.ism_crop_cond
        obj_emb_dim = int(args.obj_emb_dim)
        cond_dim = total_cond_dim(
            include_cam_k=include_cam_k,
            include_obj_id_scalar=include_obj_id_scalar,
            include_ism_crop=include_ism_crop,
        )

    use_obj_embedding = obj_emb_dim > 0
    instance_mode = (
        "random_visible"
        if (include_obj_id_scalar or include_ism_crop or use_obj_embedding)
        else "largest"
    )

    if include_ism_crop and not args.require_ism and args.resume is None:
        print("[train] 警告: --ism-crop-cond 通常需要 --require-ism", flush=True)

    pred = args.pred_dir or official_ism_predictions_dir()
    ds = YcbvIsmPoseDataset(
        split=args.split,
        pred_dir=pred,
        max_samples=args.max_samples,
        require_ism=args.require_ism,
        pose_repr="x12",
        include_cam_k=include_cam_k,
        include_obj_id=include_obj_id_scalar,
        include_ism_crop=include_ism_crop,
        use_obj_embedding=use_obj_embedding,
        instance_mode=instance_mode,
    )
    if len(ds) == 0:
        print("数据集为空。", file=sys.stderr, flush=True)
        sys.exit(1)

    drop_last = len(ds) >= args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nw = max(0, args.num_workers)
    dl_kw: dict = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": nw,
        "drop_last": drop_last,
    }
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["pin_memory"] = device.type == "cuda"
    dl = DataLoader(ds, **dl_kw)
    model = EpsMLP(
        pose_dim=12,
        cond_dim=cond_dim,
        time_dim=32,
        hidden=256,
        obj_emb_dim=obj_emb_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    betas = linear_beta_schedule(args.timesteps).to(device)
    alphas_cumprod = make_alphas_cumprod(betas)

    step = 0
    if ckpt0 is not None:
        model.load_state_dict(ckpt0["model"])
        if "optimizer" in ckpt0:
            opt.load_state_dict(ckpt0["optimizer"])
        step = int(ckpt0.get("step", 0))
        print(
            f"resume: step={step} from {args.resume}  cond_dim={cond_dim} "
            f"ism_crop={include_ism_crop} obj_emb={obj_emb_dim}",
            flush=True,
        )

    last_path = args.save.parent / f"{args.save.stem}_last.pt"

    def save_checkpoint(path: Path, *, final: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "timesteps": args.timesteps,
            "lr": args.lr,
            "pose_dim": 12,
            "pose_repr": "x12",
            "cond_dim": cond_dim,
            "include_cam_k": include_cam_k,
            "include_obj_id": include_obj_id_scalar,
            "include_ism_crop": include_ism_crop,
            "obj_emb_dim": obj_emb_dim,
            "final": final,
        }
        torch.save(payload, path)
        tag = "final" if final else "ckpt"
        print(f"saved [{tag}] step={step} -> {path}", flush=True)

    model.train()
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            x0 = batch["x0"].to(device)
            cond = batch["cond"].to(device)
            oid = batch.get("obj_id")
            if oid is not None:
                oid = oid.to(device)
            B = x0.shape[0]
            t = torch.randint(0, args.timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise, alphas_cumprod)
            eps_hat = model(x_t, t, cond, obj_id=oid)
            loss = F.mse_loss(eps_hat, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"step {step} loss={loss.item():.6f}", flush=True)
            step += 1
            se = args.save_every
            if se > 0 and step % se == 0 and step < args.steps:
                save_checkpoint(last_path, final=False)

    save_checkpoint(args.save, final=True)
    save_checkpoint(last_path, final=False)


if __name__ == "__main__":
    main()
