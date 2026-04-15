#!/usr/bin/env python3
"""
将 diffusion pose 预测导出为 BOP Challenge 2019 CSV，供 bop_toolkit 的 eval_bop19_pose.py 评测。

CSV 列（与 bop_toolkit_lib.inout.load_bop_results(..., version=\"bop19\") 一致）：
  scene_id,im_id,obj_id,score,R,t,time
其中 R 为 9 个空格分隔的 float（行优先 3×3），t 为 3 个空格分隔的 float（毫米，与 BOP GT cam_t_m2c 一致）。

输出文件名需符合 parse_result_filename，例如：{method}_ycbv-test.csv

策略说明：
  - match_largest（默认）：与 eval_full_test 一致，每帧只做一次采样；仅当 target 的 obj_id 等于该帧
    最大可见 GT 实例时写一行（其余 target 不写）。
  - per_target：与 BOP test_targets 行数对齐——对每条 (scene_id, im_id, obj_id) 各做一次 DDPM；
    条件为整图 RGB（±K）；若 checkpoint 含 obj_id 条件（训练时 --cond-obj-id），则每条 target 在 cond 中写入
    对应 obj_id。否则仍为「同图共享 cond」，易与错误物体对齐。用 GT 校验该帧存在该 obj_id，并在多个同 obj_id
    实例中取 mask 面积最大者（oracle 过滤）。同图所有行的 time 字段相同，满足 bop_toolkit check_consistent_timings。
  - duplicate：每条 target 复用同一次采样（调试用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.bop_frame import largest_instance, load_cam_K, load_gt_instances, load_rgb
from diff.dataset import build_pose_cond_vector, crop_mask_for_gt_instance, infer_cond_setup_from_ckpt
from diff.diffusion import linear_beta_schedule, sample_ddpm_eps
from diff.geometry import project_to_SO3, vec9_to_pose, x12_to_pose
from diff.model import EpsMLP
from diff.paths import BOP_YCBV, official_ism_predictions_dir

DIFF_OUT = Path(__file__).resolve().parent / "output"


def _load_ckpt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_targets(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in {path}")
    return data


def _largest_instance_index_for_obj_id(instances, obj_id: int) -> int:
    """在同 obj_id 的 GT 实例中取可见像素最多的下标；无匹配返回 -1。"""
    best_i = -1
    best_area = -1.0
    oid = int(obj_id)
    for i, inst in enumerate(instances):
        if int(inst.obj_id) != oid:
            continue
        a = float(inst.mask.sum())
        if a > best_area:
            best_area = a
            best_i = i
    return best_i


def _bop19_csv_line(
    scene_id: int,
    im_id: int,
    obj_id: int,
    score: float,
    R: np.ndarray,
    t_mm: np.ndarray,
    run_time: float,
) -> str:
    Rp = project_to_SO3(R)
    r_flat = Rp.reshape(-1).astype(np.float64)
    t_flat = np.asarray(t_mm, dtype=np.float64).reshape(3)
    r_str = " ".join(str(float(x)) for x in r_flat)
    t_str = " ".join(str(float(x)) for x in t_flat)
    return f"{int(scene_id)},{int(im_id)},{int(obj_id)},{float(score)},{r_str},{t_str},{float(run_time)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", help="BOP split 名，用于数据路径与结果文件名")
    ap.add_argument("--dataset", default="ycbv", help="数据集短名，写入结果文件名")
    ap.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="BOP19 目标列表 JSON（默认 <BOP_YCBV>/test_targets_bop19.json）",
    )
    ap.add_argument("--method", default="diffsam", help="方法名，结果文件 {method}_{dataset}-{split}.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 CSV 路径（默认 diff/output/{method}_{dataset}-{split}.csv）",
    )
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--require-ism", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-base", type=int, default=1315423911)
    ap.add_argument(
        "--policy",
        choices=("match_largest", "duplicate", "per_target"),
        default="match_largest",
        help="match_largest | per_target（每 target 一次采样，对齐 BOP 行数）| duplicate",
    )
    ap.add_argument("--max-images", type=int, default=None, help="仅处理前 N 个不同 (scene_id,im_id)，调试")
    args = ap.parse_args()

    targets_path = args.targets or (BOP_YCBV / "test_targets_bop19.json")
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)

    out_path = args.out
    if out_path is None:
        out_path = DIFF_OUT / f"{args.method}_{args.dataset}-{args.split}.csv"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    targets_raw = _load_targets(targets_path)

    by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    for rec in targets_raw:
        sid = int(rec["scene_id"])
        iid = int(rec["im_id"])
        oid = int(rec["obj_id"])
        by_image[(sid, iid)].append(oid)

    unique_pairs = sorted(by_image.keys())
    if args.max_images is not None:
        unique_pairs = unique_pairs[: args.max_images]

    if args.require_ism:
        from diff.ism_bridge import find_npz

        filt: list[tuple[int, int]] = []
        for sid, iid in unique_pairs:
            if find_npz(pred_dir, sid, iid) is not None:
                filt.append((sid, iid))
        unique_pairs = filt

    lines: list[str] = ["scene_id,im_id,obj_id,score,R,t,time"]
    t0 = time.perf_counter()
    n_infer = 0
    n_images_processed_meta = len(unique_pairs)

    if args.policy == "per_target":
        # 按 targets 文件顺序逐条处理，同 (scene_id,im_id) 共享一次加载与「整图 wall 时间」作为 time
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for ti, rec in enumerate(targets_raw):
            sid = int(rec["scene_id"])
            iid = int(rec["im_id"])
            oid = int(rec["obj_id"])
            groups[(sid, iid)].append((oid, ti))

        pair_list = sorted(groups.keys())
        if args.max_images is not None:
            pair_list = pair_list[: args.max_images]
        if args.require_ism:
            from diff.ism_bridge import find_npz

            pair_list = [(s, i) for s, i in pair_list if find_npz(pred_dir, s, i) is not None]
        n_images_processed_meta = len(pair_list)

        try:
            from tqdm import tqdm

            it = tqdm(pair_list, desc="bop19_export_per_target", mininterval=1.0)
        except ImportError:
            it = pair_list

        for sid, iid in it:
            t_img0 = time.perf_counter()
            pending: list[tuple[int, np.ndarray, np.ndarray]] = []
            try:
                rgb = load_rgb(args.split, sid, iid)
                instances = load_gt_instances(args.split, sid, iid)
            except (FileNotFoundError, KeyError):
                continue

            K = load_cam_K(args.split, sid, iid) if setup.include_cam_k else None

            for oid, ti in groups[(sid, iid)]:
                li_tgt = _largest_instance_index_for_obj_id(instances, oid)
                if li_tgt < 0:
                    continue
                cm = crop_mask_for_gt_instance(
                    rgb,
                    instances,
                    li_tgt,
                    pred_dir,
                    sid,
                    iid,
                    use_ism_crop=setup.include_ism_crop,
                )
                cond_np = build_pose_cond_vector(
                    rgb,
                    K,
                    include_cam_k=setup.include_cam_k,
                    include_obj_id_scalar=setup.include_obj_id_scalar,
                    obj_id=int(oid) if setup.include_obj_id_scalar else None,
                    include_ism_crop=setup.include_ism_crop,
                    crop_mask=cm if setup.include_ism_crop else None,
                )
                cond = torch.from_numpy(cond_np).unsqueeze(0).float()
                oid_t = None
                if obj_emb_dim > 0:
                    oid_t = torch.tensor([max(1, min(int(oid), 21))], dtype=torch.long)
                seed = int(
                    (args.seed_base + sid * 100003 + iid * 9176 + oid * 131 + ti * 17) & 0x7FFFFFFF
                )
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    x0_hat = sample_ddpm_eps(model, cond, betas=betas, device=device, obj_id=oid_t)
                n_infer += 1
                x_np = x0_hat.cpu().numpy().reshape(-1)
                if pose_repr == "vec9" or pose_dim == 9:
                    R_p, t_p = vec9_to_pose(x_np)
                else:
                    R_p, t_p = x12_to_pose(x_np)
                pending.append((oid, R_p, t_p))

            wall_img = time.perf_counter() - t_img0
            for oid, R_p, t_p in pending:
                lines.append(
                    _bop19_csv_line(
                        sid,
                        iid,
                        oid,
                        score=1.0,
                        R=R_p,
                        t_mm=t_p,
                        run_time=wall_img,
                    )
                )
    else:
        try:
            from tqdm import tqdm

            it = tqdm(unique_pairs, desc="bop19_export", mininterval=1.0)
        except ImportError:
            it = unique_pairs

        for sid, iid in it:
            seed = int((args.seed_base + sid * 100003 + iid * 9176) & 0x7FFFFFFF)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            try:
                rgb = load_rgb(args.split, sid, iid)
                instances = load_gt_instances(args.split, sid, iid)
            except (FileNotFoundError, KeyError):
                continue

            li = largest_instance(instances)
            if li < 0:
                continue
            inst = instances[li]
            largest_oid = int(inst.obj_id)

            K = load_cam_K(args.split, sid, iid) if setup.include_cam_k else None
            cm = crop_mask_for_gt_instance(
                rgb,
                instances,
                li,
                pred_dir,
                sid,
                iid,
                use_ism_crop=setup.include_ism_crop,
            )
            cond_np = build_pose_cond_vector(
                rgb,
                K,
                include_cam_k=setup.include_cam_k,
                include_obj_id_scalar=setup.include_obj_id_scalar,
                obj_id=int(largest_oid) if setup.include_obj_id_scalar else None,
                include_ism_crop=setup.include_ism_crop,
                crop_mask=cm if setup.include_ism_crop else None,
            )
            cond = torch.from_numpy(cond_np).unsqueeze(0).float()
            oid_t = None
            if obj_emb_dim > 0:
                oid_t = torch.tensor([max(1, min(int(largest_oid), 21))], dtype=torch.long)

            t_infer0 = time.perf_counter()
            with torch.no_grad():
                x0_hat = sample_ddpm_eps(model, cond, betas=betas, device=device, obj_id=oid_t)
            infer_dt = time.perf_counter() - t_infer0
            n_infer += 1

            x_np = x0_hat.cpu().numpy().reshape(-1)
            if pose_repr == "vec9" or pose_dim == 9:
                R_p, t_p = vec9_to_pose(x_np)
            else:
                R_p, t_p = x12_to_pose(x_np)

            target_obj_ids = by_image[(sid, iid)]
            for oid in target_obj_ids:
                if args.policy == "match_largest" and oid != largest_oid:
                    continue
                lines.append(
                    _bop19_csv_line(
                        sid,
                        iid,
                        oid,
                        score=1.0,
                        R=R_p,
                        t_mm=t_p,
                        run_time=infer_dt,
                    )
                )

    wall = time.perf_counter() - t0
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_rows = len(lines) - 1
    meta = {
        "checkpoint": str(args.checkpoint.resolve()),
        "targets": str(targets_path.resolve()),
        "out_csv": str(out_path),
        "policy": args.policy,
        "split": args.split,
        "n_unique_images_in_targets": len(by_image),
        "n_images_processed": n_images_processed_meta,
        "n_ddpm_forward": n_infer,
        "n_csv_data_rows": n_rows,
        "wall_time_sec": wall,
    }
    if args.policy == "per_target":
        meta["n_target_lines_in_json"] = len(targets_raw)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({n_rows} pose rows, {n_infer} inferences)", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
