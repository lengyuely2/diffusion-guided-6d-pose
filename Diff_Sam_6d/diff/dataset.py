"""YCB-V test + ISM npz → 扩散训练样本（RGB 下采样条件 + GT pose 向量）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bop_frame import largest_instance, load_cam_K, load_gt_instances, load_rgb
from .geometry import pose_to_vec9
from .ism_bridge import find_npz, load_ism_npz, match_same_class_iou
from .paths import BOP_YCBV, official_ism_predictions_dir


RGB_COND_DIM = 8 * 8 * 3
CAM_K_COND_DIM = 4
OBJ_ID_COND_DIM = 1
YCBV_OBJ_ID_NORM_MAX = 21
YCBV_NUM_OBJ_EMB = 22  # 0 保留，1..21 对应物体 id

CROP_PAD_FRAC = 0.15


def _rgb_cond(rgb: np.ndarray, size: int = 8) -> np.ndarray:
    """RGB uint8 -> (size*size*3,) float32 [0,1]."""
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def cam_k_normalized(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """fx,fy,cx,cy 按图像宽高归一，与下采样 RGB 条件一起用。"""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([fx / max(W, 1), fy / max(H, 1), cx / max(W, 1), cy / max(H, 1)], dtype=np.float32)


def obj_id_cond_value(obj_id: int) -> np.ndarray:
    oid = int(max(1, min(int(obj_id), YCBV_OBJ_ID_NORM_MAX)))
    return np.array([oid / float(YCBV_OBJ_ID_NORM_MAX)], dtype=np.float32)


def _mask_bbox_xyxy(mask: np.ndarray, pad_frac: float = CROP_PAD_FRAC) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    H, W = mask.shape[:2]
    if len(xs) == 0:
        return 0, 0, W, H
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    pad = int(max(bw, bh) * pad_frac + 0.5)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    return x1, y1, x2, y2


def rgb_crop_cond(rgb: np.ndarray, mask: np.ndarray, size: int = 8, pad_frac: float = CROP_PAD_FRAC) -> np.ndarray:
    """按 mask 外接框（带边距）裁 RGB 并缩放到 size×size，展平 [0,1]。"""
    x1, y1, x2, y2 = _mask_bbox_xyxy(mask, pad_frac)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def ism_mask_for_pred_index(data: dict, j: int, H: int, W: int) -> np.ndarray:
    seg = data["segmentation"]
    if seg.ndim == 2:
        pm = seg.astype(np.float32)
    else:
        pm = seg[int(j)].astype(np.float32)
    if pm.shape != (H, W):
        pm = cv2.resize(pm, (W, H), interpolation=cv2.INTER_NEAREST)
    return pm


def crop_mask_for_gt_instance(
    rgb: np.ndarray,
    instances: list,
    li: int,
    pred_dir: Path,
    sid: int,
    fid: int,
    *,
    use_ism_crop: bool,
) -> np.ndarray:
    """训练/推理：GT 实例 li 的裁剪用 mask；若 use_ism 且匹配到 ISM 同类预测则用预测 mask。"""
    inst = instances[li]
    mask = inst.mask
    if not use_ism_crop:
        return mask
    npz = find_npz(pred_dir, sid, fid)
    if npz is None:
        return mask
    data = load_ism_npz(npz)
    if data is None:
        return mask
    gt_masks = [x.mask for x in instances]
    gt_ids = [x.obj_id for x in instances]
    best_idx, _ = match_same_class_iou(
        gt_masks,
        gt_ids,
        data["category_id"],
        data["segmentation"],
    )
    j = best_idx[li]
    if j < 0:
        return mask
    H, W = int(rgb.shape[0]), int(rgb.shape[1])
    return ism_mask_for_pred_index(data, j, H, W)


def build_pose_cond_vector(
    rgb: np.ndarray,
    K: np.ndarray | None,
    *,
    include_cam_k: bool,
    include_obj_id_scalar: bool,
    obj_id: int | None,
    include_ism_crop: bool,
    crop_mask: np.ndarray | None,
) -> np.ndarray:
    parts: list[np.ndarray] = [_rgb_cond(rgb)]
    if include_ism_crop:
        if crop_mask is None:
            raise ValueError("include_ism_crop=True 时需要 crop_mask")
        parts.append(rgb_crop_cond(rgb, crop_mask))
    if include_cam_k:
        if K is None:
            raise ValueError("include_cam_k=True 时需要 K")
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        parts.append(cam_k_normalized(K, H, W))
    if include_obj_id_scalar:
        if obj_id is None:
            raise ValueError("include_obj_id_scalar=True 时需要 obj_id")
        parts.append(obj_id_cond_value(obj_id))
    return np.concatenate(parts, axis=0)


def visual_cond(
    rgb: np.ndarray,
    K: np.ndarray | None,
    *,
    include_cam_k: bool = True,
    include_obj_id: bool = False,
    obj_id: int | None = None,
) -> np.ndarray:
    """仅全局 RGB + 可选 K + 可选标量 obj_id（无 ISM crop，兼容旧脚本）。"""
    return build_pose_cond_vector(
        rgb,
        K,
        include_cam_k=include_cam_k,
        include_obj_id_scalar=include_obj_id,
        obj_id=obj_id,
        include_ism_crop=False,
        crop_mask=None,
    )


def total_cond_dim(
    *,
    include_cam_k: bool,
    include_obj_id_scalar: bool = False,
    include_ism_crop: bool = False,
) -> int:
    n = RGB_COND_DIM
    if include_ism_crop:
        n += RGB_COND_DIM
    if include_cam_k:
        n += CAM_K_COND_DIM
    if include_obj_id_scalar:
        n += OBJ_ID_COND_DIM
    return n


@dataclass(frozen=True)
class CondSetup:
    """与 checkpoint / 推理一致的条件布局（不含 nn.Embedding，embedding 在模型内）。"""

    cond_dim: int
    include_cam_k: bool
    include_obj_id_scalar: bool
    include_ism_crop: bool
    obj_emb_dim: int


def infer_cond_setup_from_ckpt(ckpt: dict) -> CondSetup:
    cd = int(ckpt.get("cond_dim", RGB_COND_DIM))
    emb = int(ckpt.get("obj_emb_dim", 0))
    crop = bool(ckpt.get("include_ism_crop", False))
    ik = ckpt.get("include_cam_k")
    ios = ckpt.get("include_obj_id")
    if ik is not None and ios is not None:
        return CondSetup(cd, bool(ik), bool(ios), crop, emb)
    rem = cd - RGB_COND_DIM
    if rem == 0:
        return CondSetup(cd, False, False, False, emb)
    if rem == OBJ_ID_COND_DIM:
        return CondSetup(cd, False, True, False, emb)
    if rem == CAM_K_COND_DIM:
        return CondSetup(cd, True, False, False, emb)
    if rem == CAM_K_COND_DIM + OBJ_ID_COND_DIM:
        return CondSetup(cd, True, True, False, emb)
    if rem == RGB_COND_DIM + CAM_K_COND_DIM:
        return CondSetup(cd, True, False, True, emb)
    return CondSetup(cd, cd > RGB_COND_DIM, False, False, emb)


def _meta_pairs(split: str) -> list[tuple[int, int]]:
    meta_path = BOP_YCBV / f"{split}_metaData.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    scenes = [int(s) for s in data["scene_id"]]
    frames = [int(f) for f in data["frame_id"]]
    return list(zip(scenes, frames))


class YcbvIsmPoseDataset(Dataset):
    """
    每个样本：一张图里取实例（largest 或 random_visible），条件为全局 RGB + 可选 ISM 对齐裁剪 + K + 可选 obj。
    pose_repr: "x12"（默认）或 "vec9"（6D 旋转 + t/500）。
    """

    def __init__(
        self,
        split: str = "test",
        pred_dir: Path | None = None,
        *,
        indices: list[int] | None = None,
        max_samples: int | None = None,
        require_ism: bool = True,
        pose_repr: str = "x12",
        include_cam_k: bool = True,
        include_obj_id: bool = False,
        include_ism_crop: bool = False,
        use_obj_embedding: bool = False,
        instance_mode: str = "largest",
    ):
        self.split = split
        self.pred_dir = Path(pred_dir) if pred_dir else official_ism_predictions_dir()
        self.require_ism = require_ism
        self.include_cam_k = include_cam_k
        self.include_obj_id_scalar = bool(include_obj_id)
        self.include_ism_crop = bool(include_ism_crop)
        self.use_obj_embedding = bool(use_obj_embedding)
        if self.include_obj_id_scalar and self.use_obj_embedding:
            raise ValueError("不能同时 use_obj_embedding 与标量 obj_id 条件")
        if instance_mode not in ("largest", "random_visible"):
            raise ValueError("instance_mode must be 'largest' or 'random_visible'")
        self.instance_mode = instance_mode
        if pose_repr not in ("x12", "vec9"):
            raise ValueError("pose_repr must be 'x12' or 'vec9'")
        self.pose_repr = pose_repr
        pairs = _meta_pairs(split)
        if max_samples is not None:
            pairs = pairs[:max_samples]
        self._pairs = pairs
        if indices is not None:
            self._pairs = [pairs[i] for i in indices if 0 <= i < len(pairs)]
        self._valid: list[int] = []
        self._build_index()

    def _build_index(self) -> None:
        self._valid = []
        for i, (sid, fid) in enumerate(self._pairs):
            if not self.require_ism:
                self._valid.append(i)
                continue
            npz = find_npz(self.pred_dir, sid, fid)
            if npz is not None:
                self._valid.append(i)

    def __len__(self) -> int:
        return len(self._valid)

    def raw_index(self, i: int) -> tuple[int, int]:
        j = self._valid[i]
        return self._pairs[j]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sid, fid = self.raw_index(i)
        rgb = load_rgb(self.split, sid, fid)
        instances = load_gt_instances(self.split, sid, fid)
        if self.instance_mode == "largest":
            li = largest_instance(instances)
        else:
            vis = [j for j, x in enumerate(instances) if int(x.mask.sum()) > 0]
            if not vis:
                li = largest_instance(instances)
            else:
                li = int(np.random.choice(vis))
        inst = instances[li]

        K = load_cam_K(self.split, sid, fid) if self.include_cam_k else None
        crop_mask = crop_mask_for_gt_instance(
            rgb, instances, li, self.pred_dir, sid, fid, use_ism_crop=self.include_ism_crop
        )
        cond = build_pose_cond_vector(
            rgb,
            K,
            include_cam_k=self.include_cam_k,
            include_obj_id_scalar=self.include_obj_id_scalar,
            obj_id=int(inst.obj_id) if self.include_obj_id_scalar else None,
            include_ism_crop=self.include_ism_crop,
            crop_mask=crop_mask if self.include_ism_crop else None,
        )
        if self.pose_repr == "vec9":
            x0 = torch.from_numpy(pose_to_vec9(inst.R_m2c, inst.t_mm))
        else:
            x0 = torch.from_numpy(inst.pose_x12.copy())

        npz_path = find_npz(self.pred_dir, sid, fid)
        mask_iou = torch.tensor(0.0)
        if npz_path is not None:
            data = load_ism_npz(npz_path)
            if data is not None:
                gt_masks = [x.mask for x in instances]
                gt_ids = [x.obj_id for x in instances]
                _, ious = match_same_class_iou(
                    gt_masks,
                    gt_ids,
                    data["category_id"],
                    data["segmentation"],
                )
                mask_iou = torch.tensor(ious[li], dtype=torch.float32)

        cond_t = torch.from_numpy(cond)
        out: dict[str, torch.Tensor] = {
            "x0": x0,
            "cond": cond_t,
            "mask_iou": mask_iou,
            "scene_id": torch.tensor(sid, dtype=torch.long),
            "frame_id": torch.tensor(fid, dtype=torch.long),
        }
        if self.use_obj_embedding:
            oid = int(max(1, min(int(inst.obj_id), YCBV_OBJ_ID_NORM_MAX)))
            out["obj_id"] = torch.tensor(oid, dtype=torch.long)
        return out
