"""读取 YCB-V 单帧 RGB、相机内参与 GT 实例（含 m2c 位姿与 mask）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import R9_to_matrix, pose_to_x12
from .paths import BOP_YCBV


@dataclass
class GtInstance:
    obj_id: int
    R_m2c: np.ndarray  # 3x3
    t_mm: np.ndarray  # 3
    mask: np.ndarray  # H,W float 0/1
    pose_x12: np.ndarray  # 12


def scene_dir(split: str, scene_id: int) -> Path:
    return BOP_YCBV / split / f"{int(scene_id):06d}"


def load_rgb(split: str, scene_id: int, frame_id: int) -> np.ndarray:
    p = scene_dir(split, scene_id) / "rgb" / f"{int(frame_id):06d}.png"
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_cam_K(split: str, scene_id: int, frame_id: int) -> np.ndarray:
    cam_path = scene_dir(split, scene_id) / "scene_camera.json"
    data = json.loads(cam_path.read_text(encoding="utf-8"))
    key = str(int(frame_id))
    if key not in data:
        raise KeyError(f"frame {key} not in scene_camera.json")
    K = np.asarray(data[key]["cam_K"], dtype=np.float64).reshape(3, 3)
    return K


def load_gt_instances(split: str, scene_id: int, frame_id: int) -> list[GtInstance]:
    sdir = scene_dir(split, scene_id)
    gt_path = sdir / "scene_gt.json"
    scene_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    key = str(int(frame_id))
    if key not in scene_gt:
        raise KeyError(f"frame {key} not in scene_gt")
    objs = scene_gt[key]
    out: list[GtInstance] = []
    for inst_i, rec in enumerate(objs):
        oid = int(rec["obj_id"])
        R = R9_to_matrix(rec["cam_R_m2c"])
        t = np.asarray(rec["cam_t_m2c"], dtype=np.float64).reshape(3)
        mpath = sdir / "mask_visib" / f"{int(frame_id):06d}_{inst_i:06d}.png"
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mpath)
        mask = (m > 127).astype(np.float32)
        x12 = pose_to_x12(R, t)
        out.append(GtInstance(oid, R.astype(np.float32), t.astype(np.float32), mask, x12))
    return out


def largest_instance(instances: list[GtInstance]) -> int:
    """可见像素最多的实例下标。"""
    if not instances:
        return -1
    areas = [float(inst.mask.sum()) for inst in instances]
    return int(np.argmax(areas))
