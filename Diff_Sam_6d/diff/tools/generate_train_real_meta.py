#!/usr/bin/env python3
"""生成 train_real_metaData.json（与 test_metaData.json 同格式）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from diff.paths import BOP_YCBV


def main() -> int:
    root = BOP_YCBV / "train_real"
    scene_ids: list[str] = []
    frame_ids: list[str] = []
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        cam = sd / "scene_camera.json"
        if not cam.is_file():
            continue
        data = json.loads(cam.read_text(encoding="utf-8"))
        sid = sd.name
        for fid in sorted(data.keys(), key=lambda x: int(x)):
            scene_ids.append(sid)
            frame_ids.append(str(int(fid)).zfill(6))
    out = BOP_YCBV / "train_real_metaData.json"
    out.write_text(
        json.dumps({"scene_id": scene_ids, "frame_id": frame_ids}, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}  n={len(scene_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
