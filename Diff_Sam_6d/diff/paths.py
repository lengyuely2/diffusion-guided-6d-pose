"""项目根、BOP 与 ISM 预测目录（与 ism/paths 约定一致）。"""

from __future__ import annotations

from pathlib import Path

DIFF_PKG = Path(__file__).resolve().parent
DIFF_SAM_ROOT = DIFF_PKG.parent
DATA_ROOT = DIFF_SAM_ROOT / "Data"
BOP_YCBV = DATA_ROOT / "BOP" / "ycbv"


def official_ism_predictions_dir() -> Path:
    import os

    root = os.environ.get("SAM6D_ISM_ROOT")
    if root:
        base = Path(root).resolve()
    else:
        base = (
            DIFF_SAM_ROOT.parent
            / "SAM-6D-official"
            / "SAM-6D"
            / "Instance_Segmentation_Model"
        ).resolve()
    return base / "log" / "sam" / "predictions" / "ycbv" / "result_ycbv"
