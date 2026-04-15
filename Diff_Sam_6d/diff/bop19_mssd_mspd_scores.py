#!/usr/bin/env python3
"""
对已由 eval_calc_errors.py 算好的 MSSD / MSPD 误差目录，按 BOP19 与 eval_bop19_pose.py
相同的阈值集合分别调用 eval_calc_scores.py，再对 recall 取平均（各误差类型 10 个点再 mean）。

不含 VSD（需 EGL / bop_renderer），适合 WSL 无显示环境。

用法:
  python -m diff.bop19_mssd_mspd_scores \\
    --result-stem pose6dism_ycbv-test \\
    --eval-path diff/output/bop_eval \\
    --datasets-path Data/BOP
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _find_scores_json(error_dir: Path) -> Path:
    cands = sorted(error_dir.glob("scores_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no scores_*.json under {error_dir}")
    return cands[0]


def _run_scores(
    *,
    eval_scores_py: Path,
    result_stem: str,
    err_sig: str,
    err_type: str,
    threshold: float,
    eval_path: Path,
    datasets_path: Path,
) -> float:
    rel = f"{result_stem}/{err_sig}"
    cmd = [
        sys.executable,
        str(eval_scores_py),
        f"--error_dir_paths={rel}",
        f"--eval_path={str(eval_path)}",
        f"--datasets_path={str(datasets_path)}",
        "--targets_filename=test_targets_bop19.json",
        f"--correct_th_{err_type}={threshold}",
    ]
    env = {**os.environ, "BOP_PATH": str(datasets_path)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"eval_calc_scores failed:\ncmd={' '.join(cmd)}\nstderr={r.stderr[-4000:]}\nstdout={r.stdout[-2000:]}"
        )
    ed = eval_path / result_stem / err_sig
    sf = _find_scores_json(ed)
    data = json.loads(sf.read_text(encoding="utf-8"))
    return float(data["recall"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-stem", required=True, help="与 CSV 主文件名一致，无扩展名，如 pose6dism_ycbv-test")
    ap.add_argument("--eval-path", type=Path, required=True)
    ap.add_argument("--datasets-path", type=Path, required=True)
    ap.add_argument(
        "--bop-toolkit-root",
        type=Path,
        default=_PKG / "bop_toolkit",
        help="含 scripts/eval_calc_scores.py 的目录",
    )
    args = ap.parse_args()

    eval_scores_py = args.bop_toolkit_root / "scripts" / "eval_calc_scores.py"
    if not eval_scores_py.is_file():
        print(f"missing {eval_scores_py}", file=sys.stderr)
        return 1

    eval_path = args.eval_path.resolve()
    ds = args.datasets_path.resolve()

    mssd_sig = "error=mssd_ntop=-1"
    mspd_sig = "error=mspd_ntop=-1"
    m_th = list(np.arange(0.05, 0.51, 0.05))
    p_th = list(np.arange(5, 51, 5))

    stem = args.result_stem
    m_recalls = [
        _run_scores(
            eval_scores_py=eval_scores_py,
            result_stem=stem,
            err_sig=mssd_sig,
            err_type="mssd",
            threshold=float(t),
            eval_path=eval_path,
            datasets_path=ds,
        )
        for t in m_th
    ]
    p_recalls = [
        _run_scores(
            eval_scores_py=eval_scores_py,
            result_stem=stem,
            err_sig=mspd_sig,
            err_type="mspd",
            threshold=float(t),
            eval_path=eval_path,
            datasets_path=ds,
        )
        for t in p_th
    ]

    am = float(np.mean(m_recalls))
    ap_ = float(np.mean(p_recalls))
    out = {
        "result_stem": stem,
        "bop19_average_recall_mssd": am,
        "bop19_average_recall_mspd": ap_,
        "bop19_proxy_avg_without_vsd": (am + ap_) / 2.0,
        "mssd_thresholds": [float(x) for x in m_th],
        "mssd_recalls": m_recalls,
        "mspd_thresholds": [float(x) for x in p_th],
        "mspd_recalls": p_recalls,
        "note": "不含 VSD；与 leaderboard 的 bop19_average_recall（含 VSD）不可直接等同。",
    }
    out_path = eval_path / stem / "scores_bop19_mssd_mspd_only.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("result_stem", "bop19_average_recall_mssd", "bop19_average_recall_mspd", "bop19_proxy_avg_without_vsd")}, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
