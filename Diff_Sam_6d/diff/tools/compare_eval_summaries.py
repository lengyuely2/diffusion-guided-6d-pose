#!/usr/bin/env python3
"""对比两份 eval_full_test 的 summary JSON（例如 ism x12 vs pose6d train_real）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _stat_line(d: dict, key: str) -> str:
    s = d.get(key, {})
    if not isinstance(s, dict):
        return "—"
    return f"mean={s.get('mean', 0):.2f}  med={s.get('median', 0):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=Path, help="第一次训练 eval summary，如 full_test_eval_summary.json")
    ap.add_argument("second", type=Path, help="新训练 eval summary")
    args = ap.parse_args()

    a = json.loads(args.first.read_text(encoding="utf-8"))
    b = json.loads(args.second.read_text(encoding="utf-8"))

    print("checkpoint A:", a.get("checkpoint"))
    print("checkpoint B:", b.get("checkpoint"))
    print()
    print(f"{'metric':<12} {'A (first)':<28} {'B (new)':<28}")
    print("-" * 70)
    for key, label in [
        ("trans_mm", "trans_mm"),
        ("rot_deg", "rot_deg"),
        ("score", "score"),
        ("x12_l2", "pose_vec_L2"),
    ]:
        print(f"{label:<12} {_stat_line(a, key):<28} {_stat_line(b, key):<28}")
    print()
    print("n_evaluated:", a.get("n_evaluated"), "|", b.get("n_evaluated"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
