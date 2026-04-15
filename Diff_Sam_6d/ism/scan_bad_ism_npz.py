#!/usr/bin/env python3
"""
扫描 ISM 输出目录中的 .npz（排除 *_runtime.npz），找出损坏/截断文件。

损坏的 npz 无法从磁盘“修复”，只能删除后重新跑 ISM（已有 ISM_SKIP_EXISTING_NPZ=1 时只会补缺失帧）。

用法:
  # 仅列出（默认）
  python ism/scan_bad_ism_npz.py

  # 删除损坏文件（删前会先打印数量）
  python ism/scan_bad_ism_npz.py --delete

  # 移到隔离目录而非删除
  python ism/scan_bad_ism_npz.py --move /tmp/bad_ism_npz

  # 指定目录
  python ism/scan_bad_ism_npz.py --pred-dir /path/to/result_ycbv_train_real
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# 与 diff/ism_bridge.load_ism_npz 一致的判断
try:
    _NPZ_EXC = (EOFError, OSError, zipfile.BadZipFile, ValueError)
except Exception:  # pragma: no cover
    _NPZ_EXC = (EOFError, OSError, ValueError)


def is_npz_readable(path: Path) -> bool:
    try:
        with np.load(str(path), allow_pickle=True) as z:
            for k in z.files:
                _ = np.asanyarray(z[k])
        return True
    except _NPZ_EXC:
        return False


def _npz_ok_str(path_str: str) -> tuple[str, bool]:
    """子进程用：可读则 (path, True)。强制读全部数组，避免仅元数据可读、mask 截断的情况。"""
    try:
        with np.load(path_str, allow_pickle=True) as z:
            for k in z.files:
                _ = np.asanyarray(z[k])
        return (path_str, True)
    except _NPZ_EXC:
        return (path_str, False)


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描并可选删除/隔离损坏的 ISM npz")
    root = Path(__file__).resolve().parents[1]
    default_pred = root / "Data" / "ISM_npz" / "ycbv_train_real"
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=default_pred,
        help=f"ISM npz 目录（默认 {default_pred}）",
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="删除不可读的 npz",
    )
    ap.add_argument(
        "--move",
        type=Path,
        default=None,
        help="将不可读 npz 移到此目录（与 --delete 二选一）",
    )
    ap.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="把坏文件路径写入该文本（一行一个）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多检查前 N 个文件（0 表示全部，调试用）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行进程数（0 表示 min(16, CPU)）",
    )
    args = ap.parse_args()
    pred = args.pred_dir.resolve()
    if not pred.is_dir():
        print(f"目录不存在: {pred}", file=sys.stderr)
        return 1
    if args.delete and args.move:
        print("不能同时 --delete 与 --move", file=sys.stderr)
        return 1

    files = sorted(
        p
        for p in pred.iterdir()
        if p.is_file()
        and p.suffix == ".npz"
        and "_runtime" not in p.name
    )
    if args.limit > 0:
        files = files[: args.limit]

    workers = args.workers or min(16, max(1, os.cpu_count() or 4))
    bad: list[Path] = []
    if workers <= 1 or len(files) < 100:
        for i, p in enumerate(files):
            if (i + 1) % 5000 == 0:
                print(f"已检查 {i+1}/{len(files)} ...", flush=True, file=sys.stderr)
            if not is_npz_readable(p):
                bad.append(p)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for path_str, ok in ex.map(
                _npz_ok_str,
                map(str, files),
                chunksize=max(8, len(files) // (workers * 32) or 1),
            ):
                done += 1
                if done % 5000 == 0:
                    print(
                        f"已检查 {done}/{len(files)} (workers={workers}) ...",
                        flush=True,
                        file=sys.stderr,
                    )
                if not ok:
                    bad.append(Path(path_str))

    print(f"总计 npz: {len(files)}  损坏: {len(bad)}")
    if args.list_file is not None:
        args.list_file.parent.mkdir(parents=True, exist_ok=True)
        args.list_file.write_text("\n".join(str(p) for p in bad) + ("\n" if bad else ""))
        print(f"列表已写入: {args.list_file}")

    if not bad:
        return 0

    for p in bad[:20]:
        print(f"  BAD {p}")
    if len(bad) > 20:
        print(f"  ... 另有 {len(bad) - 20} 个")

    if args.move is not None:
        args.move.mkdir(parents=True, exist_ok=True)
        for p in bad:
            shutil.move(str(p), str(args.move / p.name))
            rt = p.parent / f"{p.stem}_runtime.npz"
            if rt.is_file():
                shutil.move(str(rt), str(args.move / rt.name))
        print(f"已移动 {len(bad)} 个主 npz（及同名 _runtime 若存在）到 {args.move}")
        print("下一步: bash ism/run_ycbv_train_real_ism.sh  （会补跑缺失 npz）")
    elif args.delete:
        for p in bad:
            p.unlink(missing_ok=True)
        # 与 detector 一致：主文件 sceneX_frameY.npz 对应 sceneX_frameY_runtime.npz
        for p in bad:
            rt = p.parent / f"{p.stem}_runtime.npz"
            if rt.is_file():
                rt.unlink(missing_ok=True)
        print(f"已删除 {len(bad)} 个损坏 npz（及同名 _runtime 若存在）")
        print("下一步: bash ism/run_ycbv_train_real_ism.sh  （会补跑缺失 npz）")
    else:
        print("（未执行删除/移动）加 --delete 或 --move <dir> 可清理后重跑 ISM")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
