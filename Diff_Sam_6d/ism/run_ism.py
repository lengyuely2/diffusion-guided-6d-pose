#!/usr/bin/env python3
"""
官方 Instance Segmentation Model 推理入口提示（本仓库不内嵌 SAM-6D 权重与依赖）。

在 SAM-6D 仓库中运行推理，例如：
  cd <SAM-6D>/Instance_Segmentation_Model
  # 按官方 README 准备 ycbv 数据与 checkpoint 后执行 run_inference_custom.py 或对应脚本
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ism.paths import DIFF_SAM_ROOT, official_ism_predictions_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="打印 ISM 预测目录与官方工程路径提示")
    ap.add_argument("--show-inference-script", action="store_true", help="尝试定位 run_inference_custom.py")
    args = ap.parse_args()

    pred = official_ism_predictions_dir()
    print("ISM 预测 npz 默认输出目录（可用环境变量 SAM6D_ISM_ROOT 覆盖官方仓库根）:")
    print(pred)
    print()

    official_ism = (
        DIFF_SAM_ROOT.parent
        / "SAM-6D-official"
        / "SAM-6D"
        / "Instance_Segmentation_Model"
    ).resolve()
    print("默认官方 ISM 根目录（与 diff.paths / ism.paths 一致）:")
    print(official_ism)

    if args.show_inference_script:
        cand = official_ism / "run_inference_custom.py"
        print()
        print("常见入口脚本:")
        print(cand)
        print("exists:", cand.is_file())

    print()
    print("请将官方仓库中生成的 ycbv 预测保存到上述 predictions 目录，供 diff / ism 诊断脚本读取。")
    print()
    print("若要对 YCB-V train_real 跑 ISM（供 diffusion 大训练用 npz）：")
    print("  1) 在 Instance_Segmentation_Model 下已补齐 run_inference.py，并支持")
    print("     data.query_dataloader.split=train_real 与 prediction_subdir=...")
    print("  2) bash ism/run_ycbv_train_real_ism.sh  （需 sam6d-ism 等环境）")
    print("  3) bash ism/link_train_real_ism_npz.sh")
    print("  4) bash diff/run_train_trainreal_ismcrop_large.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
