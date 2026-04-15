#!/usr/bin/env bash
# 公平对比用：与 eps_mlp_ism_20k.pt 同一数据管线 — test + --require-ism（vec9 / pose6d）
# 用法：bash diff/tools/train_pose6d_test_ism_fair.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-diff/output/eps_mlp_pose6d_test_ism_20k.pt}"
conda run -n diffsam python diff/train_pose6d.py \
  --split test \
  --require-ism \
  --steps 20000 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --save "$OUT" \
  --save-every 2000
