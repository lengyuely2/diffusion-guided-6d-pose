#!/usr/bin/env bash
# 在 train_real 上训练完成后，对 test 全量评估 + 最好/最差可视化，并与第一次 ism x12 对比。
# 用法：
#   chmod +x diff/tools/run_post_train_eval_and_vis.sh
#   export CKPT=diff/output/eps_mlp_pose6d_trainreal_20k.pt
#   ./diff/tools/run_post_train_eval_and_vis.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CKPT="${CKPT:-diff/output/eps_mlp_pose6d_trainreal_20k.pt}"
OUT_JSON="${OUT_JSON:-diff/output/full_test_eval_pose6d_trainreal_summary.json}"
OUT_CSV="${OUT_CSV:-diff/output/full_test_eval_pose6d_trainreal_per_frame.csv}"
VIS_DIR="${VIS_DIR:-diff/output/pose_eval_vis_box3d_trainreal}"

if [[ ! -f "$CKPT" ]]; then
  echo "找不到 checkpoint: $CKPT（若训练未完成，可用 eps_mlp_pose6d_trainreal_20k_last.pt）" >&2
  exit 1
fi

conda run -n diffsam python diff/eval_full_test.py \
  -c "$CKPT" \
  --split test \
  --device cuda \
  --out-json "$OUT_JSON" \
  --out-csv "$OUT_CSV"

conda run -n diffsam python diff/eval_vis_best_worst.py \
  -c "$CKPT" \
  --split test \
  --n-requested 80 \
  --top-k 3 \
  --seed 42 \
  --out-dir "$VIS_DIR"

echo "--- 与第一次 ism x12 全量结果对比 ---"
conda run -n diffsam python diff/tools/compare_eval_summaries.py \
  diff/output/full_test_eval_summary.json \
  "$OUT_JSON"

echo "评估 JSON: $OUT_JSON"
echo "可视化目录: $VIS_DIR"
