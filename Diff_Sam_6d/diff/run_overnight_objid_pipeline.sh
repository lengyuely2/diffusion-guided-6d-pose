#!/usr/bin/env bash
# 训练 vec9+K+obj_id → BOP19 per_target 导出 → eval_bop19_pose (cpp renderer)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BOP_RENDERER_PATH="${BOP_RENDERER_PATH:-$ROOT/bop_renderer/build}"
LOG="$ROOT/diff/output/overnight_objid_pipeline.log"
CKPT="$ROOT/diff/output/eps_mlp_pose6d_test_ism_camk_objid_20k.pt"
CSV_BASENAME="diffsamobjidpt_ycbv-test.csv"
OUT_CSV="$ROOT/diff/output/bop_results/$CSV_BASENAME"

{
  echo "=== train $(date -Is) ==="
  conda run -n diffsam python diff/train_pose6d.py \
    --split test --require-ism --steps 20000 --batch-size 32 --num-workers 4 --device cuda \
    --save "$CKPT" --save-every 1000 --cond-obj-id \
    --log-file "$ROOT/diff/output/train_pose6d_test_ism_camk_objid_20k.log"

  echo "=== export per_target $(date -Is) ==="
  mkdir -p "$ROOT/diff/output/bop_results"
  conda run -n diffsam python diff/export_bop19_results.py \
    -c "$CKPT" --require-ism --policy per_target \
    --method diffsamobjidpt \
    --out "$OUT_CSV"

  echo "=== bop19 eval $(date -Is) ==="
  conda run -n diffsam python bop_toolkit/scripts/eval_bop19_pose.py \
    --renderer_type cpp \
    --result_filenames "$CSV_BASENAME" \
    --results_path "$ROOT/diff/output/bop_results" \
    --eval_path "$ROOT/diff/output/bop_eval" \
    --num_workers 4

  echo "=== done $(date -Is) ==="
} >>"$LOG" 2>&1
