#!/usr/bin/env bash
# 对若干 checkpoint 在 test 上全量 eval，并两两对比（相对 REF 第一个）
# 用法：
#   bash diff/tools/run_fair_eval_compare.sh
# 或环境变量：
#   CKPTS="diff/output/eps_mlp_ism_20k.pt diff/output/eps_mlp_pose6d.pt"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPTS="${CKPTS:-diff/output/eps_mlp_ism_20k.pt diff/output/eps_mlp_pose6d.pt diff/output/eps_mlp_pose6d_test_ism_20k.pt}"

SUMMARIES=()
i=0
for ckpt in $CKPTS; do
  if [[ ! -f "$ckpt" ]]; then
    echo "skip (missing): $ckpt" >&2
    continue
  fi
  i=$((i + 1))
  stem=$(basename "$ckpt" .pt)
  sj="diff/output/eval_${stem}_summary.json"
  sc="diff/output/eval_${stem}_per_frame.csv"
  echo "=== eval $ckpt -> $sj ==="
  conda run -n diffsam python diff/eval_full_test.py -c "$ckpt" --split test --device cuda \
    --out-json "$sj" --out-csv "$sc"
  SUMMARIES+=("$sj")
done

if [[ ${#SUMMARIES[@]} -lt 1 ]]; then
  echo "No checkpoints evaluated." >&2
  exit 1
fi

REF="${SUMMARIES[0]}"
echo ""
echo "========== 相对 REF（第一个）: $REF =========="
for s in "${SUMMARIES[@]}"; do
  if [[ "$s" == "$REF" ]]; then
    echo "--- same as REF ---"
    continue
  fi
  conda run -n diffsam python diff/tools/compare_eval_summaries.py "$REF" "$s"
  echo ""
done
