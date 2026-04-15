#!/usr/bin/env bash
# vec9 + K + ISM 对齐裁剪 (+192) + obj Embedding（默认 16 维）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
conda run -n diffsam python diff/train_pose6d.py \
  --split test --require-ism --steps 20000 --batch-size 32 --num-workers 4 --device cuda \
  --ism-crop-cond --obj-emb-dim 16 \
  --save "$ROOT/diff/output/eps_mlp_pose6d_test_ism_camcrop_emb16_20k.pt" \
  --save-every 1000 \
  --log-file "$ROOT/diff/output/train_pose6d_ismcrop_emb16_20k.log"
