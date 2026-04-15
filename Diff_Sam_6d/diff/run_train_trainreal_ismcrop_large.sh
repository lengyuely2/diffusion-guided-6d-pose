#!/usr/bin/env bash
# 在 train_real 大数据上训练 vec9 + K + ISM 裁剪 + obj embedding（与大显存匹配的大 batch）。
# 用法：
#   bash diff/run_train_trainreal_ismcrop_large.sh
#   BS=256 STEPS=80000 bash diff/run_train_trainreal_ismcrop_large.sh
#   RESUME=diff/output/xxx_last.pt BS=128 bash diff/run_train_trainreal_ismcrop_large.sh
#
# 说明：
# - 数据目录需存在 Data/BOP/ycbv/train_real_metaData.json（本仓库为 train_real，无单独 train_metaData）。
# - --require-ism 只保留 pred-dir 里有 npz 的帧。
# - 默认 PRED_DIR=Data/ISM_npz/ycbv_train_real（可先 bash ism/link_train_real_ism_npz.sh）。
# - 超大 batch 时可酌情提高学习率，例如 BS=256 时试 LR=2e-3：LR=0.002 bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BS="${BS:-128}"
STEPS="${STEPS:-50000}"
NW="${NW:-8}"
LR="${LR:-0.001}"
PRED_DIR="${PRED_DIR:-$ROOT/Data/ISM_npz/ycbv_train_real}"
EXTRA=(--pred-dir "$PRED_DIR")
if [[ -n "${RESUME:-}" ]]; then
  if [[ "${RESUME}" = /* ]]; then
    EXTRA+=(--resume "${RESUME}")
  else
    EXTRA+=(--resume "${ROOT}/${RESUME}")
  fi
fi

exec conda run -n diffsam python diff/train_pose6d.py \
  --split train_real \
  --require-ism \
  "${EXTRA[@]}" \
  --batch-size "$BS" \
  --num-workers "$NW" \
  --steps "$STEPS" \
  --lr "$LR" \
  --device cuda \
  --ism-crop-cond \
  --obj-emb-dim 16 \
  --save "$ROOT/diff/output/eps_mlp_pose6d_trainreal_ismcrop_emb16_${STEPS}.pt" \
  --save-every 2000 \
  --log-file "$ROOT/diff/output/train_pose6d_trainreal_ismcrop_emb16_${STEPS}.log"
