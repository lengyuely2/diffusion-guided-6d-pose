#!/usr/bin/env bash
# 将 ISM 在 train_real 上生成的 npz 目录链到 Diff 工程固定路径，供 train_pose6d --pred-dir 使用。
set -euo pipefail
DIFF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISM_ROOT="${ISM_ROOT:-$DIFF_ROOT/../SAM-6D-official/SAM-6D/Instance_Segmentation_Model}"
NAME_EXP="${NAME_EXP:-train_real_ism}"
PRED_SUBDIR="${PRED_SUBDIR:-result_ycbv_train_real}"
SRC="${ISM_ROOT}/log/${NAME_EXP}/predictions/ycbv/${PRED_SUBDIR}"
DST="${DIFF_ROOT}/Data/ISM_npz/ycbv_train_real"

if [[ ! -d "$SRC" ]]; then
  echo "找不到 ISM 输出目录: $SRC" >&2
  echo "请先跑: bash ism/run_ycbv_train_real_ism.sh（且 name_exp/pred_subdir 与上述一致）" >&2
  exit 1
fi
mkdir -p "$(dirname "$DST")"
ln -sfn "$SRC" "$DST"
echo "OK: $DST -> $SRC"
echo "训练时: PRED_DIR=$DST bash diff/run_train_trainreal_ismcrop_large.sh"
echo "或已写入 diff 脚本默认 PRED_DIR（若存在该 symlink）"
