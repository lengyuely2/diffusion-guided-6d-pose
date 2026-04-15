#!/usr/bin/env bash
# 在 YCB-V train_real 上跑官方 ISM，生成与 diff 训练兼容的 npz。
#
# 前置：
#   - BOP 数据在 Diff_Sam_6d/Data/BOP/ycbv/（含 train_real/ 与 train_real_metaData.json）
#   - ISM 权重与模板按 SAM-6D Instance_Segmentation_Model README 准备完毕
#
# 输出：
#   $ISM_ROOT/log/${NAME_EXP}/predictions/ycbv/${PRED_SUBDIR}/*.npz
#
# 环境：
#   默认用 conda 环境 ISM_CONDA_ENV（默认 diffsam，含 torch / PL / segment_anything）。
#   已手动 activate 时：USE_CONDA_RUN=0 bash ...
#
# 单卡（默认）：
#   CUDA_VISIBLE_DEVICES=0（可改）+ machine.trainer=local_gloo（WSL 上 NCCL 易挂，用 gloo）+ devices=1
#   原生 Linux 且 NCCL 正常：TRAINER_PROFILE=local bash ...
#
# 试跑少量 batch：
#   LIMIT_TEST_BATCHES=50 bash ism/run_ycbv_train_real_ism.sh
#
# 用法：
#   bash ism/run_ycbv_train_real_ism.sh
#   CUDA_VISIBLE_DEVICES=0 NAME_EXP=myism bash ism/run_ycbv_train_real_ism.sh
#   追加 Hydra：bash ism/run_ycbv_train_real_ism.sh machine.batch_size=8
set -euo pipefail

DIFF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISM_ROOT="${ISM_ROOT:-$DIFF_ROOT/../SAM-6D-official/SAM-6D/Instance_Segmentation_Model}"
DATA_ROOT="${DATA_ROOT:-$DIFF_ROOT/Data}"
NAME_EXP="${NAME_EXP:-train_real_ism}"
PRED_SUBDIR="${PRED_SUBDIR:-result_ycbv_train_real}"
# 完整 train_real_metaData.json（含 rgb_path）可 false；仅 scene_id/frame_id 索引时需 true
RESET_META="${RESET_META:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Hydra 子配置名：configs/machine/trainer/{local_gloo,local}.yaml
TRAINER_PROFILE="${TRAINER_PROFILE:-local_gloo}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
ISM_CONDA_ENV="${ISM_CONDA_ENV:-diffsam}"
USE_CONDA_RUN="${USE_CONDA_RUN:-1}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-}"
# 多 worker 在 WSL/DDP 下曾出现假死；默认 0 更稳。需要加速可 NUM_WORKERS=4 bash ...
NUM_WORKERS="${NUM_WORKERS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export ISM_SKIP_EXISTING_NPZ="${ISM_SKIP_EXISTING_NPZ:-1}"
export ISM_EMPTY_CUDA_EVERY="${ISM_EMPTY_CUDA_EVERY:-100}"

if [[ ! -d "$ISM_ROOT" ]]; then
  echo "找不到 ISM 仓库: $ISM_ROOT  请设置 ISM_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DATA_ROOT/BOP/ycbv/train_real_metaData.json" ]]; then
  echo "缺少 $DATA_ROOT/BOP/ycbv/train_real_metaData.json" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/BOP/ycbv/train_pbr" ]]; then
  echo "缺少 $DATA_ROOT/BOP/ycbv/train_pbr（官方 ISM 用 PBR 场景做 template/reference，需从 BOP 下载 YCB-Video train_pbr）。" >&2
  exit 1
fi

cd "$ISM_ROOT"
echo "ISM_ROOT=$ISM_ROOT"
echo "DATA_ROOT=$DATA_ROOT (user.local_root_dir)"
echo "name_exp=$NAME_EXP  prediction_subdir=$PRED_SUBDIR  reset_metaData=$RESET_META"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  machine/trainer=$TRAINER_PROFILE devices=$TRAINER_DEVICES num_workers=$NUM_WORKERS"
echo "ISM_SKIP_EXISTING_NPZ=$ISM_SKIP_EXISTING_NPZ  ISM_EMPTY_CUDA_EVERY=$ISM_EMPTY_CUDA_EVERY"
if [[ -n "$LIMIT_TEST_BATCHES" ]]; then
  echo "试跑 limit_test_batches=$LIMIT_TEST_BATCHES（全量请 unset LIMIT_TEST_BATCHES）"
else
  echo "全量约 11.3 万帧，耗时很长；试跑可设 LIMIT_TEST_BATCHES=50"
fi

HYDRA_ARGS=(
  dataset_name=ycbv
  user.local_root_dir="$DATA_ROOT"
  data.query_dataloader.split=train_real
  data.query_dataloader.reset_metaData="$RESET_META"
  prediction_subdir="$PRED_SUBDIR"
  name_exp="$NAME_EXP"
  "machine/trainer=$TRAINER_PROFILE"
  "+machine.trainer.devices=$TRAINER_DEVICES"
  machine.num_workers="$NUM_WORKERS"
)
if [[ -n "$LIMIT_TEST_BATCHES" ]]; then
  HYDRA_ARGS+=(machine.trainer.limit_test_batches="$LIMIT_TEST_BATCHES")
fi

if [[ "$USE_CONDA_RUN" == "1" ]]; then
  if ! command -v conda &>/dev/null; then
    echo "未找到 conda，请安装或设 USE_CONDA_RUN=0 并在已装依赖的环境中运行" >&2
    exit 1
  fi
  # 勿在 --no-capture-output 后再写 --，部分 conda 会生成错误的 wrapper（--: command not found）
  exec conda run -n "$ISM_CONDA_ENV" --no-capture-output python run_inference.py \
    "${HYDRA_ARGS[@]}" "$@"
else
  exec python run_inference.py "${HYDRA_ARGS[@]}" "$@"
fi
