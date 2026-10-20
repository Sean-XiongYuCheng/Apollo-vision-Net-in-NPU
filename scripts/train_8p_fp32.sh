#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

APOLLO_ROOT="${WORKSPACE_ROOT}/Apollo-Vision-Net"
CONFIG="${CONFIG:-projects/configs/bevformer/bev_tiny_det_occ_apollo.py}"
RUN_DIR="${RUN_DIR:-work_dirs/apollo_dla34_fp32_8p}"
PORT="${PORT:-29511}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${APOLLO_ROOT}"
mkdir -p "${RUN_DIR}"

echo "===== Apollo FP32训练参数 ====="
echo "配置文件: ${CONFIG}"
echo "训练输出: ${RUN_DIR}"
echo "可见NPU: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "训练进程数: ${NPROC_PER_NODE}"
echo "单卡batch: ${SAMPLES_PER_GPU}"
echo "单卡DataLoader worker: ${WORKERS_PER_GPU}"
echo "通信端口: ${PORT}"
echo "OMP线程数: ${OMP_NUM_THREADS}"

PORT="${PORT}" bash tools/dist_train.sh \
  "${CONFIG}" \
  "${NPROC_PER_NODE}" \
  --work-dir "${RUN_DIR}" \
  --seed 0 \
  --no-validate \
  --cfg-options \
    data.samples_per_gpu="${SAMPLES_PER_GPU}" \
    data.workers_per_gpu="${WORKERS_PER_GPU}" \
  2>&1 | tee "${RUN_DIR}/train.log"
