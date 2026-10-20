#!/usr/bin/env bash
set -uo pipefail

WORKSPACE_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

APOLLO_ROOT="${WORKSPACE_ROOT}/Apollo-Vision-Net"
CONFIG="${CONFIG:-projects/configs/bevformer/bev_tiny_det_occ_apollo.py}"
RUN_DIR="${RUN_DIR:-work_dirs/smoke_fp32}"
PORT="${PORT:-29511}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${APOLLO_ROOT}"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

set +e
PORT="${PORT}" bash tools/dist_train.sh \
  "${CONFIG}" \
  1 \
  --work-dir "${RUN_DIR}" \
  --seed 0 \
  --no-validate \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    runner.stop_iters=1 \
    data.samples_per_gpu=1 \
    data.workers_per_gpu="${WORKERS_PER_GPU}" \
    checkpoint_config.interval=100 \
    log_config.interval=1 \
  2>&1 | tee "${LOG_FILE}"

TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if grep -qE 'Epoch \[1\]\[1/[0-9]+\]' "${LOG_FILE}" &&
   grep -q 'STOP!!!' "${LOG_FILE}"; then
  echo "APOLLO_FP32_SMOKE_TRAIN=PASS"
  exit 0
fi

echo "APOLLO_FP32_SMOKE_TRAIN=FAIL"
echo "原始训练进程退出码: ${TRAIN_STATUS}"

if [ "${TRAIN_STATUS}" -ne 0 ]; then
  exit "${TRAIN_STATUS}"
fi

exit 1
