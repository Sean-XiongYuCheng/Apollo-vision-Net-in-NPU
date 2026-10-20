#!/usr/bin/env bash
set -euo pipefail

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1  # 910b和950不支持，运行前注释
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export HCCL_OP_EXPANSION_MODE="CCU_SCHED" # 仅支持950

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_DIR="${FASTWAM_DIR:-${SCRIPT_DIR}/../FastWAM}"
BASE_TRAIN_SCRIPT="${FASTWAM_DIR}/scripts/train_zero1.sh"

# 默认训练参数，可通过脚本参数覆盖
DEFAULT_NPROC_PER_NODE=8
DEFAULT_TASK_NAME="libero_uncond_2cam224_1e-4"
BATCH_SIZE=16
MAX_STEPS=1000

# 先使用默认值，后面根据参数覆盖
NPROC_PER_NODE="${DEFAULT_NPROC_PER_NODE}"
TASK_NAME="${DEFAULT_TASK_NAME}"

# 标记是否已经显式设置过，兼容位置参数写法
NPROC_SET=0
TASK_SET=0

HYDRA_ARGS=()

# 统一解析参数：同时兼容位置参数和 --key=value / --key value 格式
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc_per_node)
      NPROC_PER_NODE="${2:?Error: --nproc_per_node requires a value}"
      NPROC_SET=1
      shift 2
      ;;

    --nproc_per_node=*)
      NPROC_PER_NODE="${1#*=}"
      NPROC_SET=1
      shift
      ;;

    --nproc-per-node)
      NPROC_PER_NODE="${2:?Error: --nproc-per-node requires a value}"
      NPROC_SET=1
      shift 2
      ;;

    --nproc-per-node=*)
      NPROC_PER_NODE="${1#*=}"
      NPROC_SET=1
      shift
      ;;

    --task)
      TASK_NAME="${2:?Error: --task requires a value}"
      TASK_SET=1
      shift 2
      ;;

    --task=*)
      TASK_NAME="${1#*=}"
      TASK_SET=1
      shift
      ;;

    --batch-size)
      BATCH_SIZE="${2:?Error: --batch-size requires a value}"
      shift 2
      ;;

    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;

    --max-steps)
      MAX_STEPS="${2:?Error: --max-steps requires a value}"
      shift 2
      ;;

    --max-steps=*)
      MAX_STEPS="${1#*=}"
      shift
      ;;

    --)
      shift
      HYDRA_ARGS+=("$@")
      break
      ;;

    *)
      # 兼容旧写法：
      # bash test/train_performance.sh 4 libero_uncond_2cam224_1e-4
      if [[ "${NPROC_SET}" -eq 0 && "$1" =~ ^[0-9]+$ ]]; then
        NPROC_PER_NODE="$1"
        NPROC_SET=1
      elif [[ "${TASK_SET}" -eq 0 && "$1" != *=* && "$1" != --* ]]; then
        TASK_NAME="$1"
        TASK_SET=1
      else
        # 其他全部作为 Hydra 参数透传
        HYDRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

EXTRA_ARGS=(
  "task=${TASK_NAME}"
  "${HYDRA_ARGS[@]}"
  "batch_size=${BATCH_SIZE}"
  "max_steps=${MAX_STEPS}"
)

NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"

if [[ ! -f "${BASE_TRAIN_SCRIPT}" ]]; then
  echo "Error: ${BASE_TRAIN_SCRIPT} does not exist." >&2
  exit 1
fi

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
fi

export RUN_ID

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

LOG_DIR="$(pwd)/logs"
mkdir -p "${LOG_DIR}"

LOG="${LOG_DIR}/${TASK_NAME}_${RUN_ID}.log"

echo "[log] ${LOG}"

CONFIG="${FASTWAM_DIR}/configs/task/${TASK_NAME}.yaml"
CONFIG_BAK="${CONFIG}.bak_${RUN_ID}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Error: ${CONFIG} does not exist." >&2
  exit 1
fi

# 训练开始时间
start_time=$(date +%s)

# 备份 config
cp "${CONFIG}" "${CONFIG_BAK}"

restore_config() {
  if [[ -f "${CONFIG_BAK}" ]]; then
    mv -f "${CONFIG_BAK}" "${CONFIG}"
  fi
}

cleanup() {
  restore_config
}

# 设置信号捕获，如果训练
#   正常退出（EXIT）
#   用户中断（SIGINT）
#   Kill终止请求（SIGTERM）
#   命令执行失败（ERR）
# 可以自动还原对config文件的修改
trap restore_config EXIT SIGINT SIGTERM ERR
# ##################################################################

# 修改 config 中的 batch_size 和 max_steps
sed -i -E "s/^([[:space:]]*batch_size:[[:space:]]*).*/\1${BATCH_SIZE}/" "${CONFIG}"
sed -i -E "s/^([[:space:]]*max_steps:[[:space:]]*).*/\1${MAX_STEPS}/" "${CONFIG}"

set +e

(
  cd "${FASTWAM_DIR}"

  bash "${BASE_TRAIN_SCRIPT}" \
    "${NPROC_PER_NODE}" \
    "${EXTRA_ARGS[@]}"
) 2>&1 \
  | sed '/^\[launch\] nproc_per_node=/d' \
  | tee "${LOG}" \
  | grep --line-buffered -Ei "speed=[0-9.]+[[:space:]]*step/s|samples/s|loss=|error|runtimeerror|traceback|oom|out of memory|killed"

TRAIN_STATUS=${PIPESTATUS[0]}

set -e

# 训练结束时间
end_time=$(date +%s)
e2e_time=$((end_time - start_time))

echo "------------------ Final result ------------------"

# 单迭代训练时长：取最后 8 个 speed 的平均倒数
TrainingTime=$(
  grep -oE "speed=[0-9.]+" "${LOG}" \
    | tail -n 8 \
    | sed "s/speed=//" \
    | awk '{sum += 1 / $1} END {if (NR > 0) printf "%.6f", sum / NR; else print "NA"}' \
    || true
)

if [[ -z "${TrainingTime}" ]]; then
  TrainingTime="NA"
fi

# 吞吐量：取最后 3 个 samples/s 的平均值
ActualSPS=$(
  grep -oE "[0-9.]+[[:space:]]*samples/s" "${LOG}" \
    | tail -n 3 \
    | grep -oE "[0-9.]+" \
    | awk '{sum += $1} END {if (NR > 0) printf "%.6f", sum / NR; else print "NA"}' \
    || true
)

if [[ -z "${ActualSPS}" ]]; then
  ActualSPS="NA"
fi

# loss：取最后 20 个 loss 的平均值
ActualLoss=$(
  grep -oE "(^|[[:space:]])loss=[0-9.eE+-]+" "${LOG}" \
    | awk -F'loss=' '{print $2}' \
    | tail -n 20 \
    | awk '{sum += $1} END {if (NR > 0) printf "%.6f", sum / NR; else print "NA"}' \
    || true
)

if [[ -z "${ActualLoss}" ]]; then
  ActualLoss="NA"
fi

echo "Training Time per step : ${TrainingTime} s"
echo "E2E Training Time : ${e2e_time} s"
echo "Final Performance images/sec : ${ActualSPS}"
echo "Final Train Loss : ${ActualLoss}"

exit "${TRAIN_STATUS}"
