#!/usr/bin/env bash

export WANDB_MODE=offline
export WANDB_OFFLINE=true
export TORCH_HCCL_ZERO_COPY=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

set -euo pipefail

# ==================== 默认参数（按你的实际运行配置） ====================
NUM_GPUS="${NUM_GPUS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-1000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1280}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

# 你实际用的固定路径，设为默认，仍可通过命令行覆盖
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./GR00T-N1.7-3B}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-./Cosmos-Reason2-2B}"
DATASET_PATH="${DATASET_PATH:-./Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
OUTPUT_DIR="${OUTPUT_DIR:-./libero_10_checkpoints}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"

MODALITY_CONFIG_PATH=""
EXPERIMENT_NAME=""
WANDB_PROJECT=""
EXTRA_ARGS=()

# ==================== 用法说明 ====================
usage() {
    cat <<'EOF'

Options:
  --base-model-path <path>      (default: ./GR00T-N1.7-3B)
  --cosmos-model-path <path>    (default: ./Cosmos-Reason2-2B)
  --dataset-path <path>         (default: ./Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/)
  --embodiment-tag <tag>        (default: LIBERO_PANDA)
  --output-dir <path>           (default: ./libero_10_checkpoints)
  --state-dropout-prob <value>  (default: 0.2)
  --num-gpus <n>                (default: 16)
  --max-steps <n>               (default: 20000)
  --modality-config-path <path>
  --experiment-name <name>
  --wandb-project <name>
  --save-only-model
  --resume-from-checkpoint
  -- <extra launch_finetune.py args>...

EOF
}

# ==================== 参数解析 ====================
while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-model-path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --cosmos-model-path)
            COSMOS_MODEL_PATH="$2"
            shift 2
            ;;
        --cosmos_model_path)
            COSMOS_MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --modality-config-path)
            MODALITY_CONFIG_PATH="$2"
            shift 2
            ;;
        --embodiment-tag)
            EMBODIMENT_TAG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --state-dropout-prob)
            STATE_DROPOUT_PROB="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --save-only-model)
            SAVE_ONLY_MODEL=1
            shift
            ;;
        --resume-from-checkpoint)
            RESUME_FROM_CHECKPOINT=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ==================== 参数检查 ====================
for required_var in BASE_MODEL_PATH DATASET_PATH EMBODIMENT_TAG OUTPUT_DIR; do
    if [ -z "${!required_var}" ]; then
        echo "Missing required argument: ${required_var}" >&2
        usage >&2
        exit 1
    fi
done

if [ ! -d "$DATASET_PATH" ]; then
    echo "[Error] dataset path \"$DATASET_PATH\" does not exist." >&2
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "[Error] base model path \"$BASE_MODEL_PATH\" does not exist." >&2
    exit 1
fi

if [ ! -d "$COSMOS_MODEL_PATH" ]; then
    echo "[Error] cosmos model path \"$COSMOS_MODEL_PATH\" does not exist." >&2
    exit 1
fi

# ==================== 环境变量导出 ====================
export GR00T_MODEL_PATH="$COSMOS_MODEL_PATH"

# ==================== 日志设置 ====================
# 处理 OUTPUT_DIR=./ 的情况，避免路径变成 .//logs
OUTPUT_DIR="${OUTPUT_DIR%/}"
LOG_DIR="${OUTPUT_DIR}/$(date +%Y%m%d)_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${NUM_GPUS}p_$(date +%H%M%S).log"

WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG+=(--use_wandb)
fi

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag "$EMBODIMENT_TAG"
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTPUT_DIR"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 5
    --max_steps "$MAX_STEPS"
    --warmup_ratio 0.05
    --weight_decay 1e-5
    --learning_rate 1e-4
    "${WANDB_FLAG[@]}"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --shard_size "$SHARD_SIZE"
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH"
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE"
)

if [ -n "$MODALITY_CONFIG_PATH" ]; then
    LAUNCH_CMD+=(--modality_config_path "$MODALITY_CONFIG_PATH")
fi
if [ -n "$EXPERIMENT_NAME" ]; then
    LAUNCH_CMD+=(--experiment_name "$EXPERIMENT_NAME")
fi
if [ -n "$WANDB_PROJECT" ]; then
    LAUNCH_CMD+=(--wandb_project "$WANDB_PROJECT")
fi
if [ -n "$STATE_DROPOUT_PROB" ]; then
    LAUNCH_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
fi
if [ -n "${SAVE_ONLY_MODEL:-}" ]; then
    LAUNCH_CMD+=(--save_only_model)
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
    LAUNCH_CMD+=(--resume_from_checkpoint)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

# ==================== 运行训练 ====================
set +e
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=$?
set -e

# ==================== 日志后处理 ====================
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成" >&2
    exit 1
fi

# 提取所有 {'loss': x.xxxx} 并计算平均值
MeanLoss=$(grep -o "'loss': [0-9.]*" "${LOG_FILE}" | awk '{sum+=$2; n++} END {if(n>0) printf("%.4f", sum/n); else print "N/A"}')
FinalLoss=$(grep -o "'loss': [0-9.]*" "${LOG_FILE}" | awk 'END {print $2}' || true)

printf "Mean Loss: %s\n" "${MeanLoss:-N/A}"
echo "Final Loss: ${FinalLoss:-N/A}"

# ==================== FPS 性能计算（参考性能测试脚本） ====================
stepstart_time=$(grep " 400/${MAX_STEPS} " "${LOG_FILE}" | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)
stepend_time=$(grep " 900/${MAX_STEPS} " "${LOG_FILE}" | tail -n1 | awk -F '[<[]' '{print $2}' | xargs)

if [ -z "$stepstart_time" ] || [ -z "$stepend_time" ]; then
    echo "Warning: 未找到400/900步时间记录，跳过FPS计算"
else
    convert_time_to_sec() {
        local time_str=$1
        local IFS=':'
        local parts=($time_str)
        local sec=0
        if [ ${#parts[@]} -eq 3 ]; then
            sec=$((10#${parts[0]} * 3600 + 10#${parts[1]} * 60 + 10#${parts[2]}))
        elif [ ${#parts[@]} -eq 2 ]; then
            sec=$((10#${parts[0]} * 60 + 10#${parts[1]}))
        else
            echo "0"
        fi
        echo ${sec}
    }

    start_time=$(convert_time_to_sec "${stepstart_time}")
    end_time=$(convert_time_to_sec "${stepend_time}")
    total_steps=500
    total_time=$((end_time - start_time))

    if [ "$total_time" -gt 0 ]; then
        step_time=$(awk 'BEGIN{printf "%.4f\n", '"${total_time}"'/'"${total_steps}"'}')
        FPS=$(awk 'BEGIN{printf "%.2f\n", '"${GLOBAL_BATCH_SIZE}"'/'"${step_time}"'}')

        echo "Step start time in seconds: ${start_time}"
        echo "Step end time in seconds: ${end_time}"
        echo "Step time: ${step_time}"
        echo "FPS: ${FPS}"
        echo "FPS: $FPS" >> "${LOG_FILE}"
    else
        echo "Warning: 时间差为0或负数，无法计算FPS"
    fi
fi

exit $EXIT_CODE
