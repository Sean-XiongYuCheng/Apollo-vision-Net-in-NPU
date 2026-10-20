# ==================== 参数默认值 ====================
NODE_COUNT=8
VAE_PATH="./Wan2.2_VAE.pth"
DATASET_PATH="./BridgeData2-Subset-Synthetic-Captions"
PRETRAINED_MODEL_PATH="./Qwen3-VL-8B-Instruct"
RECOMPUTE="on"                # recompute switch: on=enable(full) off=disable(none)
MAX_ITER=""                   # optional: override trainer.max_iter + scheduler.cycle_lengths

# ---- seed_all 相关参数 ----
# 默认启用 seed_all，但 mode=false（只固定种子，不强制算子级确定性，兼顾可复现与性能）
SEED_ALL_ENABLED=true    # 是否启用 seed_all
SEED_ALL_SEED=1234       # 随机种子
SEED_ALL_MODE=false      # 是否开启算子级确定性计算

# ==================== 命令行参数解析 ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)
            NODE_COUNT="$2"
            shift 2
            ;;
        --vae_path)
            VAE_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --pretrained_model_path)
            PRETRAINED_MODEL_PATH="$2"
            shift 2
            ;;
        --seed_all)
            # 可选值：如果下一个参数存在且不以 - 开头，则视为值；否则裸写默认为 true
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                if [[ "$2" == "true" || "$2" == "false" ]]; then
                    SEED_ALL_ENABLED="$2"
                    shift 2
                else
                    echo "Error: --seed_all expects 'true' or 'false', got: $2"
                    exit 1
                fi
            else
                SEED_ALL_ENABLED=true
                shift
            fi
            ;;
        --seed_all_seed)
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                SEED_ALL_SEED="$2"
                shift 2
            else
                echo "Error: --seed_all_seed requires a numeric value"
                exit 1
            fi
            ;;
        --seed_all_mode)
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                if [[ "$2" == "true" || "$2" == "false" ]]; then
                    SEED_ALL_MODE="$2"
                    shift 2
                else
                    echo "Error: --seed_all_mode expects 'true' or 'false', got: $2"
                    exit 1
                fi
            else
                echo "Error: --seed_all_mode requires a value (true or false)"
                exit 1
            fi
            ;;
        --max_iter)
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                MAX_ITER="$2"
                shift 2
            else
                echo "Error: --max_iter requires a numeric value"
                exit 1
            fi
            ;;
        --recompute)
            # 如果下一个参数存在且不以 - 开头，则视为值；否则裸写默认为 on
            if [[ -n "${2:-}" ]] && [[ "$2" != -* ]]; then
                RECOMPUTE="$2"
                shift 2
            else
                RECOMPUTE="on"
                shift
            fi
            ;;
        --help|-h)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --node N                  Number of NPU devices to use (default: 8)"
            echo "  --vae_path PATH           Path to Wan2.2 VAE checkpoint"
            echo "  --dataset_path PATH       Path to the dataset directory"
            echo "  --pretrained_model_path PATH  Path to the pretrained/base model checkpoint"
            echo ""
            echo "  --seed_all [true|false]   Enable msprobe seed_all (default: true)"
            echo "                            If used without value, defaults to true"
            echo "  --seed_all_seed N         Random seed for seed_all (default: 1234)"
            echo "  --seed_all_mode [true|false]  Enable deterministic mode (default: false)"
            echo "                            true = strict reproducibility, may hurt performance"
            echo "                            false = fixed seeds only, better performance"
            echo "  --recompute on|off       Enable/disable gradient checkpointing (default: on)"
            echo "  --max_iter N             Override training iterations (syncs scheduler.cycle_lengths)"
            echo "                            If used without value, defaults to on"
            echo ""
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ==================== 参数校验 ====================
if ! [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] || [[ "$NODE_COUNT" -lt 1 ]]; then
    echo "Error: --node must be a positive integer, got: $NODE_COUNT"
    exit 1
fi

if ! [[ "$SEED_ALL_SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: --seed_all_seed must be a positive integer, got: $SEED_ALL_SEED"
    exit 1
fi

if [[ "$RECOMPUTE" != "on" && "$RECOMPUTE" != "off" ]]; then
    echo "Error: --recompute must be 'on' or 'off', got: $RECOMPUTE"
    exit 1
fi

if [[ -n "$MAX_ITER" ]] && ! [[ "$MAX_ITER" =~ ^[0-9]+$ ]]; then
    echo "Error: --max_iter must be a positive integer, got: $MAX_ITER"
    exit 1
fi

# Map to the config value consumed by model.config.activation_checkpointing.mode
if [[ "$RECOMPUTE" == "on" ]]; then
    RECOMPUTE_MODE="full"
else
    RECOMPUTE_MODE="none"
fi

if [[ ! -f "$VAE_PATH" ]]; then
    echo "Warning: VAE file not found at '$VAE_PATH'"
fi

if [[ ! -d "$DATASET_PATH" ]]; then
    echo "Warning: Dataset directory not found at '$DATASET_PATH'"
fi

if [[ ! -d "$PRETRAINED_MODEL_PATH" ]]; then
    echo "Warning: Pretrained model directory not found at '$PRETRAINED_MODEL_PATH'"
fi

echo "========================================"
echo "  Launch Configuration"
echo "========================================"
echo "  Node count (NPU devices) : $NODE_COUNT"
echo "  VAE path                 : $VAE_PATH"
echo "  Dataset path             : $DATASET_PATH"
echo "  Pretrained model path    : $PRETRAINED_MODEL_PATH"
echo ""
echo "  Seed all enabled         : $SEED_ALL_ENABLED"
echo "  Seed all seed            : $SEED_ALL_SEED"
echo "  Seed all mode            : $SEED_ALL_MODE"
echo "  Recompute (grad ckpt)    : $RECOMPUTE"
echo "  Max iterations          : ${MAX_ITER:-<toml default>}"
echo "========================================"

# ==================== 环境变量设置 ====================
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export COSMOS_DEVICE="npu"
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1

export PYTHONWARNINGS="ignore::UserWarning:triton.language.semantic"
unset http_proxy https_proxy

# ==================== 导出用户指定的路径为环境变量 ====================
export DATASET_DIR="$DATASET_PATH"
export PRETRAINED_MODEL_PATH="$PRETRAINED_MODEL_PATH"
export WAN_VAE_PATH="$VAE_PATH"

# ---- 导出 seed_all 相关配置 ----
export SEED_ALL_ENABLED="$SEED_ALL_ENABLED"
export SEED_ALL_SEED="$SEED_ALL_SEED"
export SEED_ALL_MODE="$SEED_ALL_MODE"

# 训练配置路径
CHECKPOINT_DIR="$PRETRAINED_MODEL_PATH"

# ==================== 数据集下载（如不存在） ====================
if [[ ! -f "$DATASET_DIR/sft_dataset_bridge/train/video_dataset_file.jsonl" ]]; then
    echo "Dataset not found, downloading..."
    uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
        --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 --local-dir "$DATASET_DIR"
fi

# ==================== VAE 下载（如不存在） ====================
if [[ ! -f "$VAE_PATH" ]]; then
    echo "VAE not found, downloading..."
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$VAE_PATH")"
fi

# ==================== 启动训练 ====================
export DATASET_PATH="$DATASET_DIR/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="$CHECKPOINT_DIR"

# 日志落盘路径（带时间戳，避免多次运行覆盖）
# Optional iteration overrides (empty when not set)
ITER_OVERRIDES=""
if [[ -n "$MAX_ITER" ]]; then
    ITER_OVERRIDES="trainer.max_iter=$MAX_ITER scheduler.cycle_lengths=[$MAX_ITER]"
fi

LOG_FILE="$PWD/outputs/train/train_log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting training with $NODE_COUNT NPU devices..."
echo "Realtime log will be saved to: $LOG_FILE"

# 临时关闭 set -e，以便捕获 torchrun 退出码并继续执行后续统计
set +e
IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train" torchrun --master_port=29690 --nproc_per_node="$NODE_COUNT" \
    -m cosmos_framework.scripts.train --sft-toml="examples/toml/sft_config/vision_sft_nano.toml" \
    -- model.config.activation_checkpointing.mode="$RECOMPUTE_MODE" $ITER_OVERRIDES 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "========================================"
echo "  Training Summary"
echo "========================================"

# ---- 提取 RANK 0 的 iter_speed 日志行 ----
RANK0_LINES=$(grep '\[RANK 0\].*Iteration' "$LOG_FILE" || true)

# ---- Final Loss: RANK 0 最后一行的 Loss ----
FINAL_LOSS=$(echo "$RANK0_LINES" | tail -n 1 | sed -n 's/.*Loss: \([0-9.]*\).*/\1/p')
if [[ -n "$FINAL_LOSS" ]]; then
    printf "  Final loss (RANK 0)      : %s\n" "$FINAL_LOSS"
else
    echo "  Final loss (RANK 0)      : N/A"
fi

# ---- Iteration Time: RANK 0 的 iter 30-50 平均耗时 ----
ITER_TIME=$(echo "$RANK0_LINES" | awk '
{
    for (i = 1; i <= NF; i++) {
        if ($i == "Iteration") {
            iter = $(i + 1)
            gsub(/:/, "", iter)
            if (iter >= 30 && iter <= 50) {
                for (j = i; j <= NF; j++) {
                    if ($j == "Time:") {
                        time = $(j + 1)
                        gsub(/s/, "", time)
                        sum += time
                        count++
                    }
                }
            }
        }
    }
}
END {
    if (count > 0) {
        printf "%.2fs (avg over %d iterations, iter 30-50)\n", sum / count, count
    }
}
')
if [[ -n "$ITER_TIME" ]]; then
    printf "  Iteration time (avg)     : %s\n" "$ITER_TIME"
else
    echo "  Iteration time (avg)     : N/A (no iter 30-50 data found)"
fi

# ---- TPS (Tokens Per Second) 计算 ----
# VFM 场景: grad_accum_iter=2, max_num_tokens_after_packing=45056
# Tokens per Step = grad_accum_iter * max_num_tokens_after_packing * NODE_COUNT
VFM_TOKENS_PER_STEP=$((NODE_COUNT * 90112))  # 2 * 45056 * NODE_COUNT

# 从 ITER_TIME 中提取纯数字时间值（格式如 "0.85s (avg over 10 iterations, iter 30-50)"）
ITER_TIME_VAL=$(echo "$ITER_TIME" | awk '{print $1}' | sed 's/s//')

if [[ -n "$ITER_TIME_VAL" && "$ITER_TIME_VAL" != "N/A" ]] && echo "$ITER_TIME_VAL" | grep -qE '^[0-9]+\.?[0-9]*$'; then
    VFM_TPS=$(awk -v tokens="$VFM_TOKENS_PER_STEP" -v time="$ITER_TIME_VAL" 'BEGIN {
        tps = tokens / time
        if (tps >= 1000000) {
            printf "%.2fM tokens/s (raw: %.0f)", tps / 1000000, tps
        } else if (tps >= 1000) {
            printf "%.2fK tokens/s (raw: %.0f)", tps / 1000, tps
        } else {
            printf "%.0f tokens/s", tps
        }
    }')
    printf "  TPS (Tokens/Second)      : %s\n" "$VFM_TPS"
    printf "  Tokens per Step          : %d\n" "$VFM_TOKENS_PER_STEP"
else
    echo "  TPS (Tokens/Second)      : N/A (no iter time data)"
fi


echo "========================================"

# 以 torchrun 的实际退出码退出，保持脚本错误语义
exit $TRAIN_EXIT_CODE
