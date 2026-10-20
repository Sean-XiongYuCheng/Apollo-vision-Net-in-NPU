#!/bin/bash

Network="DinoV3"

# ========== 默认值设置 ==========
npu_nums=16
batch_size=64
num_step=1250
data_path=""
extra_path=""
output_dir=""

# ========== 解析 key=value 风格参数 ==========
while [ $# -gt 0 ]; do
    case "$1" in
        npu_nums=*)
            npu_nums="${1#*=}"
            shift
            ;;
        batch_size=*)
            batch_size="${1#*=}"
            shift
            ;;
        num_step=*)
            num_step="${1#*=}"
            shift
            ;;
        data_path=*)
            data_path="${1#*=}"
            shift
            ;;
        extra_path=*)
            extra_path="${1#*=}"
            shift
            ;;
        output_dir=*)
            output_dir="${1#*=}"
            shift
            ;;
        *)
            echo "Warning: Unknown parameter '$1', ignoring..."
            shift
            ;;
    esac
done

# ========== 必填参数检查 ==========
if [ -z "$data_path" ] || [ -z "$extra_path" ] || [ -z "$output_dir" ]; then
    echo "Error: Missing required parameters!"
    echo ""
    echo "Usage: bash test/train_performance.sh \\"
    echo "    data_path=/path/to/imagenet \\"
    echo "    extra_path=/path/to/imagenet \\"
    echo "    output_dir=/path/to/output \\"
    echo "    [npu_nums=16] [batch_size=64] [num_step=5000]"
    echo ""
    echo "Example:"
    echo "    bash test/train_performance.sh data_path=/mnt/imagenet extra_path=/mnt/imagenet output_dir=/mnt/output npu_nums=8 batch_size=32 num_step=2500"
    exit 1
fi

# ========== 训练参数计算 ==========
official_epoch_length=1250
epochs=$((num_step / official_epoch_length))

# warmup_epochs 自适应：epochs 够 10 就用 10，不够就降到 1
if [ $epochs -ge 10 ]; then
    warmup_epochs=10
else
    warmup_epochs=1
fi

echo "========================================"
echo "Network:          $Network"
echo "NPU nums:         $npu_nums"
echo "Data path:        $data_path"
echo "Extra path:       $extra_path"
echo "Output dir:       $output_dir"
echo "Batch size/GPU:   $batch_size"
echo "Target steps:     $num_step"
echo "Epochs:           $epochs"
echo "Warmup epochs:    $warmup_epochs"
echo "========================================"

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/
mkdir -p ${output_path}

# ========== 清空旧 checkpoint，确保从 0 开始 ==========
if [ -d "${output_dir}/ckpt" ]; then
    echo "Removing old checkpoints in ${output_dir}/ckpt ..."
    rm -rf ${output_dir}/ckpt
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
torchrun --nproc_per_node=$npu_nums --nnodes=1 --node-rank=0 --master_addr=127.0.0.1 --master_port=29501 \
    dinov3/train/train.py --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
    --output-dir $output_dir \
    train.batch_size_per_gpu=$batch_size \
    train.OFFICIAL_EPOCH_LENGTH=$official_epoch_length \
    optim.epochs=$epochs \
    optim.warmup_epochs=$warmup_epochs \
    train.dataset_path=ImageNet:split=TRAIN:root=$data_path:extra=$extra_path \
    > ${test_path_dir}/output/train_performance_${npu_nums}p_base_fp32.log 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

cd ..

#结果打印，不需要修改
echo "------------------ Final result ------------------"

# ==================== FPS 性能计算（参考性能测试脚本）====================
LOG_FILE="${test_path_dir}/output/train_performance_${npu_nums}p_base_fp32.log"
MAX_STEPS=$((epochs * official_epoch_length))
GLOBAL_BATCH_SIZE=$((npu_nums * batch_size))

# 确定采样步数（默认400~900，若总步数不足则自适应）
step_start_idx=400
step_end_idx=900
if [ "$MAX_STEPS" -lt 900 ]; then
    step_start_idx=$((MAX_STEPS / 5))
    step_end_idx=$((MAX_STEPS * 4 / 5))
    if [ "$step_start_idx" -lt 10 ]; then
        step_start_idx=10
    fi
    if [ "$step_end_idx" -le "$step_start_idx" ]; then
        step_end_idx=$((step_start_idx + 10))
    fi
    if [ "$step_end_idx" -gt "$MAX_STEPS" ]; then
        step_end_idx=$MAX_STEPS
    fi
fi

# 从DINOv3日志中提取对应步数的行首时间戳（第2列，格式 HH:MM:SS）
stepstart_time=$(grep " \[ *${step_start_idx}/${MAX_STEPS}\]" "${LOG_FILE}" | tail -n1 | awk '{print $2}' | xargs)
stepend_time=$(grep " \[ *${step_end_idx}/${MAX_STEPS}\]" "${LOG_FILE}" | tail -n1 | awk '{print $2}' | xargs)

if [ -z "$stepstart_time" ] || [ -z "$stepend_time" ]; then
    echo "Warning: 未找到${step_start_idx}/${step_end_idx}步时间记录，回退到最后5步time均值计算"
    avg_time=$(grep ")  time:" "${LOG_FILE}" | tail -5 | awk -F'time: ' '{print $2}' | awk '{print $1}' | awk '{sum+=$1} END {print sum/NR}')
    if [ -z "$avg_time" ] || [ "$avg_time" = "0" ]; then
        avg_time=1
    fi
    ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '"${GLOBAL_BATCH_SIZE}"'/'"${avg_time}"'}')
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
        fi
        echo ${sec}
    }

    start_sec=$(convert_time_to_sec "${stepstart_time}")
    end_sec=$(convert_time_to_sec "${stepend_time}")

    # 处理跨天情况（如 23:55:00 -> 00:03:00）
    if [ "$end_sec" -lt "$start_sec" ]; then
        end_sec=$((end_sec + 86400))
    fi

    total_steps=$((step_end_idx - step_start_idx))
    total_time=$((end_sec - start_sec))

    if [ "$total_time" -gt 0 ]; then
        step_time=$(awk 'BEGIN{printf "%.4f\n", '"${total_time}"'/'"${total_steps}"'}')
        avg_time=${step_time}  # 统一变量名，方便后续写入结果日志
        ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '"${GLOBAL_BATCH_SIZE}"'/'"${step_time}"'}')

        echo "Step start time in seconds: ${start_sec}"
        echo "Step end time in seconds: ${end_sec}"
        echo "Step time: ${step_time}"
        echo "FPS: ${ActualFPS}"
        echo "FPS: $ActualFPS" >> "${LOG_FILE}"
    else
        echo "Warning: 时间差为0或负数，回退到最后5步time均值计算"
        avg_time=$(grep ")  time:" "${LOG_FILE}" | tail -5 | awk -F'time: ' '{print $2}' | awk '{print $1}' | awk '{sum+=$1} END {print sum/NR}')
        if [ -z "$avg_time" ] || [ "$avg_time" = "0" ]; then
            avg_time=1
        fi
        ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '"${GLOBAL_BATCH_SIZE}"'/'"${avg_time}"'}')
    fi
fi
# ==================== FPS 计算结束 ====================

#loss值，不需要修改
ActualLoss=$(grep -o "total_loss: [0-9.]*" "${LOG_FILE}" | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
WORLD_SIZE=${npu_nums}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${CaseName}.log
echo "TrainingTime = ${avg_time}" >>${test_path_dir}/output/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${CaseName}.log
