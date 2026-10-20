#!/bin/bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2

LOG_DIR="./libero_10_checkpoints/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}

num_gpus=8
max_steps=20000
global_batch_size=640
num_workers=4
dataset_path=./examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/
base_model_path=./GR00T-N1.6-3B
embodiment_tag=LIBERO_PANDA

# 解析命令行参数
for para in $*
do
    if [[ $para == --num_gpus* ]];then
        num_gpus=`echo ${para#*=}`
    elif [[ $para == --max_steps* ]];then
        max_steps=`echo ${para#*=}`
    elif [[ $para == --global_batch_size* ]];then
        global_batch_size=`echo ${para#*=}`
    elif [[ $para == --dataset_path* ]];then
        dataset_path=`echo ${para#*=}`
    elif [[ $para == --base_model_path* ]];then
        base_model_path=`echo ${para#*=}`
    elif [[ $para == --embodiment_tag* ]];then
        embodiment_tag=`echo ${para#*=}`
    elif [[ $para == --num_workers* ]];then
        num_workers=`echo ${para#*=}`
    fi
done

# 参数检查
if [[ "$dataset_path" == "" ]];then
    echo "[Error] para \"dataset_path\" must be configured."
    exit 1
fi
if [ ! -d "$dataset_path" ]; then
    echo "[Error] dataset path \"$dataset_path\" does not exist."
    exit 1
fi
if [ ! -d "$base_model_path" ]; then
    echo "[Error] base model path \"$base_model_path\" does not exist."
    exit 1
fi

LOG_FILE="${LOG_DIR}/train_${num_gpus}p.log"

torchrun --nproc_per_node=$num_gpus --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path ${base_model_path} \
    --dataset_path ${dataset_path} \
    --embodiment_tag ${embodiment_tag} \
    --num_gpus $num_gpus \
    --output_dir ./libero_10_checkpoints \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps ${max_steps} \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size ${global_batch_size} \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers ${num_workers} \
    --state_dropout_prob 0.8 \
    2>&1 | tee ${LOG_FILE}     # 记录log同时打印至终端


# 检查日志文件是否存在
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi

MeanLoss=$(grep -o "train_loss': [0-9.]*" ${LOG_FILE} | awk 'END {print $NF}')
FinalLoss=$(grep -o "{'loss': [0-9.]*" ${LOG_FILE} | awk 'END {print $NF}')

# 打印Loss
printf "Mean Loss: %.4f\n" "$MeanLoss"
echo "Final Loss: ${FinalLoss}"
