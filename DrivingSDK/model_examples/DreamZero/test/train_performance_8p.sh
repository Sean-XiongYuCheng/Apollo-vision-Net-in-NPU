#!/bin/bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${DREAMZERO_ROOT}/checkpoints/$(date +%Y%m%d)_logs"
mkdir -p ${LOG_DIR}
num_npu=8
batch_size=1
max_steps=100
# 解析命令行参数
for para in $*
do
    if [[ $para == --num-npu* ]];then
        num_npu=`echo ${para#*=}`
    elif [[ $para == --batch-size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --max-steps* ]];then
        max_steps=`echo ${para#*=}`
    fi
done
# 路径变量
DROID_DATA_ROOT="${DREAMZERO_ROOT}/data/droid_lerobot"
WAN22_CKPT_DIR="${DREAMZERO_ROOT}/checkpoints/Wan2.2-TI2V-5B"
IMAGE_ENCODER_DIR="${DREAMZERO_ROOT}/checkpoints/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${DREAMZERO_ROOT}/checkpoints/umt5-xxl"
OUTPUT_DIR="${DREAMZERO_ROOT}/checkpoints/dreamzero_droid_wan22_npu_performance_$(date +%Y%m%d_%H%M%S)"
# 路径校验
if [ ! -d "${DROID_DATA_ROOT}" ]; then
    echo "[Error] DROID dataset path \"${DROID_DATA_ROOT}\" does not exist."
    exit 1
fi
if [ ! -d "${WAN22_CKPT_DIR}" ]; then
    echo "[Error] Wan2.2 checkpoint dir \"${WAN22_CKPT_DIR}\" does not exist."
    exit 1
fi
if [ ! -f "${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "[Error] text encoder not found."
    exit 1
fi
if [ ! -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ]; then
    echo "[Error] VAE weights not found."
    exit 1
fi
if [ ! -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "[Error] image encoder not found."
    exit 1
fi
if [ ! -d "${TOKENIZER_DIR}" ]; then
    echo "[Error] tokenizer directory \"${TOKENIZER_DIR}\" does not exist."
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/train_${num_npu}p_performance.log"
cd "${DREAMZERO_ROOT}"
torchrun --nproc_per_node=${num_npu} --master_port=29500 \
    groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative_wan22 \
    wandb_project=dreamzero_wan22_npu \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.warmup_ratio=0.05 \
    training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json \
    training_args.output_dir=${OUTPUT_DIR} \
    per_device_train_batch_size=${batch_size} \
    max_steps=${max_steps} \
    logging_steps=10 \
    save_steps=${max_steps} \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=false \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    save_lora_only=true \
    max_chunk_size=4 \
    save_strategy=no \
    droid_data_root=${DROID_DATA_ROOT} \
    dit_version=${WAN22_CKPT_DIR} \
    text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth \
    tokenizer_path=${TOKENIZER_DIR} \
    2>&1 | tee ${LOG_FILE}
# 检查日志文件是否存在
if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"
    exit 1
fi
# 从 trainer_state.json 解析 training_step_time 计算 FPS
TRAINER_STATE="${OUTPUT_DIR}/trainer_state.json"
if [ ! -f "${TRAINER_STATE}" ]; then
    echo "Trainer State Error: trainer_state.json 未生成 (${TRAINER_STATE})"
    exit 1
fi
python3 -c "
import json
with open('${TRAINER_STATE}') as f:
    state = json.load(f)
log_history = state.get('log_history', [])
step_times = []
for entry in log_history:
    if 'training_step_time' in entry:
        step_times.append(entry['training_step_time'])
if not step_times:
    print('FPS: N/A (no training_step_time found)')
else:
    # 去掉前10步预热，取后50步平均
    warmup = 10
    if len(step_times) > warmup:
        stable = step_times[warmup:]
    else:
        stable = step_times
    if len(stable) > 50:
        stable = stable[-50:]
    avg_step_time = sum(stable) / len(stable)
    fps = ${batch_size} * ${num_npu} / avg_step_time
    print(f'平均单步耗时: {avg_step_time:.4f}s')
    print(f'FPS: {fps:.2f}')
"
