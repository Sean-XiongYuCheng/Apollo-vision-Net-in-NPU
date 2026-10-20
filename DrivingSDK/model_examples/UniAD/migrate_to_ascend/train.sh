#!/usr/bin/env bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="UniAD"
###############参数解析###############
# --stage=stage1|stage2 --performance --num_npu=N --config=PATH
# 多机参数通过环境变量传入：NNODES NODE_RANK MASTER_ADDR MASTER_PORT
Usage() {
    echo "Usage: bash migrate_to_ascend/train.sh [options]"
    echo "Options:"
    echo "  --stage=stage1|stage2   Select stage (default: stage1)"
    echo "  --performance            Performance mode (1 epoch + brake at 1000 iters)"
    echo "  --num_npu=N             NPU processes / cards to use (default: 8)"
    echo "  --config=PATH           Custom config file (default: stage config)"
    echo "  -h|--help               Show this help"
    echo "Multi-node env vars: NNODES NODE_RANK MASTER_ADDR MASTER_PORT"
}

STAGE="stage1"
PERFORMANCE=0
NUM_NPU=8
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage=*) STAGE="${1#*=}" ;;
        --performance) PERFORMANCE=1 ;;
        --num_npu=*) NUM_NPU="${1#*=}" ;;
        --config=*) CONFIG="${1#*=}" ;;
        -h|--help) Usage; exit 0 ;;
        *) echo "[UniAD][ERROR] Unknown option: $1" >&2; Usage >&2; exit 1 ;;
    esac
    shift
done

[[ "${STAGE}" == "stage1" || "${STAGE}" == "stage2" ]] || { echo "[UniAD][ERROR] Unknown stage: ${STAGE}" >&2; exit 1; }
[[ "${NUM_NPU}" =~ ^[0-9]+$ && "${NUM_NPU}" -gt 0 ]] || { echo "[UniAD][ERROR] --num_npu must be a positive integer" >&2; exit 1; }

###############路径处理###############
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
# UniAD 源码树（install_UniAD.sh 克隆），运行期切换到此目录以保证 config 相对路径解析
SOURCE_DIR="${ROOT_DIR}/UniAD"
HARNESS_ROOT="${ROOT_DIR}/test/output"

if [[ -n "${CONFIG}" ]]; then
    CONFIG_FILE="${CONFIG}"
else
    if [[ "${STAGE}" == "stage1" ]]; then
        CONFIG_FILE="${SOURCE_DIR}/projects/configs/stage1_track_map/base_track_map.py"
    else
        CONFIG_FILE="${SOURCE_DIR}/projects/configs/stage2_e2e/base_e2e.py"
    fi
fi
[[ -f "${CONFIG_FILE}" ]] || { echo "[UniAD][ERROR] Config not found: ${CONFIG_FILE}" >&2; exit 1; }

# 多机参数（环境变量，同标杆 test 脚本）
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(( ((RANDOM << 15) | RANDOM) % 49152 + 10000 ))}
(( NNODES > 0 && NODE_RANK >= 0 && NODE_RANK < NNODES )) || { echo "[UniAD][ERROR] Invalid NNODES/NODE_RANK" >&2; exit 1; }

MODE="full"
if [[ "${PERFORMANCE}" -eq 1 ]]; then
    MODE="performance"
fi
OUT_DIR="${HARNESS_ROOT}/${MODE}/${STAGE}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/train_${MODE}.log"
WORK_DIR=$(echo "${CONFIG_FILE%.*}" | sed -e "s|${SOURCE_DIR}/projects/configs|${SOURCE_DIR}/projects/work_dirs|")/
mkdir -p "${WORK_DIR}logs"

# 非平台场景时 source 环境变量
if [[ "${etp_running_flag:-}" != "true" ]]; then
    source "${SCRIPT_DIR}/env_npu.sh"
fi

# work-dir 切换（config 相对路径：data_root/plugin_dir）
cd "${SOURCE_DIR}"

CFG_OPTIONS="data.samples_per_gpu=1"
if [[ "${PERFORMANCE}" -eq 1 ]]; then
    CFG_OPTIONS="${CFG_OPTIONS} total_epochs=1 runner.max_epochs=1"
fi

export UNIAD_BRAKE_AT="${UNIAD_BRAKE_AT:-500}"  # 性能模式提前停止步数（对齐参考工程 tools/perf.py 的 brake_at(500)）

COLLECTIVE_ARGS=""
if [[ "${NNODES}" -gt 1 ]]; then
    COLLECTIVE_ARGS="--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=${NNODES} --node_rank=${NODE_RANK}"
else
    COLLECTIVE_ARGS="--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
fi

echo "[UniAD] Stage = ${STAGE}, Mode = ${MODE}, NumNPU = ${NUM_NPU}"
echo "[UniAD] Config = ${CONFIG_FILE}"
echo "[UniAD] Log = ${LOG_FILE}"

PERF_FLAG=""
if [[ "${PERFORMANCE}" -eq 1 ]]; then
    PERF_FLAG="--performance"
fi

start_time=$(date +%s)
echo "[UniAD] Training..."
set +e
PYTHONPATH="${ROOT_DIR}:${SOURCE_DIR}:${PYTHONPATH:-}" \
"${PYTHON_BIN:-python}" -m torch.distributed.run \
    --nproc_per_node="${NUM_NPU}" \
    ${COLLECTIVE_ARGS} \
    "${SCRIPT_DIR}/train.py" \
    ${PERF_FLAG} \
    "${CONFIG_FILE}" \
    --launcher pytorch \
    --deterministic \
    --cfg-options ${CFG_OPTIONS} \
    --work-dir "${WORK_DIR}" \
    > "${LOG_FILE}" 2>&1
train_status=$?
set -e

end_time=$(date +%s)
e2e_time=$(( end_time - start_time ))

###############结果统计（对齐标杆：NR>10 的 time 平均）###############
BatchSize=1
DeviceType=$(uname -m)
CaseName="${Network}_bs${BatchSize}_${NUM_NPU}p"
if [[ "${PERFORMANCE}" -eq 1 ]]; then
    CaseName="${CaseName}_perf"
fi
GLOBAL_BATCH_SIZE=$(( NUM_NPU * BatchSize ))

# 单步平均耗时（s/iter）
Iteration_time=$(grep -a 'mmdet - INFO - Epoch ' "${LOG_FILE}" | awk -F "time: " '{print $2}' | awk -F ", " '{print $1}' | awk 'NR>10 {sum+=$1; count++} END {if (count != 0) printf("%.3f", sum/count)}')
if [[ -z "${Iteration_time}" ]]; then
    Iteration_time="0.000"
fi
# 训练总耗时（s）：全部 iter 的 time 累加（对齐标杆 TrainingTime 语义，
# mmdet 日志字段为小写 "time: "）
TrainingTime=$(grep -a 'time: ' "${LOG_FILE}" | awk -F "time: " '{print $2}' | awk -F ", " '{print $1}' | awk '{sum+=$1} END {if (sum != 0) printf("%.3f", sum)}')
if [[ -z "${TrainingTime}" ]]; then
    TrainingTime="0.000"
fi
if awk "BEGIN {exit !(${Iteration_time} > 0)}"; then
    FPS=$(awk "BEGIN {printf \"%.3f\", ${GLOBAL_BATCH_SIZE} / ${Iteration_time}}")
else
    FPS="0.000"
fi

echo "------------------ Final result ------------------"
echo "Iteration time (s) : ${Iteration_time}"
echo "FPS : ${FPS}"
echo "E2E Training Duration sec : ${e2e_time}"

# 精度指标（full 模式，对齐标杆脚本：从 Epoch(val) 行提取 amota/lanes_iou/amotp）
amota=""
lanes_iou=""
amotp=""
L2_eval=""
if [[ "${PERFORMANCE}" -eq 0 ]]; then
    amota=$(grep -a "mmdet - INFO - Epoch(val)" "${LOG_FILE}" | tail -1 | awk -F "amota: " '{print $2}' | awk -F ", " '{print $1}' | awk -F ", " '{print $1}')
    lanes_iou=$(grep -a "mmdet - INFO - Epoch(val)" "${LOG_FILE}" | tail -1 | awk -F "lanes_iou: " '{print $2}' | awk -F ", " '{print $1}' | awk -F ", " '{print $1}')
    amotp=$(grep -a "mmdet - INFO - Epoch(val)" "${LOG_FILE}" | tail -1 | awk -F "amotp: " '{print $2}' | awk -F ", " '{print $1}' | awk -F ", " '{print $1}')
    if [[ "${STAGE}" == "stage1" ]]; then
        echo "amota : ${amota}"
        echo "lanes_iou : ${lanes_iou}"
    else
        echo "amota : ${amota}"
        echo "lanes_iou : ${lanes_iou}"
        echo "amotp : ${amotp}"
        L2_eval=$(grep -a "|      L2     |" "${LOG_FILE}" | tail -1 | awk -F "[| ]+" '{sum=0; count=0; for(i=2; i<=NF; i++) {if($i ~ /^[0-9]+(\.[0-9]+)?$/) {sum+=$i; count++}}; if(count>0) printf("%.4f", sum/count)}')
        echo "L2 : ${L2_eval}"
    fi
fi

# 关键信息打印（标杆布局）
echo "Network = ${Network}" > "${OUT_DIR}/${CaseName}.log"
echo "RankSize = ${NUM_NPU}" >> "${OUT_DIR}/${CaseName}.log"
echo "BatchSize = ${BatchSize}" >> "${OUT_DIR}/${CaseName}.log"
echo "DeviceType = ${DeviceType}" >> "${OUT_DIR}/${CaseName}.log"
echo "CaseName = ${CaseName}" >> "${OUT_DIR}/${CaseName}.log"
# 精度写入 CaseName.log（对齐标杆：stage1 写 AMOTA/IoU-lane；stage2 标杆为注释状态，不写）
if [[ "${PERFORMANCE}" -eq 0 && "${STAGE}" == "stage1" ]]; then
    echo "AMOTA = ${amota}" >> "${OUT_DIR}/${CaseName}.log"
    echo "IoU-lane = ${lanes_iou}" >> "${OUT_DIR}/${CaseName}.log"
fi
echo "Iteration time = ${Iteration_time}" > "${OUT_DIR}/${CaseName}_perf_report.log"
echo "FPS = ${FPS}" >> "${OUT_DIR}/${CaseName}_perf_report.log"
echo "TrainingTime = ${TrainingTime}" >> "${OUT_DIR}/${CaseName}_perf_report.log"
echo "E2ETrainingTime = ${e2e_time}" >> "${OUT_DIR}/${CaseName}_perf_report.log"

if [[ "${train_status}" -ne 0 ]]; then
    echo "[UniAD][ERROR] Training failed (status ${train_status}), check ${LOG_FILE}" >&2
    exit "${train_status}"
fi
