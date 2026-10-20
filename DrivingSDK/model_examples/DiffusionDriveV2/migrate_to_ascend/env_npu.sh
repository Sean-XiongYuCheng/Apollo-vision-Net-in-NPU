#!/bin/bash

# 数据和模型路径。
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-""}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"./dataset/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"./exp"}
# 官方约定: NAVSIM_DEVKIT_ROOT 为包含 navsim 包的仓库根 (官方脚本统一用
# $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_*.py), 默认取本脚本所在仓库根。
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"./dataset"}
export TORCH_NPU_DEVICE_CAPABILITY=${TORCH_NPU_DEVICE_CAPABILITY:-8.0}

IS_950=0
if command -v python >/dev/null 2>&1; then
    _ARCH=$("${PYTHON:-python}" -c '
from mx_driving.get_chip_info import detect_use_arch35
print("950" if detect_use_arch35() else "OTHER")
' 2>/dev/null)
    [ "$_ARCH" = "950" ] && IS_950=1
fi
unset _ARCH

# 使能环境变量
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}                 # 0-关/1-粗粒度绑核/2-细粒度
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}                 # 0-关/1-开/2-流水优化
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ACL_OP_INIT_MODE=${ACL_OP_INIT_MODE:-0}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-""}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}        # 建链等待, 单位 s

# PDMS CPU worker
export PDM_CACHE_SIZE=${PDM_CACHE_SIZE:-16}      # worker 内 metric_cache LRU 条数
export PDM_PDM_CHUNKS=${PDM_PDM_CHUNKS:-2}       # 每条样本 G 条轨迹切块数 (削长尾)
if [ -z "${PDM_WORKERS:-}" ]; then
    if [ "${IS_950}" = "1" ]; then
        export PDM_WORKERS=24
    else
        export PDM_WORKERS=12
    fi
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
