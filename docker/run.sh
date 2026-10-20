#!/usr/bin/env bash
set -euo pipefail

IMAGE="${APOLLO_IMAGE:-apollo-npu-fp32:v1}"
CONTAINER_NAME="${APOLLO_CONTAINER_NAME:-apollo_npu_fp32}"
DATA_DIR="${APOLLO_DATA_DIR:-}"
OUTPUT_DIR="${APOLLO_OUTPUT_DIR:-$(pwd)/work_dirs}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

if [ -z "${DATA_DIR}" ] || [ ! -d "${DATA_DIR}" ]; then
    echo "错误：请通过 APOLLO_DATA_DIR 指定数据集目录。" >&2
    echo "示例：export APOLLO_DATA_DIR=/data/datasets/apollo" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ "$#" -eq 0 ]; then
    set -- bash
fi

ENV_ARGS=(
    -e "ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
)

for name in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; do
    if [ -n "${!name:-}" ]; then
        ENV_ARGS+=(-e "${name}=${!name}")
    fi
done

exec docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --ipc=host \
    --network=host \
    --security-opt label=disable \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v "${DATA_DIR}:/opt/apollo/workspace/Apollo-Vision-Net/data:ro" \
    -v "${OUTPUT_DIR}:/opt/apollo/workspace/Apollo-Vision-Net/work_dirs" \
    "${ENV_ARGS[@]}" \
    "${IMAGE}" \
    "$@"
