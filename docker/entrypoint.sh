#!/usr/bin/env bash
set -e

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PATH="/opt/apollo/env/bin:${PATH}"
export PYTHONPATH="/opt/apollo/workspace/mmcv:/opt/apollo/workspace/mmdetection_2.19:/opt/apollo/workspace/mmsegmentation_0.20:/opt/apollo/workspace/mmdetection3d_0.18.1:/opt/apollo/workspace/Apollo-Vision-Net:${PYTHONPATH}"

exec "$@"
