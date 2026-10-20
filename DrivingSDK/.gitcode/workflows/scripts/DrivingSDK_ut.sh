#!/bin/bash
# 用于mxDriving门禁UT 脚本
set -e
WORKSPACE=$1
pr_id=$2
today=$(date -d "today" +%Y%m%d)
# 进入工作目录
cd ${WORKSPACE}
ldconfig
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/python3.8/lib:/usr/local/python3.8/lib/site-packages/mx_driving/packages/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
export GPU_OUT_PATH=/home/smoke/gpu_compare_data/


wget --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.1.0/torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
/opt/conda/envs/torch2.1/bin/pip install torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall --no-deps
rm -rf torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

wget --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/cache/v2.1.0/pytorchv2.1.0_3.8_aarch64.tar.gz
tar -zxvf pytorchv2.1.0_3.8_aarch64.tar.gz
/opt/conda/envs/torch2.1/bin/pip install torch_npu*aarch64.whl --force-reinstall --no-deps
/opt/conda/envs/torch2.1/bin/pip install mx_driving*_aarch64.whl --force-reinstall --no-deps
rm -rf *.tar.gz
rm -rf *.whl

cp -rf modify_files.txt ${WORKSPACE}/CODE
cd ${WORKSPACE}/CODE
/opt/conda/envs/torch2.1/bin/pip list
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.8/lib/python3.8/site-packages/mx_driving/packages/vendors/customize/
mkdir -p /home/smoke/ut_data_cache
export MXDRIVING_CACHE_PATH=/home/smoke/ut_data_cache

cd ${WORKSPACE}/CODE/ci
/opt/conda/envs/torch2.1/bin/python access_control_test.py
