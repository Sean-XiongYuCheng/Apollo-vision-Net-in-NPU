#!/bin/bash
# 用于 mxDriving 门禁编译任务

set -e
WORKSPACE=$1
pr_id=$2

today=$(date -d "today" +%Y%m%d)
ARCH=$(uname -m)

export ASCEND_HOME=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/set_env.sh

echo "Start setting up environment variables"
export ASCEND_HOME_PATH=$ASCEND_HOME
export ASCEND_DIR=$ASCEND_HOME
export ASCEND_HOME_DIR=$ASCEND_HOME
export ASCEND_INSTALL_PATH=$ASCEND_HOME
export ASCEND_OPP_PATH=$ASCEND_HOME/opp
export DDK_PATH=$ASCEND_HOME
export NPU_HOST_LIB=$ASCEND_HOME/runtime/lib64/stub
export ASCEND_OPP_KERNEL_PATH=$ASCEND_HOME
export ASCEND_AICPU_PATH=$ASCEND_HOME
export TOOLCHAIN_HOME=$ASCEND_HOME/toolkit
export PATH=$ASCEND_HOME/bin:$ASCEND_HOME/compiler/ccec_compiler/bin:$ASCEND_HOME/toolkit/tools/profiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_HOME/lib64:$ASCEND_HOME/aarch64-linux/devlib:$ASCEND_HOME/x86_64-linux/devlib:$ASCEND_HOME/aarch64-linux/lib64:$ASCEND_HOME/x86_64-linux/lib64:$ASCEND_HOME/lib64/plugin/opskernel:$ASCEND_HOME/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/python/site-packages:$ASCEND_HOME/toolkit/python/site-packages:$ASCEND_HOME/pyACL/python/site-packages:$ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
echo "End setting up environment variables"

function pre_compile() {
    if [ ${ARCH} = "x86_64" ]; then
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.1.0/torch-2.1.0%2Bcpu-cp38-cp38-linux_x86_64.whl
        pip3.8 install torch-2.1.0+cpu-cp38-cp38-linux_x86_64.whl --force-reinstall
        rm -rf torch-2.1.0+cpu-cp38-cp38-linux_x86_64.whl

        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/cache/v2.1.0/pytorchv2.1.0_3.8_x86_64.tar.gz
        tar -zxvf pytorchv2.1.0_3.8_x86_64.tar.gz
        pip3.8 install torch_npu*x86_64.whl --force-reinstall --no-deps
        rm -rf pytorchv2.1.0_3.8_x86_64.tar.gz
        rm -rf *.whl

    else
        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/torch/v2.1.0/torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
        pip3.8 install torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --force-reinstall
        rm -rf torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

        wget --no-check-certificate --no-verbose -P ${WORKSPACE} https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/cache/v2.1.0/pytorchv2.1.0_3.8_aarch64.tar.gz
        tar -zxvf pytorchv2.1.0_3.8_aarch64.tar.gz
        pip3.8 install torch_npu*aarch64.whl --force-reinstall --no-deps
        rm -rf pytorchv2.1.0_3.8_aarch64.tar.gz
        rm -rf *.whl
    fi

}

function compile() {
    mkdir -p "${WORKSPACE}"/output/"${pr_id}"
    cd "${WORKSPACE}"/CODE
    export BUILD_WITHOUT_SHA=True
    sh ci/build.sh --python=3.8
    cp -rf dist/*.whl "${WORKSPACE}"

}
cd "${WORKSPACE}"
pre_compile
compile
