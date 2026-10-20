ARG CANN_VERSION=9.1.0-beta.3
ARG NPU_TYPE=910b
ARG OS_TYPE=openeuler24.03
ARG PYTHON_VERSION=3.12

# ============================================================================
# Stage 1: Base Environment
# ============================================================================
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VERSION}-${NPU_TYPE}-${OS_TYPE}-py${PYTHON_VERSION} AS base

# Install system dependencies based on OS type
RUN yum install -y \
    gcc gcc-c++ make cmake \
    git wget curl vim \
    protobuf protobuf-devel \
    net-tools hostname && \
    yum clean all

# ============================================================================
# Stage 2: Install Miniconda
# ============================================================================
FROM base AS env-builder

# Install Miniconda
RUN ARCH=$(uname -m) && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ============================================================================
# Stage 3: Create torch2.1.0 Environments
# ============================================================================
FROM env-builder AS torch21-envs

# Create PyTorch2.1.0 environment
RUN conda create -n torch2.1 python=3.8 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    'timeout = 600' \
    'retries = 10' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/torch2.1/pip.conf

# Install dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torch2.1 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir --upgrade pip setuptools==68.0.0 wheel ninja && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir torch-npu==2.1.0.post17 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir \
        numpy protobuf decorator sympy scipy attrs cloudpickle psutil synr tornado pyyaml \
        expecttest hypothesis packaging pytest prettytable pydantic \
        opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 opencv-contrib-python-headless==4.11.0.86 \
        torch-scatter

# Build and install mmcv-full (NPU version)
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torch2.1 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir addict yapf pyyaml regex ordered-set terminaltables pyquaternion Pillow six && \
    cd /opt && \
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git mmcv && \
    cd mmcv && \
    MMCV_WITH_OPS=1 FORCE_NPU=1 /opt/conda/envs/torch2.1/bin/python setup.py bdist_wheel && \
    /opt/conda/envs/torch2.1/bin/pip install --no-deps --force-reinstall dist/mmcv_full-*.whl && \
    cd / && rm -rf /opt/mmcv

# Build and install cumm (CPU-only version) + spconv
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torch2.1 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir pccm==0.3.4 ccimport==0.3.7 && \
    cd /opt && \
    git clone https://github.com/FindDefinition/cumm.git -b v0.2.9 cumm && \
    cd cumm && \
    sed -i 's|// #include <cstddef>|#include <cstddef>|' include/tensorview/core/defs.h && \
    sed -i 's|// #include <cstdint>|#include <cstdint>|' include/tensorview/core/defs.h && \
    CUMM_CUDA_VERSION="" CUMM_DISABLE_JIT="1" /opt/conda/envs/torch2.1/bin/python setup.py bdist_wheel && \
    /opt/conda/envs/torch2.1/bin/pip install --force-reinstall dist/cumm-*.whl && \
    cd / && rm -rf /opt/cumm && \
    cd /opt && \
    git clone https://github.com/traveller59/spconv.git -b v2.1.25 spconv && \
    cd spconv && \
    sed -i 's/if hasattr(_ext, "cumm"):/if 0:/' spconv/pytorch/ops.py && \
    sed -i '26,30s/^/#DISABLE_GPU_/' spconv/utils/__init__.py && \
    CUMM_CUDA_VERSION="" SPCONV_DISABLE_JIT="1" /opt/conda/envs/torch2.1/bin/python setup.py bdist_wheel && \
    /opt/conda/envs/torch2.1/bin/pip install --no-deps --force-reinstall dist/spconv-*.whl && \
    sed -i 's/if hasattr(_ext, "cumm"):/if False:  # CPU-only build/' \
        /opt/conda/envs/torch2.1/lib/python3.8/site-packages/spconv/cppconstants.py && \
    cd / && rm -rf /opt/spconv

# Configure environment variables
ENV SPCONV_DISABLE_JIT=1
ENV LD_LIBRARY_PATH=/opt/conda/envs/torch2.1/lib/python3.8/site-packages/torch/lib:/opt/conda/envs/torch2.1/lib/python3.8/site-packages/torch_npu/lib:${LD_LIBRARY_PATH}

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate torch2.1" >> ~/.bashrc
