ARG CANN_VERSION=9.1.0
ARG NPU_TYPE=910b
ARG OS_TYPE=openeuler24.03
ARG BASE_PY_VERSION=3.11
ARG PTA_VERSION=2.7.1
ARG PY_VERSION=3.10
ARG CONDA_ENV_NAME=torch2.7.1_py310
ARG TORCH_NPU_POST=6
ARG SETUPTOOLS_VERSION=76.0.0

# ============================================================================
# Stage 1: Base Environment
# ============================================================================
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${CANN_VERSION}-${NPU_TYPE}-${OS_TYPE}-py${BASE_PY_VERSION} AS base

# Install system dependencies (same as 26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10/Dockerfile)
RUN yum install -y \
    gcc gcc-c++ make cmake \
    git wget curl vim \
    protobuf protobuf-devel \
    net-tools && \
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
# Stage 3: Builder - Create conda env with PyTorch + torch-npu + deps
# ============================================================================
FROM env-builder AS builder

ARG PTA_VERSION
ARG PY_VERSION
ARG CONDA_ENV_NAME
ARG TORCH_NPU_POST
ARG SETUPTOOLS_VERSION

# Copy requirements file for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Create conda environment
RUN conda create -n ${CONDA_ENV_NAME} python=${PY_VERSION} -y && \
    conda clean -ay && \
    # Create fixed symlink for VS Code interpreter path
    ln -sf /opt/conda/envs/${CONDA_ENV_NAME} /opt/conda/envs/default

# Configure pip mirror for the env
RUN PIP_CONF_DIR="/opt/conda/envs/${CONDA_ENV_NAME}" && \
    mkdir -p "${PIP_CONF_DIR}" && \
    printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > "${PIP_CONF_DIR}/pip.conf"

# Install PyTorch CPU, torch-npu, and project dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    pip install --no-cache-dir --upgrade pip setuptools==${SETUPTOOLS_VERSION} wheel && \
    pip install --no-cache-dir torch==${PTA_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-npu==${PTA_VERSION}.post${TORCH_NPU_POST} && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 4: Dev - Compile and install mainline DrivingSDK
# ============================================================================
FROM builder AS dev

ARG PTA_VERSION
ARG PY_VERSION
ARG CONDA_ENV_NAME

# Clone mainline DrivingSDK and compile from source
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    cd /tmp && \
    git clone https://gitcode.com/Ascend/DrivingSDK.git && \
    cd DrivingSDK && \
    bash ci/build.sh --python=${PY_VERSION} && \
    pip install --force-reinstall --no-deps dist/*.whl && \
    cd / && \
    rm -rf /tmp/DrivingSDK

# Shell configuration: conda init + auto-activate env + CANN env
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

WORKDIR /workspace/DrivingSDK_op
