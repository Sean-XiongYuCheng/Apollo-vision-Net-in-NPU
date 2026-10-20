ARG CANN_VERSION=9.1.0
ARG NPU_TYPE=a3
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
# Stage 3: Create torch2.1.0 Environment
# ============================================================================
FROM env-builder AS torch21-env

# Copy requirements file from Driving SDK repository
COPY requirements.txt /tmp/requirements.txt
COPY docker/install_drivingsdk.sh /tmp/install_drivingsdk.sh

# Create PyTorch2.1.0 environment
RUN conda create -n torch2.1 python=3.8 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/torch2.1/pip.conf

# Install dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torch2.1 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir --upgrade pip setuptools==63.0.0 wheel && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir torch-npu==2.1.0.post17 && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/conda/envs/torch2.1/bin/pip install --no-cache-dir mx-driving==26.1.0.post1

RUN printf '%s\n' \
    'source /opt/conda/etc/profile.d/conda.sh' \
    '' \
    'case "$-" in' \
    '    *i*)' \
    '        printf "%s\n" "========================================"' \
    '        printf "%s\n" " Driving SDK Container Environment"' \
    '        printf "%s\n" "========================================"' \
    '        printf "%s\n" " Available conda environments:"' \
    '        conda env list' \
    '        printf "%s\n" " To activate an environment, run:"' \
    '        printf "%s\n" "   conda activate <env_name>"' \
    '        printf "%s\n" "========================================"' \
    '        ;;' \
    'esac' \
    >> ~/.bashrc
