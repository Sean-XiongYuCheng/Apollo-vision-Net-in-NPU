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
# Stage 3: Create torch2.7.1 Environment
# ============================================================================
FROM env-builder AS torch271-env

# Copy requirements file from Driving SDK repository
COPY requirements.txt /tmp/requirements.txt
COPY docker/install_drivingsdk.sh /tmp/install_drivingsdk.sh

# Create PyTorch2.7.1 environment
RUN conda create -n torch2.7.1 python=3.10 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/torch2.7.1/pip.conf

# Install dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate torch2.7.1 && \
    /opt/conda/envs/torch2.7.1/bin/pip install --no-cache-dir --upgrade pip setuptools==76.0.0 wheel && \
    /opt/conda/envs/torch2.7.1/bin/pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/torch2.7.1/bin/pip install --no-cache-dir torch-npu==2.7.1.post6 && \
    /opt/conda/envs/torch2.7.1/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/conda/envs/torch2.7.1/bin/pip install --no-cache-dir mx-driving==26.1.0.post2

# ============================================================================
# Stage 4: Create BEVFormer Environment
# ============================================================================
FROM torch271-env AS bevformer-env

# Copy requirements file from Driving SDK repository
COPY docker/install_bevformer.sh /tmp/install_bevformer.sh
COPY model_examples/BEVFormer /tmp/BEVFormer

# Create bevformer environment
RUN conda create -n bevformer python=3.10 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/bevformer/pip.conf

# Install BEVFormer dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate bevformer && \
    /opt/conda/envs/bevformer/bin/pip install --no-cache-dir --upgrade pip setuptools==76.0.0 wheel && \
    /opt/conda/envs/bevformer/bin/pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/bevformer/bin/pip install --no-cache-dir torch-npu==2.7.1.post6 && \
    /opt/conda/envs/bevformer/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/conda/envs/bevformer/bin/pip install --no-cache-dir mx-driving==26.1.0.post2

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate bevformer && \
    bash /tmp/install_bevformer.sh

# ============================================================================
# Stage 5: Create BEVFusion Environment
# ============================================================================
FROM bevformer-env AS bevfusion-env

# Copy requirements file from Driving SDK repository
COPY docker/install_bevfusion.sh /tmp/install_bevfusion.sh
COPY model_examples/BEVFusion /tmp/BEVFusion

# Create bevfusion environment
RUN conda create -n bevfusion python=3.10 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/bevfusion/pip.conf

# Install BEVFusion dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate bevfusion && \
    /opt/conda/envs/bevfusion/bin/pip install --no-cache-dir --upgrade pip setuptools==76.0.0 wheel && \
    /opt/conda/envs/bevfusion/bin/pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/bevfusion/bin/pip install --no-cache-dir torch-npu==2.7.1.post6 && \
    /opt/conda/envs/bevfusion/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/conda/envs/bevfusion/bin/pip install --no-cache-dir mx-driving==26.1.0.post2

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate bevfusion && \
    bash /tmp/install_bevfusion.sh

# ============================================================================
# Stage 6: Create Sparse4D Environment
# ============================================================================
FROM bevfusion-env AS sparse4d-env

# Copy requirements file from Driving SDK repository
COPY docker/install_sparse4d.sh /tmp/install_sparse4d.sh
COPY model_examples/Sparse4D /tmp/Sparse4D

# Create sparse4d environment
RUN conda create -n sparse4d python=3.10 -y && \
    conda clean -ay

RUN printf '%s\n' \
    '[global]' \
    'index-url = https://mirrors.aliyun.com/pypi/simple/' \
    '' \
    '[install]' \
    'trusted-host = mirrors.aliyun.com' \
    > /opt/conda/envs/sparse4d/pip.conf

# Install Sparse4D dependencies
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sparse4d && \
    /opt/conda/envs/sparse4d/bin/pip install --no-cache-dir --upgrade pip setuptools==76.0.0 wheel && \
    /opt/conda/envs/sparse4d/bin/pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/sparse4d/bin/pip install --no-cache-dir torch-npu==2.7.1.post6 && \
    /opt/conda/envs/sparse4d/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/conda/envs/sparse4d/bin/pip install --no-cache-dir mx-driving==26.1.0.post2

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sparse4d && \
    bash /tmp/install_sparse4d.sh

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
