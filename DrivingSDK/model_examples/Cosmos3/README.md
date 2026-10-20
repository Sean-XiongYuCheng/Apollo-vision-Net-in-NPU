# Cosmos3 for PyTorch

# 目录

- [Cosmos3 for PyTorch](#cosmos3-for-pytorch)
- [目录](#目录)
  - [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
  - [准备数据集](#准备数据集)
    - [获取预训练权重](#获取预训练权重)
    - [准备数据集](#准备数据集-1)
  - [快速开始](#快速开始)
    - [训练结果展示](#训练结果展示)

## 简介

### 模型介绍

Cosmos 3 是 NVIDIA 于 2026 年 5 月发布的新一代全模态世界模型, 它首次将语言、图像、视频、音频和动作五种模态统一到一个框架内，既能理解也能生成。 模型采用 Mixture-of-Transformers（MoT） 架构， 在运行时，它提供两个接口：Reasoner（推理器） 用于世界理解、物理推理和任务规划；Generator（生成器） 用于世界模拟、视频生成和动作预测。

* 参考实现： https://github.com/NVIDIA/cosmos/tree/main

### 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

| 模型| 任务列表 | 是否支持 | Released |
| :---:| :---: | :---: | :---: |
|  Cosmos3-Nano generator| SFT训练 | √ | N |
|  Cosmos3-Nano reasoner| SFT训练 | √ | N |

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://link.gitcode.com/?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes&from=https%3A%2F%2Fgitcode.com%2FAscend%2FDrivingSDK%2Fblob%2Fmaster%2Fmodel_examples%2FGR00T-N1.6%2FREADME.md&lang=zh&theme=white)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

| 软件类型| 首次支持版本 |
| :---: | :---: |
| TorchNPU | 26.2.0 |
| CANN | 9.2.0 |
| Triton | 26.2.0 |

### 安装模型环境

当前模型支持的 Pytorch 版本和已知三方库依赖如下表所示：
**表 2** 版本支持表

| 三方库  | 支持版本 |
| --------- | ---------- |
| Python  | 3.12     |
| PyTorch | 2.11   |

0. 激活 CANN 环境
1. 创建环境
   创建conda环境

    ```sh
    conda create -n cosmos3 python=3.12
    conda activate cosmos3
    bash setup.sh
    ```

    参考原仓编译安装 Driving SDK 加速库：[https://gitcode.com/Ascend/DrivingSDK](https://gitcode.com/Ascend/DrivingSDK)

2. 准备模型源码，安装Cosmos3

    在 Cosmos3根目录下，克隆原始仓，替换其中部分代码并安装

    ```sh
    git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
    cd packages/cosmos3
    git checkout 300faa14daab3910be9d303c31708a0a1d6e4371
    cp -f ../../cosmos3_nano.patch ./
    git apply --reject cosmos3_nano.patch
    pip install -e .
    pip install -e packages/diffusers-cosmos3
    pip install -e packages/transformers-cosmos3
    cp -f ../../test/train* ./
    ```

3. 安装ffmpeg

    - 推荐基于conda安装

    ```sh
    # 安装ffmpeg
    conda install -c conda-forge ffmpeg=4.4.2
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 下载源码
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    # 执行此步时环境中可能需要手动下载部分依赖包
    ./configure --prefix=/usr/local/ffmpeg --disable-doc --disable-openssl --enable-avresample --enable-demuxer=dash --enable-hardcoded-tables --enable-libfreetype --enable-libfontconfig --enable-libopenh264 --enable-gnutls --enable-libmp3lame --enable-libvpx --enable-pthreads --enable-gpl --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libxml2 --enable-pic --enable-shared --disable-static --enable-version3 --enable-zlib
    make -j 64
    make install
    cd ..

    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile
    # 运行命令后应正常输出相关配置等信息
    ffmpeg
    ```

4. 安装torchcodec

    - 推荐基于conda安装

    ```sh
    # 安装pybind11
    conda install -c conda-forge pybind11
    # 源码安装torchcodec
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    pip install -e . --no-build-isolation
    cd ..
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 源码安装pybind11
    git clone https://github.com/pybind/pybind11.git
    cd pybind11
    mkdir build && cd build
    cmake .. -DPYBIND11_TEST=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc)
    make install
    cd ..
    # 源码安装torchcodec
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    pip install -e . --no-build-isolation
    cd ..
    ```

## 准备数据集

### 获取预训练权重

下载模型权重Qwen3-VL-8B-Instruct

```sh
pip install huggingface-hub
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir $PRETRAINED_MODEL_PATH
```

下载模型权重Wan2.2_VAE

```sh
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir $VAE_PATH
```

### 准备数据集

* BridgeData2-Subset-Synthetic-Captions数据集下载 —— Cosmos3 generator数据集

  ```sh
  pip install uv
  uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
      --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 --local-dir "$DATASET_DIR"
  ```

* VideoPhy-2数据集下载以及预处理 —— Cosmos3 reasoner数据集

  ```sh
  cd packages/cosmos3
  python -m cosmos_framework.scripts.vlm.prepare_videophy2_from_hf --out_root "$VIDEOPHYSICS_ROOT" --split both
  ```

## 快速开始

进入到Comos3 原始仓目录下（/packages/cosmos3）执行以下指令

* 单机8卡generator训练-精度

```sh
bash train_generator.sh --node 8 --vae_path $VAE_PATH --dataset_path $DATASET_DIR --pretrained_model_path $PRETRAINED_MODEL_PATH
```

* 单机8卡generator训练-性能

```sh
bash train_generator.sh --node 8 --vae_path $VAE_PATH --dataset_path $DATASET_DIR --pretrained_model_path $PRETRAINED_MODEL_PATH --max_iter 50
```

* 单机8卡reasoner训练-精度

```sh
bash train_reasoner.sh --node 8 --vae_path $VAE_PATH --videophysics_root $VIDEOPHYSICS_ROOT --pretrained_model_path $PRETRAINED_MODEL_PATH --max_iter 500
```

* 单机8卡reasoner训练-性能

```sh
bash train_reasoner.sh --node 8 --vae_path $VAE_PATH --videophysics_root $VIDEOPHYSICS_ROOT --pretrained_model_path $PRETRAINED_MODEL_PATH --recompute off
```

### 训练结果展示

**表 3** 训练结果展示表

* Cosmos3-Nano Reasoner

| 芯片          | 卡数 | max steps | Final loss | Iteration time(s)    | Tokens per second    |
| --------------- | ------  | ----------- | ------------ | -------- | -------- |
| 竞品A         | 8p   |  500     | 1.7041    | 26.69 | 38.73 K|
| Atlas 800T A3 | 8p-16die   |  500     | -     | 25.15 | 81.43 K|
| Atlas 950 SuperPod | 8p   |  500     | 1.6761    | 10.74 |95.34 K|

* Cosmos3-Nano Generator

| 芯片          | 卡数 | max steps | Final loss | Iteration time(s)    | Tokens per second   |
| --------------- | ------  | ----------- | ------------ | -------- | -------- |
| 竞品A         | 8p   |  500     |  0.8604   | 36.32 |19.85 K|
| Atlas 800T A3 | 8p-16die   |  500     | -     | 33.81 |42.64 K|
| Atlas 950 SuperPod | 8p   |  500     |   0.8593  | 20.71 |34.81 K|

### FAQ

Atlas 950 SuperPod上reasoner进行训练时，支持关闭重计算，在Atlas 800T A3上关闭重计算可能会爆显存，需要开启重计算
