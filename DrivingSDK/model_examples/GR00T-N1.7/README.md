# GR00T-N1.7 for PyTorch

# 目录

- [GR00T-N1.7 for PyTorch](#gr00t-n17-for-pytorch)
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

Isaac GR00T-N1.7 为 GR00T-N1.6升级版

* 参考实现： [GitHub - NVIDIA/Isaac-GR00T: NVIDIA Isaac GR00T N1.7 - A Foundation Model for Generalist Robots. · GitHub](https://github.com/NVIDIA/Isaac-GR00T/tree/main)

### 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

| 模型| 任务列表 | 是否支持 | Released |
| :---:| :---: | :---: | :---: |
|  GR00T-N1.7| SFT训练 | √ | N |

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

| 软件类型| 首次支持版本 |
| :---: | :---: |
| TorchNPU | 26.0.0 |
| CANN | 9.0.0 |

### 安装模型环境

当前模型支持的 Pytorch 版本和已知三方库依赖如下表所示：
**表 2** 版本支持表

| 三方库  | 支持版本 |
| --------- | ---------- |
| Python  | 3.10     |
| PyTorch | 2.7.1    |

0. 激活 CANN 环境
1. 创建环境
   创建conda环境

    ```sh
    conda create -n gr00t python=3.10
    conda activate gr00t
    ```

    参考原仓编译安装 Driving SDK 加速库：[https://gitcode.com/Ascend/DrivingSDK](https://gitcode.com/Ascend/DrivingSDK)

2. 准备模型源码，安装gr00t

    在 GR00T-N1.7根目录下，克隆原始仓，替换其中部分代码并安装

    ```sh
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
    cd Isaac-GR00T
    git checkout e5749287857afd97b78f1147166137de29746392
    cp -f ../gr00t_n1d7.patch ./
    git apply --reject gr00t_n1d7.patch
    pip install -e .
    cp -f ../test/train* ./
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

下载模型权重GR00T-N1.7-3B

```sh
pip install huggingface-hub
hf download nvidia/GR00T-N1.7-3B --local-dir ./GR00T-N1.7-3B
```

下载模型权重Cosmos-Reason2-2B

```sh
hf download nvidia/Cosmos-Reason2-2B --local-dir ./Cosmos-Reason2-2B
```

### 准备数据集

* 以LIBERO 10微调为例，安装数据集

  ```sh
  huggingface-cli download \
      --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
      --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

  cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
  ```

## 快速开始

进入到GR00T-N1.7 原始仓目录下（/Isaac-GR00T）执行以下指令, 将其中cosmos-model-path,dataset-path,base-model-path后的路径替换为实际Cosmos-Reason2-2B, libero数据集, GR00T-N1.7-3B存放的位置

* 单机8卡训练精度

```sh
bash train_8p.sh --num_gpus 8 --max_steps 20000 --cosmos-model-path ./Cosmos-Reason2-2B --dataset-path ./Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base-model-path ./GR00T-N1.7-3B
```

* 单机16卡训练性能

```sh
bash train_performance_8p.sh --num_gpus 16 --max_steps 1000 --cosmos-model-path ./Cosmos-Reason2-2B --dataset-path ./Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base-model-path ./GR00T-N1.7-3B
```

### 训练结果展示

**表 3** 训练结果展示表

| 芯片          | 卡数 | global batch size | max steps | Final loss | FPS    |
| --------------- | ------ | ------------------- | ----------- | ------------ | -------- |
| 竞品A         | 8p   | 640               | 20000     | 0.0147     | 444.4 |
| Atlas 800T A3 | 4p   | 640              | 20000     | 0.0147     | 460.4 |
| Atlas 800T A3 | 8p   | 640              | 20000     | -    | 752.9 |
| Atlas 800T A3 | 8p   | 1280              | 20000     | -    | 836.6 |
| Atlas 950 SuperPoD | 8p   | 640              | 20000     | 0.0146    | 885.2 |
