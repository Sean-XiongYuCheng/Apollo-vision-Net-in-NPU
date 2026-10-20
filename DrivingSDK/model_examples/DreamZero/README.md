# DreamZero for PyTorch

## 目录

- [DreamZero for PyTorch](#dreamzero-for-pytorch)
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

DreamZero 是一个面向机器人控制的 World Action Model（WAM，世界动作模型），联合预测机器人动作与未来视频。模型以视频生成骨干学习视觉世界动态，并将动作预测作为联合训练目标，可用于零样本机器人策略和新本体适配。

* 参考实现：[GitHub - dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero.git)

### 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

| 模型 | 任务列表 | 是否支持 | Released |
| :---: | :---: | :---: | :---: |
| DreamZero Wan2.2-TI2V-5B | LoRA训练 | √ | N |

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

| 软件类型 | 首次支持版本 |
| :---: | :---: |
| TorchNPU | 26.0.0 |
| CANN | 9.0.0 |

### 安装模型环境

当前模型支持的 Pytorch 版本和已知三方库依赖如下表所示：

**表 2** 版本支持表

| 三方库 | 支持版本 |
| --------- | ---------- |
| Python  | 3.11     |
| PyTorch | 2.7.1    |

0. 激活 CANN 环境
1. 创建环境
   创建conda环境

    ```sh
    conda create -n dreamzero python=3.11
    conda activate dreamzero
    ```

    参考原仓编译安装 Driving SDK 加速库：[https://gitcode.com/Ascend/DrivingSDK](https://gitcode.com/Ascend/DrivingSDK)

2. 准备模型源码，安装DreamZero

    在 DreamZero 根目录下，克隆原始仓，替换其中部分代码并安装

    ```sh
    git clone https://github.com/dreamzero0/dreamzero.git DreamZero
    cd DreamZero
    git checkout ab790c198fbce33503358efbbd4187ce9a89adf3
    cp -f ../dreamzero.patch ./
    git apply --reject --whitespace=fix dreamzero.patch
    pip install -e .
    cp -r ../test ./
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

## 准备数据集

### 获取预训练权重

下载模型权重 Wan2.2-TI2V-5B

```sh
pip install huggingface-hub
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
```

下载模型权重 Wan2.1-I2V-14B-480P（图像编码器）

```sh
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
```

下载模型权重 UMT5-XXL（tokenizer）

```sh
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

### 准备数据集

以 DROID LeRobot 预处理数据集为例：

```sh
hf download GEAR-Dreams/DreamZero-DROID-Data \
  --repo-type dataset \
  --local-dir ./data/droid_lerobot
```

## 快速开始

进入到 DreamZero 原始仓目录下（/DreamZero）执行以下指令。

* 单机8卡训练精度

```sh
bash train_8p.sh --num-npu=8 --batch-size=1 --max-steps=2000
```

* 单机8卡训练性能

```sh
bash train_performance_8p.sh --num-npu=8 --batch-size=1 --max-steps=100
```

### 训练结果展示

**表 3** 训练结果展示表

| 芯片 | 卡数 | global batch size | max steps | Final loss | FPS |
| --- | --- | --- | --- | --- | --- |
| NVIDIA H100 | 8p | 8 | 2000 | 0.637 | 5.82 |
| Atlas 800T A2 | 8p | 8 | 2000 | 0.640 | 2.01 |
