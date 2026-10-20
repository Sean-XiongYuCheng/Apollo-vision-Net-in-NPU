# DiffusionDriveV2

## 目录

- [DiffusionDriveV2](#diffusiondrivev2)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集与权重](#准备数据集与权重)
  - [数据集下载](#数据集下载)
  - [数据缓存生成](#数据缓存生成)
  - [下载权重](#下载权重)
- [快速开始](#快速开始)
  - [精度训练](#精度训练)
  - [性能训练](#性能训练)
  - [模型验证](#模型验证)
  - [训练脚本支持的命令行参数](#训练脚本支持的命令行参数)
  - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

DiffusionDriveV2 是由华中科技大学、地平线及武汉大学联合提出的端到端自动驾驶模型，该模型在前代 DiffusionDrive 的截断扩散框架基础上，创新性地引入强化学习来约束低质量模式并探索更优轨迹。
生成式扩散模型在端到端自动驾驶任务中，普遍存在因模式崩溃而导致轨迹保守且单一的问题。尽管 DiffusionDrive 通过预定义锚点对动作空间进行划分，在一定程度上缓解了该问题，但其对模仿学习的依赖使模型在轨迹多样性与生成质量之间缺乏有效平衡。针对这一挑战，DiffusionDriveV2 提出了基于强化学习的截断扩散建模框架，旨在约束低质量模式并探索高回报轨迹，同时保留模型的多模态生成能力。其核心设计包括：面向轨迹规划的尺度自适应乘性噪声机制、用于锚点内优势估计的 GRPO 方法，以及用以协调锚点间比较的截断 GRPO 策略。实验结果表明，以 ResNet-34 为骨干网络，该模型在 NAVSIM v1 与 v2 基准上分别达到了 91.2 PDMS 和 85.5 EPDMS 的领先性能，有效解决了截断扩散模型在多样性与质量之间的根本性权衡难题。

## 支持任务列表

| 模型             | 任务列表 | 是否支持           |
| ---------------- | -------- | ------------------ |
| DiffusionDriveV2 | 训练     | ✔ |

## 代码实现

* 参考的官方实现：

  ```shell
  url=https://github.com/hustvl/DiffusionDriveV2.git
  commit_id=1cd12a1e155c34dcc471261835444c8d5587580b
  ```

* 适配昇腾AI处理器的实现

  ```shell
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/DiffusionDriveV2
  ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《 [PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide) 》文档搭建昇腾环境。本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| CANN              | 9.2.0 |
| TorchNPU          | 26.2.0 |

## 安装模型环境

当前模型使用的 Python 版本和已知三方库依赖如下表所示，Python 版本为 **3.10**（非官方仓默认的 3.9/2.0.1 组合，请勿按官方 environment.yaml 创建环境）。

**表 2** 版本支持表

| 三方库      | 支持版本 |
| ----------- | -------- |
| Python      | 3.10.20  |
| PyTorch     | 2.7.1    |

安装顺序：

1. 激活CANN环境

2. 创建Python环境

    ```shell
    conda create -n diffusiondrivev2 python=3.10 -y
    conda activate diffusiondrivev2
    ```

3. 参考[安装昇腾环境](#安装昇腾环境) 安装PyTorch 2.7.1和配套的TorchNPU

4. 参考原仓编译安装Driving SDK加速库：[https://gitcode.com/Ascend/DrivingSDK](https://gitcode.com/Ascend/DrivingSDK)

5. 安装本模型依赖

    ```shell
    pip install -r requirements.txt
    ```

6. 在模型目录下应用Patch更新源码

    ```shell
    git clone https://github.com/hustvl/DiffusionDriveV2.git
    cd DiffusionDriveV2
    git checkout 1cd12a1e155c34dcc471261835444c8d5587580b
    cp -r ../migrate_to_ascend ./
    cp -r ../DiffusionDriveV2.patch ./
    git apply --reject --whitespace=fix DiffusionDriveV2.patch
    ```

# 准备数据集与权重

## 数据集下载

1. `navtrain` 并非全量 `trainval` 数据集（完整 `trainval` 传感器数据超 **2000 GB**），而是从中筛选出的 1,192 个有挑战性的场景，用于官方基准训练，可参考官方源代码中的脚本 `download/super_download.sh` 或 `download/download_navtrain.sh` 下载。

2. `navtest` 是从 `test` 数据包中筛选出的 136 个场景，用于 NAVSIM v1 官方测试。可参考官方源代码中的脚本 `download/download_test.sh` 下载。

3. 如需快速验证功能或性能，建议使用 `mini` 数据集，可参考官方源代码中的脚本 `download/download_mini.sh` 下载，可大幅缩短下载与缓存生成时间。

4. 除了上述数据集外，还需要下载 nuplan 高精地图数据，可参考官方源代码中的脚本`download/download_maps.sh` 脚本下载，包含波士顿、匹兹堡、拉斯维加斯、新加坡等城市的矢量化地图数据（.gpkg 格式）。其中`download`下还有全量训练和验证集trainval等的下载脚本。

5. 下载完成后，将文件移至相应位置，并按照以下目录结构组织：

  ```shell
  DiffusionDriveV2
  ├── navsim/                          # 开发工具包
  ├── exp/                             # 实验输出
  └── dataset/
      ├── maps/                        # 地图数据
      ├── navsim_logs/                 # 场景日志
      │   ├── test/
      │   ├── trainval/
      │   └── mini/
      ├── sensor_blobs/                # 传感器数据
      │   ├── test/
      │   ├── trainval/
      │   └── mini/
  ```

## 数据缓存生成

NAVSIM 框架的数据加载和预处理是训练与评估的性能瓶颈，每次从原始传感器数据（.sensor_blobs）和场景日志（navsim_logs）中构建样本，都需要执行地图查询、坐标变换、传感器数据反序列化等高开销操作，若每个 epoch 重复处理，训练时间将成倍增长。可使用下列命令将预处理的特征和PDMS打分所需的环境信息缓存成cache，后续训练或验证时直接load，且不再需要原始dataset数据，能大幅降低数据加载时间：

  ```shell
  # cache dataset for training
  python navsim/planning/script/run_dataset_caching.py agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache   train_test_split=navtrain

  # cache dataset for evaluation
  python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=exp/metric_cache

  # cache dataset for calculating PDMS during training.
  python navsim/planning/script/run_metric_caching.py train_test_split=navtrain cache.cache_path=exp/train_pdm_cache

  # cache dataset for fast evaluation
  python navsim/planning/script/run_dataset_caching.py agent=diffusiondrivev2_rl_agent experiment_name=diffusiondrivev2_cache   train_test_split=navtest cache_path=exp/metric_feature_cache

  ```

由于 PDMS 打分 worker 每次加载都需要先解压 `exp/train_pdm_cache`下的`metric_cache.pkl`，这会降低训练性能，为了避免这一性能瓶颈，可执行下面的脚本预生成一份未压缩的明文副本（`.plane` 文件），省去打分 worker 冷加载时的解压开销，从而提升训练性能：

  ```shell
   python scripts/bench/prepare_pdm_plain_caches.py
   ```

该脚本默认只处理`exp/train_pdm_cache`，且该步骤为可选项，跳过不影响训练正确性（找不到 `.plane` 时训练会自动回退读取压缩文件），但训练性能会略低，代价是磁盘占用增加：`train_pdm_cache` 的体积约从 30 GB 增大到 120 GB，执行前请确认剩余磁盘空间。

## 下载权重

从 HuggingFace 下载以下文件放入 `ckpts/` 目录：

  - `resnet34.a1_in1k`：来自 <https://huggingface.co/timm/resnet34.a1_in1k>
  - `diffusiondrive_navsim_88p1_PDMS`：来自 <https://huggingface.co/hustvl/DiffusionDrive>
  - `diffusiondrivev2_rl.ckpt`、`diffusiondrivev2_sel.ckpt`：来自 <https://huggingface.co/hustvl/DiffusionDriveV2>

为了在模式选择器训练期间利用 GTRS 轨迹作为数据增强，请下载 GTRS 词汇表对应的模拟真值。

  ```shell
  cd gtrs_traj
  wget https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
  ```

完成数据缓存生成和权重/GTRS词汇表下载的最终目录结构如下：

  ```shell
  DiffusionDriveV2
  ├── ckpts/
      ├── resnet34.a1_in1k/
      ├── diffusiondrivev2_rl.ckpt
      ├── diffusiondrivev2_sel.ckpt
      ├── diffusiondrive_nusc_stage2.pth
      └── diffusiondrive_navsim_88p1_PDMS
  ├── dataset/                         # 有数据缓存后非必需
      ├── maps/                        # 地图数据
      ├── navsim_logs/                 # 场景日志
      │   ├── test/
      │   ├── trainval/
      │   └── mini/
      ├── sensor_blobs/                # 传感器数据
      │   ├── test/
      │   ├── trainval/
      │   └── mini/
  ├── exp/                             # 数据缓存及实验输出目录
      ├── diffusiondrivev2_cache/
      ├── metric_cache/
      ├── metric_feature_cache/
      ├── training_cache/
      ├── train_pdm_cache/
  ├── gtrs_traj/
      ├── 16384.npy
      ├── navtrain_16384.pkl
  └── navsim/                          # 开发工具包
  ```

# 快速开始

## 精度训练

精度训练需要先完成 RL 训练，再使用本次 RL 训练得到的权重继续进行 SEL 训练，最后使用 SEL 权重执行
`navtest` 全量 PDMS 评估。推荐使用 `train_full.sh` 一条命令完成完整流程：

```shell
# 全量精度训练: RL 10 epochs -> SEL 20 epochs -> navtest PDMS 评估,默认8卡
bash migrate_to_ascend/train_full.sh
```

## 性能训练

性能训练用于快速测量训练性能。RL 和 SEL 阶段分别从 `ckpts/` 下的初始权重启动，不依赖本次运行产生的
RL 权重，默认每个阶段训练 1 个 epoch 并输出性能数据。

```shell
# RL/SEL分别使用官方权重各训练 1 epoch测试性能，默认8卡，可指定卡数
bash migrate_to_ascend/train_performance.sh --rank_size 8
```

## 模型验证

模型验证用于传入指定的模型权重并计算 PDMS 分数，不进行训练。脚本封装了官方
`run_pdm_score_fast.py` 的缓存式快速评估：

```shell
# 使用指定 SEL 权重进行 navtest PDMS 评估，并打印评估分数
bash migrate_to_ascend/eval_pdm_score.sh --ckpt /path/to/your/ckpt
```

## 训练脚本支持的命令行参数

`migrate_to_ascend/train.sh` 是底层通用训练脚本，可通过 `--stage rl|sel|all` 单独运行或串联 RL/SEL
阶段。常规使用直接选择上面的 `train_full.sh`、`train_performance.sh` 和
`eval_pdm_score.sh`：

**表 3** `migrate_to_ascend/train.sh`支持参数

| 参数 | 含义 | 默认值 |
| ---- | ---- | ------ |
| `--stage S` | 运行阶段：`all`\|`rl`\|`sel` | all |
| `--rl_epochs N` | RL 阶段 epochs | 10 |
| `--sel_epochs N` | SEL 阶段 epochs | 20 |
| `--rank_size N` | 使用的 NPU 卡数 | 8 |
| `--batch_size N` | 每卡 batch size | 64 |

## 训练结果

### 性能训练结果

**表 4** 性能训练结果展示表

| 阶段   | 芯片   | 卡数 | global batch size | epoch | Precision | 平均step耗时(s) | FPS |
| ----   | ----   | ---- |  ----------------- | ---- | --------- |  --------------- | --- |
| RL stage   |     竞品A              | 8p | 512 | 1 |16-mixed | 8.298 | 61.7 |
| RL stage   |     Atlas 800T A3      | 16p | 1024 | 1 | 16-mixed | 2.917 | 351.1 |
| RL stage   |     Atlas 950 SuperPoD | 8p | 512 | 1 | 16-mixed | 1.405 | 364.5 |
| SEL stage   |     竞品A             | 8p | 512 | 1 | 16-mixed | 15.422 | 33.2 |
| SEL stage   |     Atlas 800T A3     | 16p | 1024 | 1 | 16-mixed | 4.085 | 250.7 |
| SEL stage   |     Atlas 950 SuperPoD | 8p | 512 | 1 | 16-mixed | 1.808 |283.2 |

### 精度训练结果

**表 5** 精度训练结果展示表

| 芯片   | 训练方式 | 卡数 | global batch size | Precision |  PDMS分数(%) |
| ----   | ------ | ---- |  ----------------- | --------- |  ---- |
| 竞品A              | RL 10 epoch+SEL 20 epoch | 8p | 512 | 16-mixed |  90.71 |
| Atlas 800T A3      | RL 10 epoch+SEL 20 epoch | 8p | 512 | 16-mixed |  90.79 |
| Atlas 950 SuperPoD | RL 10 epoch+SEL 20 epoch | 8p | 512 | 16-mixed |  90.62 |

# 变更说明

2026.08：首次发布。

# FAQ

Q: 安装nuplan devkit的依赖包时无法成功安装Fiona，报错：No such file or directory: 'gdal-config', 如何解决？

A: 这个错误的原因是 Fiona 作为一个对 GDAL 的 Python 封装，在安装时需要调用 gdal-config 脚本来获取 GDAL 的编译参数和头文件路径。系统中缺少 gdal-config，通常是因为 GDAL 的开发包（gdal-devel）未安装，可以参考以下几种方式来安装GDAL：

1. conda 安装

    conda 会自动处理 GDAL 及其依赖的二进制库（如 PROJ、GEOS 等），无需手动安装系统级依赖：

    ```shell
    conda install -c conda-forge gdal
    ```

    安装完成后，gdal-config 会自动出现在 PATH 中，之后再次尝试安装 Fiona 即可。

2. yum 安装（CentOS / Rocky Linux / openEuler）

    适用于使用 yum 包管理的 Linux 发行版：

    ```shell
    # 安装 GDAL 运行库
    sudo yum install -y gdal

    # 安装 GDAL 开发包（提供 gdal-config 及头文件）
    sudo yum install -y gdal-devel

    # 安装其他常用依赖
    sudo yum install -y proj-devel geos-devel
    ```

3. 源码安装

    适用于无法使用包管理工具或需要定制 GDAL 版本的环境：需要手动安装gmp, mpfr, OpenBLAS, sqlite3, curl, PROJ, GDAL等一些C++依赖库，此处提供采用源码编译安装的方法。

    - 安装gmp

    ```shell
    wget https://ftp.swin.edu.au/gnu/gmp/gmp-6.1.0.tar.bz2
    tar -jxvf gmp-6.1.0.tar.bz2
    cd gmp-6.1.0
    ./configure --prefix=/usr/local/gmp
    # 如果报错：error: No usable m4 in $PATH or /usr/bin (see config.log for reasons).，说明没有安装m4，使用yum install m4，然后再执行
    make
    make install
    ```

    - 安装mpfr的命令

    ```shell
    wget https://ftp.swin.edu.au/gnu/mpfr/mpfr-4.1.1.tar.gz
    tar -zxvf mpfr-4.1.1.tar.gz
    cd mpfr-4.1.1
    ./configure --prefix=/usr/local/mpfr --with-gmp=/usr/local/gmp
    # 如果报错：configure: error: gmp.h can't be found, or is unusable，替换为 ./configure --with-gmp=/usr/local/gmp
    make
    make install
    ```

    - 安装OpenBLAS

    ```shell
    wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.zip
    unzip v0.3.24.zip
    cd OpenBLAS-0.3.24
    make -j8
    make PREFIX=/usr/local install
    ```

    - 安装sqlite3

    ```shell
    wget https://github.com/sqlite/sqlite/archive/refs/tags/version-3.36.0.tar.gz
    tar -xzvf version-3.36.0.tar.gz
    cd sqlite-version-3.36.0
    CFLAGS="-DSQLITE_ENABLE_COLUMN_METADATA=1" ./configure
    make
    make install
    ```

    - 安装curl

    ```shell
    yum install libcurl-devel
    ```

    - 安装PROJ

    ```shell
    wget https://github.com/OSGeo/PROJ/archive/refs/tags/7.2.0.tar.gz
    tar -xzvf 7.2.0.tar.gz
    cd PROJ-7.2.0
    mkdir build
    cd build
    cmake ..
    如果报Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR)： yum install libtiff-devel
    cmake --build .
    cmake --build . --target install
    ```

    - GDAL编译安装 (编译需要约1小时)

    ```shell
    git clone https://github.com/OSGeo/gdal.git
    cd gdal
    mkdir build
    cd build
    cmake ..
    cmake --build .
    cmake --build . --target install
    ```

    依赖库安装成功后，Fiona可以正常安装。
