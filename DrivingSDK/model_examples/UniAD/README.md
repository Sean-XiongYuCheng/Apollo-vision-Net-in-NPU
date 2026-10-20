# UniAD for PyTorch

## 目录

- [UniAD for PyTorch](#uniad-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [UniAD](#uniad)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

自动驾驶是一项高度复杂的技术，需要多个学科领域的知识和技能，包括传感器技术、机器学习、路径规划等方面。自动驾驶还需要适应不同的道路规则和交通文化，与其他车辆和行人进行良好的交互，以实现高度可靠和安全的自动驾驶系统。面对这种复杂的场景，大部分自动驾驶相关的工作都聚焦在具体的某个模块，关于框架性的研讨则相对匮乏。自动驾驶通用算法框架——Unified Autonomous Driving（UniAD）首次将检测、跟踪、建图、轨迹预测，占据栅格预测以及规划整合到一个基于Transformer的端到端网络框架下， 完美契合了”多任务”和“高性能”的特点，是自动驾驶中的重大技术突破。

## 代码实现

- 参考实现：

  ```shell
  url=https://github.com/OpenDriveLab/UniAD
  commit_id=7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
  ```

- 适配昇腾 AI 处理器的实现：

  ```shell
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/UniAD
  ```

# UniAD

## 准备训练环境

### 安装环境

**表 1** 三方库版本支持表

| 三方库  | 支持版本 |
| ------- | -------- |
| PyTorch | 2.7.1 |

### 安装昇腾环境

- 创建 Python 环境

  ```shell
  conda create -n uniad python=3.10 -y
  ```

- 安装CANN和TorchNPU

请参考昇腾社区中《 [PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide) 》文档搭建昇腾环境。本仓已支持表2中软件版本。

**表 2** 昇腾软件版本支持表

| 软件类型          | 支持版本     |
| ----------------- |----------|
| TorchNPU | TorchNPU 26.0.0    |
| CANN              | CANN 9.0.0  |

- 安装Driving SDK加速库，安装方法参考[原仓](https://gitcode.com/Ascend/DrivingSDK)。

  推荐使用依赖安装一键配置脚本，可使用如下指令完成后续步骤的安装：

    ```shell
    bash install_UniAD.sh
    ```

- 在模型根目录下安装依赖

  ```shell
  pip install -r requirements_torch2.7.1.txt
  ```

- 安装mmcv

  ```shell
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cp -f mmcv.patch mmcv/
  cd mmcv
  git apply --reject --whitespace=fix mmcv.patch
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  cd ..
  ```

- 安装mmdet3d

  ```shell
  git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
  cp -f mmdet3d.patch mmdetection3d/
  cd mmdetection3d
  git apply --reject --whitespace=fix mmdet3d.patch
  pip install -v -e . --no-build-isolation
  cd ..
  ```

- 拉取模型源码

  ```shell
  git clone https://github.com/OpenDriveLab/UniAD.git
  cd UniAD
  git checkout 7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
  ```

- 根据操作系统，安装tcmalloc动态库。（可选）

  - OpenEuler系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。

  ```shell
  mkdir gperftools
  cd gperftools
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
  tar -zvxf gperftools-2.16.tar.gz
  cd gperftools-2.16
  ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
  make
  make install
  echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
  ldconfig
  export LD_LIBRARY_PATH=/usr/local/lib/lib/:$LD_LIBRARY_PATH
  export PATH=/usr/local/lib/bin:$PATH
  export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
  ```

  - Ubuntu系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。在安装tcmalloc前，需确保环境中含有autoconf和libtool依赖包。

  安装libunwind依赖：

  ```shell
  git clone https://github.com/libunwind/libunwind.git
  cd libunwind
  autoreconf -i
  ./configure --prefix=/usr/local
  make -j128
  make install
  ```

  安装tcmalloc动态库：

  ```shell
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
  tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
  ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
  make -j128
  make install
  export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
  ```

### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下；

```shell
UniAD
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```

> **说明： 该数据集的训练过程脚本只作为一种参考示例。**

### 准备预训练权重

- 根据原仓Installation章节下载预训练权重bevformer_r101_dcn_24ep.pth和uniad_base_track_map.pth，并放在模型根目录ckpts下：

```shell
UniAD
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── uniad_base_track_map.pth
```

- （可选）可通过修改config文件中的load_from值来更改预训练权重

```shell
# projects/configs/stage1_track_map.py
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"
```

## 快速开始

本任务主要提供**单机多卡**与**多机多卡**训练能力，训练/性能/精度验证统一通过 `migrate_to_ascend/train.sh`。

### 训练任务

#### 开始训练

1. 在模型根目录下，运行统一的训练脚本 `migrate_to_ascend/train.sh`。

    该脚本支持以下参数：

    |         参数         |                        说明                        |   默认值    |
    | :------------------: | :------------------------------------------------: | :---------: |
    | `--stage=stage1/2`   |            选择默认配置（track_map/e2e）             | `stage1`    |
    | `--performance`      |     性能模式（1 epoch + `UNIAD_BRAKE_AT` 提前停止）   | 关闭        |
    | `--num_npu=N`        |                 训练使用的NPU卡数                  |     8      |
    | `--config=PATH`      |              指定配置文件（优先于stage）             | 见stage默认 |

    多机参数通过环境变量传入：`NNODES`（节点数）、`NODE_RANK`（当前节点rank）、
    `MASTER_ADDR`（master地址）、`MASTER_PORT`（端口，默认随机空闲）。
    性能模式提前停止迭代数可通过环境变量 `UNIAD_BRAKE_AT` 调整（默认500）。

    - 单机指定卡数性能训练

      ```shell
      bash migrate_to_ascend/train.sh --stage=stage1 --performance --num_npu=8   # stage1 8卡
      bash migrate_to_ascend/train.sh --stage=stage2 --performance --num_npu=8   # stage2 8卡
      ```

    - 单机8卡精度训练

      ```shell
      bash migrate_to_ascend/train.sh --stage=stage1 --num_npu=8    # stage1
      bash migrate_to_ascend/train.sh --stage=stage2 --num_npu=8    # stage2
      ```

    - 双机多卡训练（每节点8卡，共16卡）

      ```shell
      # master节点
      NNODES=2 NODE_RANK=0 MASTER_ADDR=xx.xx.xx.xx MASTER_PORT=3389 bash migrate_to_ascend/train.sh --stage=stage2
      # 副节点
      NNODES=2 NODE_RANK=1 MASTER_ADDR=xx.xx.xx.xx MASTER_PORT=3389 bash migrate_to_ascend/train.sh --stage=stage2
      ```

2. 训练日志与产物输出到 `test/output/{performance,full}/{stage}/` 下（含
   `train_{performance,full}.log`、`<CaseName>.log` 与 `<CaseName>_perf_report.log`；模型权重存于
   `UniAD/projects/work_dirs/`）。

#### 训练结果

单机八卡

| 阶段     | 芯片          | 卡数 | global batch size | Precision | 性能-单步迭代耗时(ms) | FPS |amota &uarr;|   L2 &darr;  |
|--------| ------------- | ---- |-------------------| --------- |---------------|--------|------|---------|
| stage1 | 竞品A           | 8p   | 8                 | fp32      | 5883        |   1.359 | 0.380 | -      |
| stage1 | Atlas 800T A2 | 8p   | 8                 | fp32      | 7979         |  1.002  | 0.376 | -      |
| stage1 | Atlas 800T A3 | 8p   | 16                 | fp32      | 6802         |  2.352  | - | -      |
| stage1 | Atlas 950 SuperPoD | 8p   | 8                 | fp32      | 3199         |   2.500  | -     | - |
| stage2 | 竞品A           | 8p   | 8                 | fp32      | 3990        |  2.000  | -     | 0.9127 |
| stage2 | Atlas 800T A2 | 8p   | 8                 | fp32      | 5145         |   1.554  | -     | 0.9014 |
| stage2 | Atlas 800T A3 | 8p   | 16                 | fp32      |   3451       |  4.636   | -     | - |
| stage2 | Atlas 950 SuperPoD | 8p   | 8                 | fp32      | 1760         |   4.545  | -     | - |

多机多卡线性度

| 阶段     | 芯片          | 卡数 | global batch size | Precision | 性能-单步迭代耗时(ms) | FPS | 线性度 |
|--------| ------------- | ---- |-------------------| --------- |---------------|--------|------|
| stage1 | Atlas 800T A2*2 | 16p   | 16               | fp32      | 8333        |  1.920  | 96.30%     |
| stage2 | Atlas 800T A2*2 | 16p  | 16                | fp32      | 5412         |   2.956  | 95.12%     |

# 变更说明

2025.02.19: 代码上仓，stage1性能0.6倍竞品，stage2性能0.5倍竞品。

2025.02.26: stage2性能优化，消除部分free time, FA替换优化, stage2性能达到0.54倍竞品。

2025.04.25: 增加多机多卡训练脚本，增加多机多卡训练性能数据。

2025.08.04: 模型性能优化，更新性能数据。

2025.09.01: 优化模型stage 1性能，更新性能数据。

2026.06.10: 新增A5模型适配。

2026.09.03：重构模型代码，优化模型stage2性能，更新性能数据。

# FAQ

1. tcmalloc的动态库文件位置可能因环境配置会有所不同，找不到文件时可以进行搜索，一般安装在`/usr/lib64`或者`/usr/local`目录下：

    ```shell
    find /usr -name libtcmalloc.so*
    ```

    找到对应路径下的动态库文件，`libtcmalloc.so`或者`libtcmalloc.so.版本号`都可以使用。

2. Shapely 2.0+ 不再支持直接迭代 MultiPolygon，如果遇到报错 'MultiPolygon' object is not iterable，
可参考[issue](https://gitcode.com/Ascend/DrivingSDK/issues/184) ，降级 Shapely 到 1.8.x 版本。
