# Pi-0.5

## 目录

- [Pi-0.5](#pi-05)
  - [目录](#目录)
  - [概述](#概述)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
  - [前期准备](#前期准备)
    - [环境准备](#环境准备)
      - [安装昇腾环境](#安装昇腾环境)
      - [安装模型环境](#安装模型环境)
    - [数据准备](#数据准备)
    - [权重准备](#权重准备)
  - [快速开始](#快速开始)
    - [执行训练](#执行训练)
    - [训练结果](#训练结果)
  - [变更说明](#变更说明)
  - [FAQ](#faq)

## 概述

本仓提供 pi0.5 模型基于openpi的昇腾适配。

### 模型介绍

具身智能模型 pi0.5 是一款视觉 - 语言 - 动作 (VLA) 模型，通过多源数据融合和分层推理实现开放世界泛化能力，能在完全陌生环境中执行复杂任务并持续操作 10-15 分钟。
该模型采用 "双系统" 架构，高层决策与底层执行协同，能理解任务语义并拆解复杂流程，已在家庭服务、工业自动化等领域展示应用潜力。
作为 pi0 的升级版，pi0.5 通过异构数据协同训练显著提升泛化性能，可在训练时未见过的场景中保持与特定环境训练模型相当的执行效果。

### 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: |
| Pi-0.5 |   训练   |    ✔     |    N     |

### 代码实现

- 参考实现：

    ```shell
    url=https://github.com/Physical-Intelligence/openpi.git
    commit_id=15a9616a00943ada6c20a0f158e3adb39df2ccac
    ```

- 适配昇腾 AI 处理器的实现：

    ```shell
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Pi-0.5/openpi
    ```

## 前期准备

### 环境准备

#### 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| TorchNPU | 26.0.0 |
|       CANN        | 9.0.0  |

#### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| Python | 3.11 |
| PyTorch |   2.7.1  |

0. 激活 CANN 环境

1. 创建conda环境

    ```shell
    conda create -n openpi python=3.11
    conda activate openpi
    ```

2. 拉取 Driving SDK 代码仓

    ```shell
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    cd DrivingSDK/model_examples/Pi-0.5/openpi
    ```

3. 安装 openpi

    在 openpi 目录下，克隆原始仓，切换至指定 commit 并应用 patch

    ```shell
    git clone https://github.com/Physical-Intelligence/openpi.git
    cd openpi
    git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
    git apply ../openpi.patch
    pip install -e .
    ```

4. 替换 transformers 文件

    确认 transformers 版本是4.53.2

    ```shell
    pip show transformers
    ```

    查找transformers安装目录,并用openpi提供文件替换

    ```shell
    transformers_path=$(pip show transformers | grep Location | awk '{print $2}'/transformers)
    echo $transformers_path
    cp -r ./src/openpi/models_pytorch/transformers_replace/* $transformers_path/
    ```

### 权重准备

1. 下载Pi-0.5预训练权重，将权重的绝对路径记作 pi05_weights

   ```shell
   python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"
   ```

2. 下载的预训练权重是jax格式，需要转换pytorch格式，转换后保存的绝对路径记作 pi05_torch_weights

    ```shell
    python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir $pi05_weights \
    --config_name pi05_libero \
    --output_path $pi05_torch_weights
    ```

3. 下载 tokenizer 模型，存的绝对路径记作 paligemma_tokenizer

    ```shell
    hf download ndkhanh95/Paligemma tokenizer.model --local-dir $paligemma_tokenizer
    ```

### 数据准备

1. 下载[libero_10_no_noops_1.0.0_lerobot数据集](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot/tree/main)，将数据集的绝对路径记作 dataset_path

2. 生成数据集对应norm_stats.json

    ```shell
    export LIBERO_DATASET_REPO_ID=$dataset_path
    python scripts/compute_norm_stats.py --config-name pi05_libero
    ```

## 快速开始

本任务主要提供A2单机训练

### 执行训练

进入 DrivingSDK\model_examples\Pi-0.5\openpi\openpi 目录

```shell
conda activate openpi
export LIBERO_DATASET_REPO_ID=$dataset_path
export PALIGEMMA_TOKENIZER_PATH=$paligemma_tokenizer
export PYTORCH_WcEIGHT_PATH=$pi05_torch_weights
```

- 单机单卡

    ```shell
    python scripts/train_pytorch.py pi05_libero --exp_name npu_train --save_interval 1000  --batch_size 32
    ```

- 单机8卡

    ```shell
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_libero --exp_name multi_npu_train --save_interval 1000
    ```

    ---
    参数说明

    | 参数| 可选/必选 | 说明 |
    |------|---------|------|
    |--exp_name | 必选| 训练任务名|
    |--save_interval | 可选| 保存间隔step|
    |--batch_size | 可选| batch size|

### 训练结果

- A2单机8卡

    |  NAME       | Precision     |     iterations    |    global batchsize      |    training loss      |     FPS      |
    |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
    |  竞品 A   |      bf16    |        10k     |     256    |        0.010   |    445.2    |
    |  Atlas 800T A2   |     bf16    |        10k     |      256    |        0.010   |      347.1    |

    ---
    字段说明

    |字段|说明|
    |-----|-----|
    |NAME|芯片类别|
    |Precision|训练精度|
    |iterations|训练迭代步数|
    |global batchsize|每次迭代的样本总数|
    |training loss|训练结束时的损失|
    |FPS|平均每秒处理的样本总数|

## 变更说明

2026.7.25：首次发布。

## FAQ

无
