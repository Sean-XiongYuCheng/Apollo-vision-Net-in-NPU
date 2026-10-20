# Driving SDK安装指导

本文介绍部署Driving SDK环境的三种方式：**镜像安装**、**在线安装**和**源码安装**。我们推荐您使用镜像安装，以便快速上手。

## 硬件配套和支持的操作系统

**表 1** 产品硬件支持列表

在Driving SDK安装前，请确保选择以下经过验证的昇腾硬件：

| 产品 | 架构 |  是否支持 |
|----------|------|------|
|<term>Atlas A2训练系列产品</term>| x86_64, aarch64 |  √ |
|<term>Atlas A3训练系列产品</term>| x86_64, aarch64 |  √ |
|<term>Ascend 950PR&950DT系列产品</term>| x86_64, aarch64 |  √ |

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

当前经过验证的操作系统为Ubuntu 22.04 、openEular 24.03。

## 安装前准备

1. 下载配套软件

    请参见《版本说明》中的[相关产品版本配套说明](../release_note/release_notes.md#相关产品版本配套说明)章节，下载对应版本的软件。

2. 安装固件与驱动

    请单击[固件与驱动](https://hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

## 安装Driving SDK

### 方式一：镜像安装

> [!NOTE]
>
> - 配套镜像已预装配套的CANN、PyTorch、TorchNPU插件及Driving SDK运行环境，您可根据需要选用。
> - 最新镜像均配套[Driving SDK的branch_v26.1.0分支](https://gitcode.com/Ascend/DrivingSDK/tree/branch_v26.1.0)，请单击[获取镜像](https://www.hiascend.com/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176)
> - 如存在以下场景，可通过自定义构建镜像安装，参见[镜像概述](../../../docker/OVERVIEW.zh.md)。
>     - 需要使用特定CANN版本或NPU类型组合，而镜像仓库未覆盖。
>     - 需要在基础镜像上添加额外的系统依赖或Python包。
>     - 需要基于Driving SDK镜像进行二次定制开发。

#### 前置依赖

宿主机需已完成NPU固件与驱动安装，并已安装1.11.2及以上版本的Docker，且Docker网络可用。

#### 安装步骤

本文以`26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10`版本为例，介绍Driving SDK镜像安装步骤。

1. 拉取镜像

    > [!NOTE]
    >
    > 在[镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176)中，根据硬件型号和所需软件版本确定Driving SDK镜像版本。例如：
    >
    > - <term>Atlas A2训练系列产品</term>：`26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10`
    > - <term>Atlas A3训练系列产品</term>：`26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10`

    以`26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10`版本为例，拉取对应版本的镜像：

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10
    ```

    查看镜像名称和标签名称，确认拉取成功。将REPOSITORY下的镜像名称记作{IMAGE_NAME}，TAG下的镜像标签记作{IMAGE_TAG}：

    ```bash
    docker images
    ```

2. 创建容器

    执行以下命令创建容器。复制启动命令前，请注意以下两点：
    1. 请将-v参数内的{path-to-data}、{path-to-weights}两处路径，替换为宿主机真实数据、模型权重存储路径，否则容器启动后无法正确挂载宿主机路径。
    2. 请将{IMAGE_NAME}和{IMAGE_TAG}替换为docker images中的镜像名称和标签名称，例如此处的swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk和26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10。

    ```bash
    # 使用NPU设备运行（以使用设备/dev/davinci0为例，如需挂载多张卡，通过--device参数追加挂载）
    docker run -it -d \
    --name drivingsdk \
    --ipc=host \
    --network=host \
    --privileged -u=root \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v {path-to-data}:/data \
    -v {path-to-weights}:/weights \
    {IMAGE_NAME}:{IMAGE_TAG} \
    /bin/bash
    ```

3. 加载容器并确认环境状态

   ```bash
   # 查询本地运行中的容器ID/名称
   docker ps -a
   # 加载容器
   docker exec -it 容器ID/名称 bash
   # 确认NPU是否可以正常使用
   npu-smi info
   ```

4. 激活conda环境

    ```bash
    conda activate torch2.7.1
    ```

    由于不同模型的依赖环境存在差异，镜像中仅预先提供了BEVFormer/BEVFusion/Sparse4D的模型环境，以及torch2.1/torch2.7.1的基础开发环境。对于其他模型，用户在拉取镜像并启动容器后，可以克隆基础开发环境，并根据目标模型的README文件手动安装该模型所需的依赖环境。镜像中提供的conda环境、Python和PyTorch配套关系如下表：

    **表 2** 配套说明

    | conda 环境 | Python | PyTorch | 用途 |
    |-----------|--------|---------|------|
    | torch2.1 | 3.8 | 2.1.0 | 通用PyTorch开发 |
    | torch2.7.1 | 3.10 | 2.7.1 | 最新PyTorch特性 |
    | bevformer | 3.10 | 2.7.1 | BEVFormer模型训练 |
    | bevfusion | 3.10 | 2.7.1 | BEVFusion模型训练 |
    | sparse4d | 3.10 | 2.7.1 | Sparse4D模型训练 |

### 方式二：在线安装

Driving SDK已发布至[PyPI仓库](https://pypi.org/project/mx-driving/)，提供预编译的whl包。相比源码安装，在线安装无需本地编译，可直接获取与硬件平台匹配的预编译包。

#### 包选择

mx_driving 根据硬件平台提供两个独立的PyPI包，两者提供相同的`import mx_driving`接口，请根据硬件型号选择：

| PyPI包名 | 硬件平台 | 安装命令 |
|-----------|----------|----------|
| `mx-driving` | <term>Atlas A2训练系列产品</term><br><term>Atlas A3训练系列产品</term> | `pip install mx-driving` |
| `mx-driving-a5` | <term>Ascend 950PR&950DT系列产品</term> | `pip install mx-driving-a5` |

#### 安装步骤

1. 安装CANN

    安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops，可选NNAL）并配置CANN环境变量，具体请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》。

2. 安装PyTorch以及TorchNPU

    请参考《[TorchNPU 快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download)》，获取配套版本的PyTorch以及TorchNPU软件包。

3. 安装mx_driving

    > [!NOTE]
    >
    > `mx-driving`和`mx-driving-a5`不能同时安装，两者均提供`mx_driving`Python包，同时安装会发生冲突。

    **不指定mx-driving版本号**
    <!-- npu="A3,910b" id3 -->
    - Atlas A2/A3训练系列产品

        ```bash
        pip install mx-driving
        ```
    <!-- end id3 -->

    <!-- npu="950" id4 -->
    - Ascend 950PR&950DT系列产品

        ```bash
        pip install mx-driving-a5
        ```
    <!-- end id4 -->

    **指定 mx_driving版本号**

    如需安装与特定PyTorch版本匹配的mx_driving，可指定版本号：

    <!-- npu="A3,910b" id6 -->
    - Atlas A2/A3 + PyTorch 2.7.1

        ```bash
        pip install mx-driving==26.1.0.post2
        ```
    <!-- end id6 -->

    <!-- npu="950" id7 -->
    - Ascend 950PR&950DT + PyTorch 2.7.1

        ```bash
        pip install mx-driving-a5==26.1.0.post2
        ```
    <!-- end id7 -->

     **mx-driving版本配套表，请参考下表：**
     <!-- npu="A3,910b" id1 -->
     **表3** mx-driving（<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>）配套关系

    | 版本 | CPU架构 | Python版本 | PyTorch版本 | TorchNPU代码分支名称 |
    |------|----------|-------------|--------------|----------------|
    | 26.1.0.post2 | x86 & aarch64 | 3.9, 3.10, 3.11 | 2.7.1 | v2.7.1-26.1.0 |
    <!-- end id1 -->

    <!-- npu="950" id2 -->
    **表4** mx-driving-a5（<term>Ascend 950PR&950DT系列产品</term>）配套关系

    | 版本 | CPU架构 | Python版本 | PyTorch版本 | TorchNPU代码分支名称 |
    |------|----------|-------------|--------------|----------------|
    | 26.1.0.post2 | x86 & aarch64 | 3.9, 3.10, 3.11 | 2.7.1 | v2.7.1-26.1.0|
    <!-- end id2 -->

### 方式三：源码安装

#### 前置依赖

安装Driving SDK仓编译依赖的组件。
> [!NOTE]
>
> 编译本仓推荐使用GCC 10.2版本。

1. 安装Python依赖

    使用`pip3 install -r requirements.txt`命令安装Python依赖，`requirements.txt`文件需位于项目根目录下。

2. （可选）安装`protobuf-devel-3.14.0`

    如需编译ONNX插件，请执行此步骤安装`protobuf-devel-3.14.0`。

    在CentOS系统上可执行`yum install protobuf-devel-3.14.0`，否则请将`CMakePresets.json`中的`ENABLE_ONNX`选项改为`FALSE`，`CMakePresets.json`文件需位于项目根目录下。

3. 设置运行环境权限

    安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如使用`umask 0027`将umask调整为`0027`，以保证文件权限正确。更多安全相关内容请参见《[安全声明](../../../SECURITYNOTE.md)》的说明。

#### 安装步骤

1. 安装CANN

    安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops，可选NNAL）并配置CANN环境变量，具体请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》。

2. 安装PyTorch以及TorchNPU

    请参考《[TorchNPU 快速安装](https://www.hiascend.com/zh/developer/software/ai-frameworks/pytorch/download)》章节，获取配套版本的PyTorch以及TorchNPU软件包。

3. 下载Driving SDK源码

    ```bash
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    ```

4. 编译源码

    > [!NOTE]
    >
    > 本节以使用Python3.9为例。如遇到编译问题，可查看[FAQ](../faq/faq.md)或在Issue中留言。

    <!-- npu="A3,910b" id8 -->
    - 若使用<term>Atlas A2训练系列产品</term>或者<term>Atlas A3训练系列产品</term>，请在仓库根目录下执行以下编译命令：

        ```bash
        bash ci/build.sh --python=3.9
        ```

        或者

        ```bash
        python3.9 setup.py bdist_wheel
        ```
    <!-- end id8 -->

    <!-- npu="950" id9 -->
    - 若使用<term>Ascend 950PR&950DT系列产品</term>，请在仓库根目录下执行以下编译命令：

        ```bash
        bash ci/build.sh --a5 --python=3.9
        ```

        或者

        ```bash
        python3.9 setup.py bdist_wheel --a5
        ```
    <!-- end id9 -->

    > [!NOTE]
    >
    > `--python`参数指定编译过程中使用的Python版本，支持Python3.8及以上，缺省值为3.8。生成的whl包位于`DrivingSDK/dist`目录下，命名规则为`mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`。

5. 安装Driving SDK

    ```bash
    cd dist
    pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

    如需保存安装日志，可在`pip3 install` 命令后添加 `--log <PATH>`参数，并对指定目录做好权限控制。

## 升级

Driving SDK的三种安装方式均是通过pip安装mx_driving软件包。升级时，请先获取新版本的安装包（在线安装方式直接通过pip获取，源码安装方式需重新编译生成whl包），然后参考以下任一方式完成升级。

### 方式一：卸载旧版本后安装新版本

1. 卸载旧版本

    ```bash
    pip uninstall mx_driving
    ```

2. 安装新版本

    **在线安装方式**
    <!-- npu="A3,910b" id10 -->
    - Atlas A2/A3训练系列产品

        ```bash
        pip install mx-driving==<新版本号>
        ```
    <!-- end id10 -->

    <!-- npu="950" id11 -->
    - Ascend 950PR&950DT系列产品

        ```bash
        pip install mx-driving-a5==<新版本号>
        ```
    <!-- end id11 -->

    **源码安装方式**

    重新编译源码生成新的whl包后安装：

    ```bash
    pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

### 方式二：强制重新安装

使用`--force-reinstall`参数直接覆盖安装新版本，无需先卸载旧版本：

**在线安装方式**

<!-- npu="A3,910b" id12 -->
- Atlas A2/A3训练系列产品

    ```bash
    pip install --force-reinstall mx-driving==<新版本号>
    ```
<!-- end id12 -->

<!-- npu="950" id13 -->
- Ascend 950PR&950DT系列产品

     ```bash
    pip install --force-reinstall mx-driving-a5==<新版本号>
    ```
<!-- end id13 -->

**源码安装方式**

    ```bash
    pip3 install --force-reinstall mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

## 卸载

用户如需移除Driving SDK软件包部署，可参考以下命令进行卸载。

```bash
pip uninstall mx_driving
```
