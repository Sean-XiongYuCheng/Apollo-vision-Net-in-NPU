# 昇腾 DrivingSDK Docker 镜像

> [English](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/OVERVIEW.md) | 中文

## 快速参考

- DrivingSDK 由 [DrivingSDK 团队](https://gitcode.com/Ascend/DrivingSDK) 维护
- 从哪里获取帮助
  - [DrivingSDK 仓库](https://gitcode.com/Ascend/DrivingSDK)
  - [问题反馈](https://gitcode.com/Ascend/DrivingSDK/issues)
- 支持的架构：`x86_64` (AMD64)、`aarch64` (ARM64)
- 支持的操作系统：Ubuntu 22.04、openEuler 24.03
- 支持的 DrivingSDK 版本：`26.1.0`、`26.0.0`、`7.3.0`

---

## DrivingSDK

DrivingSDK 基于昇腾 CANN、PyTorch 和 TorchNPU，为自动驾驶模型提供 NPU 加速算子和运行环境。

---

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

DrivingSDK 镜像的所有 Tag 遵循以下格式：

```text
<drivingsdk版本>-<cann版本>-torch_npu<TorchNPU版本>-<芯片系列>-<操作系统>-py<python版本>
```

其中，`8.5.1-*` 和 `9.0.0-*` 是历史 Tag，其名称以 CANN 版本开头；`devel` 是源码构建镜像。

| 字段 | 示例值 | 说明 |
| --- | --- | --- |
| `drivingsdk版本` | `26.1.0`、`26.0.0`、`7.3.0` | DrivingSDK 版本号 |
| `cann版本` | `9.1.0`、`9.0.0`、`8.5.1` | CANN 版本号 |
| `TorchNPU版本` | `2.1.0.post17`、`2.7.1.post6` | TorchNPU 版本号 |
| `芯片系列` | `950`、`a3`、`910b` | 目标昇腾芯片系列 |
| `操作系统` | `ubuntu22.04`、`openeuler24.03` | 基础操作系统 |
| `python版本` | `py3.8`、`py3.10` | Python 版本 |
| `devel` | `devel` | 开发态镜像所具有的特殊标记 |

### 最新版本 DrivingSDK `26.1.0`

如下所示是 DrivingSDK 最新发布的 26.1.0 版本的所有镜像，历史版本所有的 Tag 请参考 [Supported Tags](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/supported_tags.md)。

| Tag | Dockerfile | 镜像内容 |
| --- | --- | --- |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-910b-openeuler24.03-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-910b-openeuler24.03-py3.8/Dockerfile) | PyTorch 2.1.0、TorchNPU 2.1.0.post17、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-910b-ubuntu22.04-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-910b-ubuntu22.04-py3.8/Dockerfile) | PyTorch 2.1.0、TorchNPU 2.1.0.post17、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-a3-openeuler24.03-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-a3-openeuler24.03-py3.8/Dockerfile) | PyTorch 2.1.0、TorchNPU 2.1.0.post17、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-a3-ubuntu22.04-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-a3-ubuntu22.04-py3.8/Dockerfile) | PyTorch 2.1.0、TorchNPU 2.1.0.post17、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-950-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-950-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-950-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-950-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-a3-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-a3-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1、TorchNPU 2.7.1.post6、DrivingSDK、模型环境 |

---

## 快速开始

### 前置要求（可选）

#### 安装驱动

请在宿主机安装与所选 Dockerfile 中 CANN 版本兼容的昇腾 NPU 驱动。驱动由宿主机提供，Dockerfile 不负责安装驱动。

---

### 运行 DrivingSDK 容器

```bash
docker run \
    --name drivingsdk_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -it drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10 bash
```

### 如何本地构建

```bash
docker buildx build \
    -t drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10 \
    -f docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10/Dockerfile .
```

### 如何二次开发

```dockerfile
# 在 DrivingSDK 镜像上叠加用户软件
FROM drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10

RUN apt-get update && apt-get install -y <package>
```

`devel` 镜像用于源码开发，会从仓库源码构建 DrivingSDK：

```bash
docker buildx build -t drivingsdk:devel -f docker/devel/Dockerfile .
```

## 安全风险

在使用 Docker 容器运行时，需要注意以下安全风险：

- **使用 root 用户运行：** 容器默认以 root 用户身份运行，可能带来安全隐患。建议在生产环境中创建非特权用户来运行应用程序。

## 许可证

请遵循仓库许可证，以及镜像中预装的 CANN、PyTorch、TorchNPU、Python 和系统软件包各自的许可证。

## 免责声明

发布的昇腾软件镜像均是社区版本，不对商业负责，仅作为生产实践的参考。
