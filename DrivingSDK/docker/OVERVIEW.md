# DrivingSDK Docker Images

> English | [中文](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/OVERVIEW.zh.md)

## Quick Reference

- DrivingSDK is maintained by the [DrivingSDK team](https://gitcode.com/Ascend/DrivingSDK)
- Where to get help
  - [DrivingSDK repository](https://gitcode.com/Ascend/DrivingSDK)
  - [Issue Tracker](https://gitcode.com/Ascend/DrivingSDK/issues)
- Supported architectures: `x86_64` (AMD64), `aarch64` (ARM64)
- Supported operating systems: Ubuntu 22.04, openEuler 24.03
- Supported DrivingSDK versions: `26.1.0`, `26.0.0`, `7.3.0`

---

## DrivingSDK

DrivingSDK provides NPU-accelerated operators and runtime environments for autonomous-driving models based on Ascend CANN, PyTorch, and TorchNPU.

---

## Supported Tags and Dockerfile Links

### Tag Naming Rules

All DrivingSDK image tags follow this pattern:

```text
<drivingsdk-version>-<cann-version>-torch_npu<torchnpu-version>-<chip-series>-<os>-py<python-version>
```

The `8.5.1-*` and `9.0.0-*` directories are legacy tags whose names start with the CANN version. The `devel` directory is the source-build image.

| Field | Example values | Description |
| --- | --- | --- |
| `drivingsdk-version` | `26.1.0`, `26.0.0`, `7.3.0` | DrivingSDK version number |
| `cann-version` | `9.1.0`, `9.0.0`, `8.5.1` | CANN version number |
| `torchnpu-version` | `2.1.0.post17`, `2.7.1.post6` | TorchNPU version number |
| `chip-series` | `950`, `a3`, `910b` | Target Atlas chip series |
| `os` | `ubuntu22.04`, `openeuler24.03` | Base operating system |
| `python-version` | `py3.8`, `py3.10` | Python version |
| `devel` | `devel` | Special tag representing a development image |

### LATEST DrivingSDK `26.1.0`

The following table lists all images for the latest DrivingSDK 26.1.0 release. For all historical tags, see [Supported Tags](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/supported_tags.md).

| Tag | Dockerfile | Image contents |
| --- | --- | --- |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-910b-openeuler24.03-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-910b-openeuler24.03-py3.8/Dockerfile) | PyTorch 2.1.0, TorchNPU 2.1.0.post17, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-910b-ubuntu22.04-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-910b-ubuntu22.04-py3.8/Dockerfile) | PyTorch 2.1.0, TorchNPU 2.1.0.post17, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-a3-openeuler24.03-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-a3-openeuler24.03-py3.8/Dockerfile) | PyTorch 2.1.0, TorchNPU 2.1.0.post17, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.1.0.post17-a3-ubuntu22.04-py3.8` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.1.0.post17-a3-ubuntu22.04-py3.8/Dockerfile) | PyTorch 2.1.0, TorchNPU 2.1.0.post17, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-950-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-950-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-950-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-950-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |
| `26.1.0-9.1.0-torch_npu2.7.1.post6-a3-ubuntu22.04-py3.10` | [Dockerfile](https://gitcode.com/Ascend/DrivingSDK/blob/master/docker/26.1.0-9.1.0-torch_npu2.7.1.post6-a3-ubuntu22.04-py3.10/Dockerfile) | PyTorch 2.7.1, TorchNPU 2.7.1.post6, DrivingSDK, model environments |

---

## Quick Start

### Prerequisites (optional)

#### Install Driver

Install an Ascend NPU driver compatible with the CANN version used by the selected Dockerfile. The driver is provided by the host and is not installed by these Dockerfiles.

---

### Running a DrivingSDK Container

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

### How to build

```bash
docker buildx build \
    -t drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10 \
    -f docker/26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10/Dockerfile .
```

### Development

```dockerfile
# Add required software on top of a DrivingSDK image
FROM drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-ubuntu22.04-py3.10

RUN apt-get update && apt-get install -y <package>
```

The `devel` image is intended for source development and builds DrivingSDK from the repository:

```bash
docker buildx build -t drivingsdk:devel -f docker/devel/Dockerfile .
```

## Security Risks

Please be aware of the following security risks when running in Docker containers: 

- **Running as the root user:** Containers run as the root user by default, which may introduce security risks. It is recommended to create a non-privileged user for running applications in production environments.

## License

Refer to the repository license and the licenses of the pre-installed CANN, PyTorch, TorchNPU, Python, and system packages.

## Disclaimer

The released Ascend software images are community versions and are not intended for commercial accountability. They are provided solely as references for production practices.
