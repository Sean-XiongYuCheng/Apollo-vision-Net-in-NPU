# Driving SDK Installation Guide

This document introduces three ways to deploy the Driving SDK environment: **image installation**, **online installation**, and **source installation**. You are advised to use image installation to get started quickly.

## Hardware Compatibility and Supported OSs

**Table 1** Product Hardware Support List

Before you install Driving SDK, ensure that you use one of the following verified Ascend hardware options:

| Product | Architecture | Supported |
|----------|--------------|-----------|
| <term>Atlas A2 training products</term> | x86_64, aarch64 | √ |
| <term>Atlas A3 training products</term> | x86_64, aarch64 | √ |
| <term>Ascend 950 products</term> | x86_64, aarch64 | √ |

> [!NOTE]
>
> In the table in this section, "√" indicates supported, and "x" indicates unsupported.

The currently verified OSs are Ubuntu 22.04 and openEuler 24.03.

## Preparations Before Installation

1. Download the matching software.

    See the [product version compatibility](../release_note/release_notes.md#related-product-versions) section in the Release Notes to download the software of the matching version.

2. Install the firmware and drivers.

    Click [firmware and drivers](https://hiascend.com/en/hardware/firmware-drivers) and follow the instructions to install the firmware and drivers.

## Installing Driving SDK

### Method 1: Image Installation

> [!NOTE]
>
> - The matching images come with the matching CANN, PyTorch, TorchNPU, and Driving SDK runtime environments preinstalled. You can select an image as needed.
> - The latest images all match the [branch_v26.1.0 branch of Driving SDK](https://gitcode.com/Ascend/DrivingSDK/tree/branch_v26.1.0). Click [Obtain the image](https://www.hiascend.com/en/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176).
> - In the following scenarios, you can install Driving SDK by building a custom image. See [Image Overview](../../../docker/OVERVIEW.md).
>     - You need a specific combination of CANN version and NPU type that the image repository does not cover.
>     - You need to add extra system dependencies or Python packages to the base image.
>     - You need to carry out further custom development based on the Driving SDK image.

#### Prerequisites

The NPU firmware and drivers must be installed on the host machine, Docker 1.11.2 or later must be installed, and the Docker network must be available.

#### Installation Steps

This document uses the `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10` version as an example to describe the Driving SDK image installation steps.

1. Pull the image.

    > [!NOTE]
    >
    > In the [image repository](https://www.hiascend.com/en/developer/ascendhub/detail/696b50584fa04d4a8e99f7894f8eb176), determine the Driving SDK image version based on your hardware model and the required software version. For example:
    >
    > - <term>Atlas A2 training products</term>: `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10`
    > - <term>Atlas A3 training products</term>: `26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10`

    Take the `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10` version as an example and pull the image of this version:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10
    ```

    Check the image name and tag name to confirm that the pull succeeded. Record the image name in the REPOSITORY column as `{IMAGE_NAME}` and the image tag in the TAG column as `{IMAGE_TAG}`:

    ```bash
    docker images
    ```

2. Create the container.

    Run the following command to create a container. Before copying the startup command, note the following two points:
    1. Replace the two paths `{path-to-data}` and `{path-to-weights}` in the `-v` parameter with the actual data and model weight storage paths on the host machine. Otherwise, the host machine paths cannot be mounted correctly after the container starts.
    2. Replace `{IMAGE_NAME}` and `{IMAGE_TAG}` with the image name and tag name in the `docker images` output, for example, `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk` and `26.1.0-9.1.0-torch_npu2.7.1.post6-910b-openeuler24.03-py3.10` here.

    ```bash
    # Run the container with NPU devices (using /dev/davinci0 as an example; to mount more devices, add the mount options with the --device parameter)
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

3. Load the container and confirm the environment status.

   ```bash
   # Query the ID/name of the local container
   docker ps -a
   # Load the container
   docker exec -it <container_id_or_name> bash
   # Confirm that the NPU is working properly
   npu-smi info
   ```

4. Activate the Conda environment.

    ```bash
    conda activate torch2.7.1
    ```

    Because different models require different dependency environments, the image provides only the model environments for BEVFormer/BEVFusion/Sparse4D and the basic development environments for torch2.1/torch2.7.1. For other models, after you pull the image and start the container, you can clone a basic development environment and manually install the dependency environment required by the target model based on its README file. The compatibility among the Conda environments, Python, and PyTorch provided in the image is as follows:

    **Table 2** Compatibility

    | Conda Environment | Python | PyTorch | Purpose |
    |-------------------|--------|---------|---------|
    | torch2.1 | 3.8 | 2.1.0 | General PyTorch development |
    | torch2.7.1 | 3.10 | 2.7.1 | Latest PyTorch features |
    | bevformer | 3.10 | 2.7.1 | BEVFormer model training |
    | bevfusion | 3.10 | 2.7.1 | BEVFusion model training |
    | sparse4d | 3.10 | 2.7.1 | Sparse4D model training |

### Method 2: Online Installation

Driving SDK is published in the [PyPI repository](https://pypi.org/project/mx-driving/) and provides precompiled wheel packages. Compared with source installation, online installation does not require local compilation. You can directly obtain precompiled packages that match your hardware platform.

#### Package Selection

mx_driving provides two independent PyPI packages based on the hardware platform. Both packages provide the same `import mx_driving` interface. Select a package based on your hardware model:

| PyPI Package | Hardware Platform | Installation Command |
|--------------|-------------------|----------------------|
| mx-driving | <term>Atlas A2 training products</term><br><term>Atlas A3 training products</term> | pip install mx-driving |
| mx-driving-a5 | <term>Ascend 950 products</term> | pip install mx-driving-a5 |

#### Installation Steps

1. Install CANN.

    Install the NPU driver firmware and CANN software (Toolkit and ops, and optionally NNAL) of the matching versions and configure the CANN environment variables. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download?versionId=788&ids=d806%2Ch0501%2Ch0601%2Ch0703).

2. Install PyTorch and TorchNPU.

    See the [Installing TorchNPU](https://www.hiascend.com/document/detail/en/Pytorch/2610/installguide/swinstall/docs/en/installation_guide/installation_via_binary_package.md) section in "TorchNPU Software Installation" to obtain the PyTorch and TorchNPU packages of the matching versions.

3. Install mx_driving.

    > [!NOTE]
    >
    >`mx-driving` and `mx-driving-a5` cannot be installed at the same time. Both provide the `mx_driving` Python package, and installing them at the same time causes a conflict.

    - Install without specifying the mx-driving version number.

        ```bash
        # Atlas A2/A3 training products
        pip install mx-driving

        # Ascend 950 products
        pip install mx-driving-a5
        ```

    - Specify the mx_driving version number.

        If you need to install an mx_driving version that matches a specific PyTorch version, specify the version number:

        ```bash
        # Atlas A2/A3 + PyTorch 2.7.1
        pip install mx-driving==26.1.0.post2

        # Ascend 950 + PyTorch 2.7.1
        pip install mx-driving-a5==26.1.0.post2
        ```

        **`mx-driving` Version Compatibility**

        **Table 3** Compatibility of mx-driving (<term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>)

        | Version | CPU Architecture | Python Version | PyTorch Version | TorchNPU Code Branch Name |
        |---------|------------------|----------------|-----------------|---------------------------|
        | 26.1.0.post2 | x86 & aarch64 | 3.9, 3.10, 3.11 | 2.7.1 | v2.7.1-26.1.0 |

        **Table 4** Compatibility of mx-driving-a5 (<term>Ascend 950 products</term>)

        | Version | CPU Architecture | Python Version | PyTorch Version | TorchNPU Code Branch Name |
        |---------|------------------|----------------|-----------------|---------------------------|
        | 26.1.0.post2 | x86 & aarch64 | 3.9, 3.10, 3.11 | 2.7.1 | v2.7.1-26.1.0 |

### Method 3: Source Installation

#### Prerequisites

Install the components that the Driving SDK repository depends on for compilation.
> [!NOTE]
>
> GCC 10.2 is recommended for compiling this repository.

1. Install the Python dependencies.

    Run the `pip3 install -r requirements.txt` command to install the Python dependencies. The `requirements.txt` file must be in the project root directory.

2. (Optional) Install `protobuf-devel-3.14.0`.

    If you need to build the ONNX plugin, perform this step to install `protobuf-devel-3.14.0`.

    On CentOS systems, you can run `yum install protobuf-devel-3.14.0`. Otherwise, set the `ENABLE_ONNX` option in `CMakePresets.json` to `FALSE`. The `CMakePresets.json` file must be in the project root directory.

3. Set the permissions of the runtime environment.

    You are advised to install and run programs as a non-root user and control the permissions on the program installation directories and files: set the directory permissions to 750 and the file permissions to 640. You can control the permissions of files after installation by setting the umask, for example, run `umask 0027` to set the umask to `0027` to ensure that file permissions are correct. For more security-related content, see the [Security Statement](../../../SECURITYNOTE_en.md).

#### Installation Steps

1. Install CANN.

    Install the NPU driver firmware and CANN software (Toolkit and ops, and optionally NNAL) of the matching versions and configure the CANN environment variables. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download?versionId=788&ids=d806%2Ch0501%2Ch0601%2Ch0703).

2. Install PyTorch and TorchNPU.

    See the [Installing TorchNPU](https://www.hiascend.com/document/detail/en/Pytorch/2610/installguide/swinstall/docs/en/installation_guide/installation_via_binary_package.md) section in "TorchNPU Software Installation" to obtain the PyTorch and TorchNPU packages of the matching versions.

3. Download the Driving SDK source code.

    ```bash
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    ```

4. Compile the source code.

    > [!NOTE]
    >
    > This section uses Python 3.9 as an example. If you encounter build issues, see [FAQ](../faq/faq.md) or leave a comment on the issue.

    - If you use <term>Atlas A2 training products</term> or <term>Atlas A3 training products</term>, run the following build commands in the repository root directory:

        ```bash
        bash ci/build.sh --python=3.9
        ```

        Or

        ```bash
        python3.9 setup.py bdist_wheel
        ```

    - If you use <term>Ascend 950 products</term>, run the following build commands in the repository root directory:

        ```bash
        bash ci/build.sh --a5 --python=3.9
        ```

        Or

        ```bash
        python3.9 setup.py bdist_wheel --a5
        ```

    > [!NOTE]
    >
    > The `--python` parameter specifies the Python version used during the build process. It supports Python 3.8 and later, and the default value is 3.8. The generated wheel package is in the `DrivingSDK/dist` directory, and the naming pattern is `mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`.

5. Install Driving SDK.

    ```bash
    cd dist
    pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

    If you need to save the installation log, add the `--log <PATH>` parameter to the `pip3 install` command and control the permissions on the specified directory.

## Upgrade

All three installation methods of Driving SDK install the mx_driving package with pip. To upgrade, first obtain the installation package of the new version, and then complete the upgrade in either of the following ways. With online installation, you can obtain the package directly through pip. With source installation, you need to recompile the source code to generate a wheel package.

### Method 1: Uninstalling the Old Version and Installing the New Version

1. Uninstall the old version.

    ```bash
    pip uninstall mx_driving
    ```

2. Install the new version.

    - Online installation

        ```bash
        # Atlas A2/A3 training products
        pip install mx-driving==<new_version_number>

        # Ascend 950 products
        pip install mx-driving-a5==<new_version_number>
        ```

    - Source installation

        Recompile the source code to generate a new wheel package, and then install it:

        ```bash
        pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
        ```

### Method 2: Force Reinstallation

Use the `--force-reinstall` parameter to directly overwrite and install the new version without uninstalling the old version first:

- Online installation

    ```bash
    # Atlas A2/A3 training products
    pip install --force-reinstall mx-driving==<new_version_number>

    # Ascend 950 products
    pip install --force-reinstall mx-driving-a5==<new_version_number>
    ```

- Source installation

    ```bash
    pip3 install --force-reinstall mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
    ```

## Uninstallation

If you need to remove the Driving SDK package deployment, you can uninstall it by running the following command:

```bash
pip uninstall mx_driving
```
