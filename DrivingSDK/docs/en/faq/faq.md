# Driving SDK FAQ

We have listed some common issues that you may encounter when using Driving SDK, along with the corresponding solutions. If you find that some issues are missing, refer to the [contribution guide](../../../CONTRIBUTING_en.md) and submit a pull request to enrich this list. If you cannot find help here, [submit an issue](https://gitcode.com/Ascend/DrivingSDK/issues) and fill in all the required fields in the description. This helps us locate the problem faster.

## Contents

- [1. Deployment and Installation Issues](#1-deployment-and-installation-issues)
- [2. Compilation Errors](#2-compilation-errors)
- [3. Dataset Errors](#3-dataset-errors)
- [4. Component and Dependency Errors](#4-component-and-dependency-errors)
- [5. Training Errors](#5-training-errors)
- [6. Environment Variable and Configuration Errors](#6-environment-variable-and-configuration-errors)
- [7. LTO and PGO Compilation Optimization Errors](#7-lto-and-pgo-compilation-optimization-errors)
- [8. Other Issues](#8-other-issues)

---

### 1. Deployment and Installation Issues

#### 1.1 Error Reported When Compiling Driving SDK: `fatal error: proto/onnx/ge_onnx.pb.h: No such file or directory`

If you do not need to use `onnx` for inference, disable the `ENABLE_ONNX` option in `CMakePresets.json` by changing `True` to `False`. If you need `onnx`, you can try running `bash ci/docker/ARM/build_protobuf.sh` to install `protobuf`.

#### 1.2 Error Reported When Compiling Driving SDK: `third_party/acl/inc/acl/acl_base.h: No such file or directory`

You may not have installed `torch_npu` successfully. Reinstall it.

#### 1.3 Error Reported When Running Driving SDK: `undefined symbol: _ZN2at4_ops4view4callERKNS_6TensorEN3c108ArrayRefIlEE`

The versions of `torch` and `torch_npu` may not match.

#### 1.4 Error Reported When Running Driving SDK: `opbuild ops error: Invalid socVersion ascend910_93 of xxx`

Install the latest `Ascend-cann-toolkit` package.

### 2. Compilation Errors

#### 2.1 Error Reported When Compiling the `mmdet3d` Component from Source: `File".../setuptools/build_meta.py" ModuleNotFoundError: No module named 'torch'`

You can resolve this error by changing the version of the `setuptools` component. You are advised to use `pip install setuptools==75.3.0`. For the detailed cause and solution, see the open-source community issue [No module named 'torch', why?](https://github.com/facebookresearch/pytorch3d/issues/1892).

#### 2.2 Error Reported When Compiling the `cumm` Component from Source: `TypeError: ccimport() got multiple values for argument 'std'`

You can use the following command to install the files required for compilation:

```shell
pip install ccimport==0.3.7
```

#### 2.3 Error Reported When Compiling `OpenPCDet` from Source: `subprocess.CalledProcessError: Command ['which', 'c++'] return non-zero exit status 1.`

This error occurs because the GCC version on the OS is too high. You are advised to compile with `gcc 10.2`, the version recommended in the [Driving SDK repository README](../../../README_en.md).

### 3. Dataset Errors

#### 3.1 I Already Placed the Dataset in the Specified Path as Required by the README. Why Does Training Report That No `.pkl` File Exists?

Most models currently need to train on a preprocessed dataset. The model README usually describes the preprocessing steps in the "Prepare the Dataset" section.

#### 3.2 Do I Need to Preprocess the Files Again Every Time I Train the Model?

No. You only need to preprocess the dataset once.

#### 3.3 The Model Has Trained or Validated for Some Iterations, but Training or Validation Suddenly Reports a Missing Dataset File, Such as `FileNotFoundError: [Error2] No such file or directory: 'dataset/xxx.pcb.bin'`

Some files may be missing during dataset download or extraction. Check whether the dataset is complete.

### 4. Component and Dependency Errors

#### 4.1 During Model Training, the `yapf` Component Reports: `EOFError: Ran out of input`

This error occurs because `yapf` creates a `~/.cache/YAPF` cache. In a multiprocess environment, some processes create the cache and other processes see that the cache file exists before any content is written to it. They then try to read the file and report `EOFError: Ran out of input`. When you see this error, restart model training. For more details on the cause and solution, see the open-source community issue [\[Bug\] \[Crash\]\[Reproducible\] EOFError: Ran out of input when import yapf with multiprocess](https://github.com/google/yapf/issues/1204).

#### 4.2 During Model Training, the `blas` Component Reports: `ImportError: libblas.so.3: cannot open shared object file: No such file or directory`

This issue occurs because the OS does not have the `openblas` dependency installed, which leaves a dependency missing. The following solution applies to openEuler:

```shell
yum install openblas
find / -name libopenblas*so
ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/libblas.so.3
ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/liblapack.so.3
```

#### 4.3 During Model Training, the `protobuf` Component Reports: `pkg_resources.DistributionNotFound: The 'protobuf' distribution was not found and is required by the application`

Use the following command to resolve the issue:

```shell
pip install protobuf
```

#### 4.4 During Model Training, the Following Error Occurs: `ImportError: cannot import name 'gcd' from 'fraction'`

This issue is caused by a mismatch between the `networkx` component version and the model. You can resolve it by upgrading the dependency with `pip install networkx==3.1`.

#### 4.5 During Model Training, the Following Error Occurs: `ImportError: libGL.so.1, cannot open shared object file: No such file or directory`

When the model installs the `opencv-python` component, you also need to install the same version of `opencv-python-headless`. When it installs the `opencv-contrib-python` component, you also need to install the same version of `opencv-contrib-python-headless`.

#### 4.6 During Model Training, the Following Error Occurs: `libc.so.6: version 'GLIBC_xxx' not found`

This error occurs because the `GLIBC` component version of the OS is too low.

```shell
ldd --version # Check the system Glibc version
```

If the `GLIBC` component version is lower than 2.31, upgrade it. The following command upgrades the `GLIBC` component on openEuler:

```shell
yum upgrade glibc glibc-devel
```

#### 4.7 After Model Training Finishes, the Script Reports a Performance Evaluation Error: `syntax error at or near`

This issue occurs because many model training scripts use `awk` regular expressions to extract performance, accuracy, and other data, while the OS does not support extended regular expressions in `awk`. You need to install `gawk` for support. The following commands show how to install the relevant package on openEuler:

```shell
yum install -y gawk
```

#### 4.8 During Model Training, the Following Error Occurs: `torch has no attribute: uint64_t`

This error occurs because the `safetensors` version does not match the `PyTorch` version. When the `PyTorch` version is 2.1.0, you need a `safetensors` version earlier than 0.6.0. You can change the dependency version with `pip install safetensors==0.5.1`.

#### 4.9 During Model Training, the Following Error Occurs: `AttributeError: module 'attr' has no attribute 's'`

This error occurs because the `attr` component was installed incorrectly. The following commands resolve the issue:

```shell
pip uninstall attr
pip install attrs
```

#### 4.10 During Model Training, the `numpy` Component Reports an Attribute Error, a Function Error, or Another Similar Error

This error occurs because the `NumPy` component version does not match the model. You can usually resolve it with `pip install numpy==1.23.5`. A later `numpy` version can make some deprecated `numpy` usages in the code unavailable.

#### 4.11 When Installing the Model Environment, After Installing `mmcv_full==1.7.2`, an Error Occurs When Installing `mmdet`, `mmdet3d`, or Other `mm`-Related Components: The `mmcv_full` Component Does Not Match the Maximum Compatible Version They Require

You need to apply the corresponding patch file or modify the installation steps as required by the model README.

#### 4.12 During Model Training, the `av2` Component Reports: `TypeError: Type subscription requires python >= 3.9`

This error occurs because the `av2` component version does not match `Python`. If you use `python==3.8`, you can resolve it with `pip install av2==0.2.1`.

#### 4.13 An Error Related to the `decord` or `ffmpeg` Dependency Occurs When Running the Model

This issue may be caused by missing environment variables. After you add them, if the error persists, you can try reinstalling `decord`:

```shell
# Edit the global configuration file
vim /etc/profile.d/ffmpeg.sh

# Add the following content
export PATH="/usr/local/ffmpeg/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

# Make the configuration take effect immediately
source /etc/profile

# If the error persists, recompile and install decord
```

#### 4.14 The `tcmalloc` Dynamic Library File Cannot Be Found When Running the Model

The location of the `tcmalloc` dynamic library file may vary depending on the environment configuration. If you cannot find the file, you can search for it. It is usually installed in the `/usr/lib64` or `/usr/local` directory:

```shell
find /usr -name libtcmalloc.so*
```

After you find the dynamic library file in the corresponding path, either `libtcmalloc.so` or `libtcmalloc.so.<version>` can be used.

#### 4.15 When Installing the Model Environment, `nuplan-devkit` Is Required, and the Installation Reports `No such file or directory: 'gdal-config'`

This issue occurs because the OS does not have the `gmp`, `mpfr`, `OpenBLAS`, `sqlite3`, `curl`, `PROJ`, `GDAL`, and other C++ dependency libraries installed. The following commands show how to install the relevant dependency libraries on openEuler:

```shell
wget https://ftp.swin.edu.au/gnu/gmp/gmp-6.1.0.tar.bz2
yum install m4 libcurl-devel libtiff-devel
tar -jxvf gmp-6.1.0.tar.bz2
cd gmp-6.1.0
./configure --prefix=/usr/local/gmp
make -j128
make install
cd ../
wget https://ftp.swin.edu.au/gnu/mpfr/mpfr-4.1.1.tar.gz
tar -zxvf mpfr-4.1.1.tar.gz
cd mpfr-4.1.1
./configure --prefix=/usr/local/mpfr --with-gmp=/usr/local/gmp
make -j128
make install
cd ../
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.zip
unzip v0.3.24.zip
cd OpenBLAS-0.3.24
make -j128
make PREFIX=/usr/local install
cd ../
wget https://github.com/sqlite/sqlite/archive/refs/tags/version-3.36.0.tar.gz
tar -xzvf version-3.36.0.tar.gz
cd sqlite-version-3.36.0
CFLAGS="-DSQLITE_ENABLE_COLUMN_METADATA=1" ./configure
make -j128
make install
cd ../
wget https://github.com/OSGeo/PROJ/archive/refs/tags/7.2.0.tar.gz
tar -xzvf 7.2.0.tar.gz
cd PROJ-7.2.0
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
cd ../
git clone https://github.com/OSGeo/gdal.git
cd gdal
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```

#### 4.16 When Installing the Model Environment, `OpenEXR` Is Required, but the Installation Fails

This error occurs because the OS does not have `OpenEXR`, `OpenEXR-devel`, and other dependency libraries installed. The following commands show how to install the relevant dependency libraries on openEuler:

```shell
sudo yum makecache
sudo yum install gcc gcc-c++ cmake
sudo yum install OpenEXR
sudo yum install OpenEXR-devel
```

#### 4.17 When Installing the Model Environment, `omegaconf==2.1.0` Is Required, but the Installation Reports `ERROR: Could not find a version that satisfies the requirement omegaconf==2.1.0`

This issue occurs because the `pip` version is too low to install the component correctly. Upgrade `pip` to the latest version.

```shell
pip install --upgrade pip
```

#### 4.18 An Error Occurs When Installing the `h5py` Component: `ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects`

For this issue, you are advised to manage the environment with Anaconda and use `conda install h5py` instead of `pip` to install this dependency.

#### 4.19 An Error Occurs When Installing the `loguru` Component: `error: subprocess-exited-with-error pip subprocess to install build dependencies did not run successfully.`

This error occurs because the `setuptools` component version does not match the `loguru` component version. You can resolve the error with `pip install loguru==0.7.2`. For more details on the cause and solution, see the open-source community issue [pip subprocess to install build dependencies did not run successfully](https://github.com/pypa/packaging-problems/issues/721).

#### 4.20 The Training Process Reports That the `datasets` Component Cannot Be Found

This error occurs because the environment contains a `datasets` component with the same name as the one in the model directory, which prevents the model from finding the `datasets` directory in the model path. You need to uninstall the third-party library with the same name in the model environment.

#### 4.21 `MMCV==2.2.0 is used but incompatible`

This issue is caused by an `MMCV` version conflict. You can resolve it by modifying `__version__ = '2.0.1'` in the `mmcv/version.py` source file of `MMCV`.

#### 4.22 An Error Occurs When `import open3d` Is Run on an x86 Server: `OSError: libX11.so.6: cannot open shared object file`

This issue occurs because the OS lacks the X11-related dependency libraries (`mesa-libGL`, `libX11`, and `libXext`). The following shows how to install them on openEuler:

```shell
yum install -y mesa-libGL libX11 libXext
```

#### 4.23 An Error Occurs When the Model Runs the Data Preprocessing Script: `ModuleNotFoundError: No module named 'tools.data_converter'`

You can resolve this issue by referring to the BEVFormer upstream issue [Issues/223](https://github.com/fundamentalvision/BEVFormer/issues/223) or [pull/241](https://github.com/fundamentalvision/BEVFormer/pull/241).

#### 4.24 An Error Related to `torchcodec` Occurs During Running

This issue may be caused by the original ffmpeg in the environment. You need to rename the original `ffmpeg` directory (for example, `mv ffmpeg ffmpeg_bak`) to avoid conflicts, so that only the conda version is used. You can then recompile and install `torchcodec`.

#### 4.25 Obvious Performance Degradation During Running

This may be an `ffmpeg` configuration issue. You are advised to check whether the `ffmpeg` in the environment was installed in the way recommended by the current model README.

#### 4.26 The CUDA Version Check of `triton` Fails During Inference

When `torch.compile` runs on an NPU, `torch._dynamo` calls a function in `_triton.py` to check the CUDA device capability. However, in the NPU environment, the CUDA-related interfaces return `None`, causing a type comparison error. You can try commenting out this check or refer to the following modification:

```python
_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
if (
    _cap is not None
    and _cap >= (9, 0)
    and not torch.version.hip
):
```

### 5. Training Errors

#### 5.1 The Model Depends on `mmcv_full==1.7.2`, and Training Reports: `File ".../torch/nn/parallel/_functions.py", line 117, in _get_stream: if device.dtype == "cpu": AttributeError: 'int' object has no attribute 'type'`

You can modify it as follows:

```shell
pip show mmcv_full # Obtain the mmcv_full installation path and record it as mmcv_install_path
cd mmcv_install_path
vim mmcv/parallel/_functions.py
```

Add the following statement on line 8 of the file:

```python
from packaging import version
```

Change line 74 of the file from `streams = [_get_stream(device) for device in target_gpus]` to:

```python
if version.parse(torch.__version__) >= version.parse('2.1.0'):
    streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
else:
    streams = [_get_stream(device) for device in target_gpus]
```

For more details on the cause, see the open-source community issue [AttributeError: 'int' object has no attribute 'type'](https://github.com/open-mmlab/mmdetection/issues/10720).

#### 5.2 An `AssertionError` Occurs Occasionally During Model Training and Interrupts Training

You can resolve this issue by restarting training. For the detailed cause and solution, see the open-source community issue [Assertion Error On Finiteness](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus/issues/4).

#### 5.3 The Training Process Reports: `KeyError: 'road_plane'`

You need to modify `tools/cfgs/kitti_models/pointpillar.yaml` and set `USE_ROAD_PLANE` to `False` in the file.

#### 5.4 Training Reports: `RuntimeError: The server socket has failed to listen on any local network address`

This error indicates that the default port is already in use. You can modify the default `PORT` value in the `tools/dist_train.sh` source file of `mmdetection3d`.

#### 5.5 Training Reports: `RuntimeError: ACL stream synchronize failed, error code:507018`

This error is most likely caused by residual processes or other programs occupying resources during dataset preprocessing. Clean up all processes globally and rerun the training.

#### 5.6 NPU Training Results Differ Greatly from GPU Results with the Same Parameter Configuration

This may be caused by inconsistent random number generator states during EMA model initialization. The NPU optimizes this process. In the original logic, the EMA model is first randomly initialized on the CPU and then overwritten by the pre-trained weights of the main model. After optimization, the NPU version skips this redundant step and directly loads the pre-trained weights, which speeds up training startup. You can apply the same modification to the GPU to unify the random number state without affecting the training result.

#### 5.7 An HCCL Error Occurs During Multi-Node Training and the Error Code Is 1 or 7

First check whether residual processes were not fully killed. You can use the following command to terminate the remaining processes on the server:

```shell
pkill -9 python; pkill -9 torchrun
```

#### 5.8 Training Is Stuck at Startup

If training is stuck at startup, check whether a network proxy or the like exists in the environment. Disable the proxy and retry.

### 6. Environment Variable and Configuration Errors

#### 6.1 The Model Training Process Reports: `ImportError:/usr/local/gcc-7.5.0/lib64/libgomp.so.1:cannot allocate memory in static TLS block`

This issue occurs because of `glibc` version compatibility. You can upgrade the `glibc` version or import the environment variable manually:

```shell
export LD_PRELOAD=/usr/local/gcc-7.5.0/lib64/libgomp.so.1
```

#### 6.2 The Model Training Process Reports: `ImportError: {conda_env_path}/bin/../lib/libgomp.so.1:cannot allocate memory in static TLS block`

This issue is similar to the error in 6.1. You can import the environment variable manually:

```shell
export LD_PRELOAD={conda_env_path}/bin/../lib/libgomp.so.1:$LD_PRELOAD
```

#### 6.3 The Model Training Process Reports: `ImportError: {conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`

This issue is similar to the errors in 6.1 and 6.2. You can import the environment variable manually:

```shell
export LD_PRELOAD={conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
```

#### 6.4 Training Reports: `Environment variable [HCCL_IF_IP] is invalid`

When this error occurs, comment out the `export HCCL_IF_IP=...` line in the environment variable setting script `test/env_npu.sh`.

#### 6.5 An Error Occurs When Setting the `TORCH_HCCL_ZERO_COPY` Environment Variable

Currently, this environment variable mainly supports A3. Enabling it on A2 is not recommended for now.

### 7. LTO and PGO Compilation Optimization Errors

#### 7.1 Why Use Compilation Optimization, and Can I Skip It?

Compilation optimization is highly effective in data- and compute-intensive scenarios with major front-end bottlenecks, such as databases and distributed storage. It can substantially improve performance. You can build Python, PyTorch, and TorchNPU from source with the Bisheng compiler to improve model performance. If you do not need the higher performance that compilation optimization provides, you can skip it.

#### 7.2 An `undefined symbol` Error Occurs When You Compile TorchNPU with Optimization: `ImportError: .../torch_npu/lib/libtorch_npu.so: undefined symbol`

This issue occurs because compilation optimization imposes strict version requirements on GCC and other build dependencies. You are advised to use a dedicated image for PyTorch and TorchNPU compilation optimization. For the image usage and compilation optimization steps, see the [Compilation Optimization](https://www.hiascend.com/document/detail/en/Pytorch/2610/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/comp_opt_intro.md) chapter of the *TorchNPU Framework Features* document.

### 8. Other Issues

#### 8.1 How Do I Resolve a Pre-Trained Weight File Download Failure Caused by Network Issues?

If the pre-trained weight file fails to download because of network issues, you can manually download it according to the error link and copy it to the corresponding user directory:

```shell
wget $ckpt_url # Record the link to the pretrained weight file as $ckpt_url
cp $ckpt_file $HOME/.cache/torch/hub/checkpoints/$pretrain_ckpt # Record the name of the downloaded weight file as $ckpt_file, the user home directory as $HOME, and the name of the previously failed download as $pretrain_ckpt
```

#### 8.2 Can I Change the Model Training Configuration Myself?

You are advised to use the training configuration provided in the model README file.

#### 8.3 When Training Different Models, Do I Need to Create a Separate Environment for Each Model? Can I Use the Same Environment to Manage All Models?

You are advised not to manage all models in the same environment. Each model uses different component and dependency versions, and some models use the `tcmalloc` high-performance memory library and compilation optimization techniques to improve performance. If you use the same environment, you may affect the performance of models that do not use `tcmalloc` or compilation optimization techniques.

#### 8.4 How Do I Download Pre-Trained Weights in an Environment Without Network Access or with a Firewall?

In an environment without network access, you can download the pre-trained weight file yourself and copy the downloaded weights to the corresponding directory. For example, to download the `resnet50` pre-trained weights:

```shell
# Manually download the pre-trained weight file
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
# Copy the pre-trained weights to the TorchHub cache directory
cp resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
```

If you use the `SwinTransformer` pre-trained weights, you can download [swin_base_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and copy it to the `~/.cache/torch/hub/checkpoints/` directory.

#### 8.5 An Error Occurs When Running the Model Without Access to the Hugging Face Hub

You can use a Hugging Face mirror source or ModelScope to download the model and dataset yourself when network access is available, and then organize the files according to the file structure in the model README. After downloading, set the environment variables:

```shell
export HF_HOME="/{path_to_caches}/caches/"
export HUGGINGFACE_HUB_CACHE="/{path_to_caches}/caches/"
```

#### 8.6 The Paligemma Weights Cannot Be Automatically Downloaded in an Environment Without Network Access

You can download the [paligemma weights](https://huggingface.co/google/paligemma-3b-pt-224) yourself, record the weight path as `paligemma_weights`, and then run the following command to use the script to replace the model code with the local weight path:

```shell
bash test/paligemma_weights_mod.sh ${paligemma_weights}
```
