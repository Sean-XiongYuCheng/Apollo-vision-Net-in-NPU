# Driving SDK 常见问题FAQ

我们在这里列出了使用Driving SDK时的一些常见问题及其相应的解决方案。如果您发现有一些问题被遗漏，请参考[贡献指南](../../../CONTRIBUTING.md)提PR丰富这个列表。如果您无法在此获得帮助，请[提交Issue](https://gitcode.com/Ascend/DrivingSDK/issues)，并在描述中填写所有必填信息，这有助于我们更快定位问题。

## 部署与安装问题

- **编译Driving SDK时报错：`fatal error: proto/onnx/ge_onnx.pb.h: No such file or directory`**

    答：如果你不需要使用`onnx`进行推理，请在`CMakePresets.json`中关闭`ENABLE_ONNX`选项，将`True`改为`False`。如果需要`onnx`可尝试执行`bash ci/docker/ARM/build_protobuf.sh`安装`protobuf`。

- **编译Driving SDK时报错：`third_party/acl/inc/acl/acl_base.h: No such file or directory`**

    答：你可能没有成功安装`torch_npu`，重新安装即可。

- **运行Driving SDK时报错：`undefined symbol: _ZN2at4_ops4view4callERKNS_6TensorEN3c108ArrayRefIlEE`**

    答：`torch`与`torch_npu`的版本可能不配套。

- **运行Driving SDK时报错：`opbuild ops error: Invalid socVersion ascend910_93 of xxx`**

    答：更换最新的`Ascend-cann-toolkit`套件。

## 编译错误

- **源码编译`mmdet3d`组件时，编译过程报错：`File".../setuptools/build_meta.py" ModuleNotFoundError: No module named 'torch'`**

    答：该报错可通过改变`setuptools`组件的版本解决，建议使用`pip install setuptools==75.3.0`。详细的报错原因和解决方法可参考[开源社区Issue：No module named 'torch', why?](https://github.com/facebookresearch/pytorch3d/issues/1892)

- **源码编译`cumm`组件时，编译过程报错：`TypeError: ccimport() got multiple values for argument 'std'`**

    答：该报错可参考以下语句安装编译所需文件：

    ```shell
    pip install ccimport==0.3.7
    ```

- **源码编译`OpenPCDet`时，执行`python setup.py develop`语句报错：`subprocess.CalledProcessError: Command ['which', 'c++'] return non-zero exit status 1.`**

    答：该报错是由于操作系统的GCC版本过高，推荐使用[Driving SDK仓库README文档](https://gitcode.com/Ascend/DrivingSDK/blob/master/README.md)中的建议版本`gcc 10.2`进行编译。

## 数据集错误

- **已经按照README要求在指定路径下放置数据集，训练时为何会报错无pkl格式文件？**

    答：目前大部分模型需要使用预处理后的数据集训练，通常在模型README文件的"准备数据集"一节说明预处理步骤。

- **需要每一次训练模型时都重新预处理文件吗？**

    答：不需要。数据集只需预处理一次即可。

- **模型已训练或验证一些Iter，但却在训练或验证过程中突然报错缺少数据集文件，如`FileNotFoundError: [Error2] No such file or directory: 'dataset/xxx.pcb.bin'`**

    答：可能在数据集下载或解压过程中缺少文件，需检查数据集是否完整。

## 组件及依赖错误

- **模型训练时，`yapf`组件报错：`EOFError: Ran out of input`**

    答：该报错的原因是，`yapf`组件会创建`~/.cache/YAPF`缓存，在多进程环境中，部分进程创建该缓存后，还未向缓存文件写入内容时，其他进程识别到缓存文件存在，并试图读取文件中的内容，从而报出`EOFError: Ran out of input`错误。遇到此报错时，重新拉起模型训练即可解决。更详细的报错原因及解决方案可参考[开源社区Issue\[Bug\] \[Crash\]\[Reproducible\] EOFError: Ran out of input when import yapf with multiprocess](https://github.com/google/yapf/issues/1204)。

- **模型训练时，`blas`组件报错：`ImportError: libblas.so.3: cannot open shared object file: No such file or directory`**

    答：该问题原因为操作系统未安装openblas依赖，导致依赖缺失，以下给出OpenEuler操作系统的解决方法：

    ```shell
    yum install openblas
    find / -name libopenblas*so
    ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/libblas.so.3
    ln -s /usr/lib64/libopenblas-r0.3.9.so /usr/lib64/liblapack.so.3
    ```

- **模型训练时，`protobuf`组件报错：`pkg_resources.DistributionNotFound: The 'protobuf' distribution was not found and is required by the application`**

    答：该问题的解决方法为：

    ```shell
    pip install protobuf
    ```

- **模型训练过程报错：`ImportError: cannot import name 'gcd' from 'fraction'`**

    答：该问题由`networkx`组件版本与模型不匹配引起，使用`pip install networkx==3.1`升级依赖版本即可。

- **模型训练过程报错：`ImportError: libGL.so.1, cannot open shared object file: No such file or directory`**

    答：当模型安装`opencv-python`组件时，需配套安装相同版本的`opencv-python-headless`组件，安装`opencv-contrib-python`组件时，需配套安装相同版本的`opencv-contrib-python-headless`组件。

- **模型训练过程报错：`libc.so.6: version 'GLIBC_xxx' not found`**

    答：该报错由操作系统Glibc组件版本过低引起。

    ```shell
    ldd --version # 查看系统Glibc版本
    ```

    若Glibc组件版本低于2.31，需升级组件，以下给出OpenEuler操作系统升级Glibc组件的命令：

    ```shell
    yum upgrade glibc glibc-devel
    ```

- **模型训练结束后，脚本评估性能时，报错：`syntax error at or near`**

    答：该问题是由于许多模型训练脚本使用`awk`正则表达式获取性能、精度等数据，而操作系统不支持`awk`的拓展正则表达式。需安装`gawk`依赖提供支持，以下给出OpenEuler操作系统相关依赖库的安装方式：

    ```shell
    yum install -y gawk
    ```

- **模型训练过程报错：`torch has no attribute: uint64_t`**

    答：报错原因是`safetensors`版本与`PyTorch`版本不匹配，`PyTorch`版本为2.1.0，需匹配0.6.0以下的`safetensors`，使用`pip install safetensors==0.5.1`改变依赖版本即可。

- **模型训练过程报错：`AttributeError: module 'attr' has no attribute 's'`**

    答：报错原因是`attr`组件安装出错。以下给出解决方法：

    ```shell
    pip uninstall attr
    pip install attrs
    ```

- **模型训练过程中，`numpy`组件报无属性、无函数或其他类似错误**

    答：报错原因是`numpy`组件版本与模型不匹配，通常使用`pip install numpy==1.23.5`可解决，过高的numpy版本会导致代码中部分被废弃的numpy用法不可用。

- **模型安装环境时，安装`mmcv_full==1.7.2`后，安装`mmdet`、`mmdet3d`或其他`mm`相关组件时，报错`mmcv_full`组件与其要求的最大兼容版本不匹配**

    答：需按照模型README要求，应用对应patch文件，或按照环境安装步骤进行修改适配。

- **模型训练过程中，`av2`组件报错：`TypeError: Type subscription requires python >= 3.9`**

    答：报错原因是`av2`组件版本与`Python`不匹配，若使用`python==3.8`，`pip install av2==0.2.1`即可解决。

- **模型运行中出现decord或ffmpeg相关依赖报错**

    答：该问题可能是环境变量未添加，添加后若仍报错可尝试重新安装decord：

    ```shell
    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile

    # 若仍报错则重新编译decord并安装
    ```

- **模型运行时报错tcmalloc的动态库文件找不到**

    答：tcmalloc的动态库文件位置可能因环境配置会有所不同，找不到文件时可以进行搜索，一般安装在`/usr/lib64`或者`/usr/local`目录下：

    ```shell
    find /usr -name libtcmalloc.so*
    ```

    找到对应路径下的动态库文件，`libtcmalloc.so`或者`libtcmalloc.so.版本号`都可以使用。

- **安装模型环境时，需安装`nuplan-devkit`，报错`No such file or directory: 'gdal-config'`**

    答：该问题是由于操作系统未安装gmp、mpfr、OpenBLAS、sqlite3、curl、PROJ、GDAL等C++依赖库，以下给出OpenEuler操作系统相关依赖库的安装方式：

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

- **安装模型环境时，需安装`OpenEXR`，安装过程失败**

    答：该报错是由于操作系统未安装OpenEXR和OpenEXR-devel等依赖库，以下给出OpenEuler操作系统相关依赖库的安装方式：

    ```shell
    sudo yum makecache
    sudo yum install gcc gcc-c++ cmake
    sudo yum install OpenEXR
    sudo yum install OpenEXR-devel
    ```

- **安装模型环境时，需安装`omegaconf==2.1.0`组件，报错：`ERROR: Could not find a version that satisfies the requirement omegaconf==2.1.0`**

    答：该问题是由于`pip`版本过低，无法正确安装组件，需升级`pip`至最新版本。

    ```shell
    pip install --upgrade pip
    ```

- **安装`h5py`组件，报错：`ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects`**

    答：该问题推荐使用anaconda管理环境，并使用`conda install h5py`代替`pip`安装此依赖。

- **安装`loguru`组件，报错：`error: subprocess-exited-with-error pip subprocess to install build dependencies did not run successfully.`**

    答：报错原因是由于`setuptools`组件版本与`loguru`组件版本不匹配，可使用`pip install loguru==0.7.2`解决报错，更详细的报错原因和解决方法可参考[开源社区Issue：pip subprocess to install build dependencies did not run successfully](https://github.com/pypa/packaging-problems/issues/721)。

- **训练过程报错：无法找到`datasets`组件**

    答：该报错是由于环境中存在与模型目录下同名的`datasets`组件，导致模型不能找到模型目录下的`datasets`。需卸载模型环境中的同名三方库。

- **`MMCV==2.2.0 is used but incompatible`**

    答：该问题由MMCV版本冲突引起，修改`MMCV`源码文件`mmcv/version.py`中的`__version__ = '2.0.1'`即可解决。

- **x86服务器上运行模型时`import open3d`报错：`OSError: libX11.so.6: cannot open shared object file`**

    答：该问题是由于操作系统缺少X11相关依赖库（mesa-libGL/libX11/libXext），以下给出OpenEuler操作系统的安装方式：

    ```shell
    yum install -y mesa-libGL libX11 libXext
    ```

- **模型执行数据预处理脚本时报错：`ModuleNotFoundError: No module named 'tools.data_converter'`**

    答：该问题可参考BEVFormer原仓Issue进行处理：[Issues/223](https://github.com/fundamentalvision/BEVFormer/issues/223) 或 [pull/241](https://github.com/fundamentalvision/BEVFormer/pull/241)。

- **运行过程中出现`torchcodec`相关报错**

    答：可能是受到环境内系统原有ffmpeg的影响，需将原有的ffmpeg目录更名（如`mv ffmpeg ffmpeg_bak`）来避免冲突，从而确保只依赖于conda版本，随后可重新编译安装torchcodec。

- **运行过程中性能劣化明显**

    答：可能是ffmpeg的配置问题，建议检查环境内的ffmpeg是否为当前模型README推荐的方式安装所得。

- **推理过程中`triton`的CUDA版本校验失败**

    答：`torch.compile`在NPU上运行时，`torch._dynamo`会调用`_triton.py`中的函数检查CUDA设备能力，但NPU环境下CUDA相关接口返回None，导致类型比较错误。可尝试注释该校验，或者参考以下修改：

    ```python
    _cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if (
        _cap is not None
        and _cap >= (9, 0)
        and not torch.version.hip
    ):
    ```

- **torch_npu初始化遇到了“module 'numpy' has no attribute 'exceptions'..”报错**

    答：可能是numpy版本太低， 如果当前numpy是1.23.x，可以考虑升级numpy为1.26.4版本。

## 训练错误

- **模型依赖`mmcv_full==1.7.2`，训练过程报错：`File ".../torch/nn/parallel/_functions.py", line 117, in _get_stream: if device.dtype == "cpu": AttributeError: 'int' object has no attribute 'type'`**

    答：该报错可按照以下方式修改：

    ```shell
    pip show mmcv_full # 获取mmcv_full安装路径，将路径记为mmcv_install_path
    cd mmcv_install_path
    vim mmcv/parallel/_functions.py
    ```

    在文件第8行新增语句：

    ```python
    from packaging import version
    ```

    将文件第74行`streams = [_get_stream(device) for device in target_gpus]`修改为：

    ```python
    if version.parse(torch.__version__) >= version.parse('2.1.0'):
        streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
    else:
        streams = [_get_stream(device) for device in target_gpus]
    ```

    详细的报错原因见[开源社区Issue：AttributeError: 'int' object has no attribute 'type'](https://github.com/open-mmlab/mmdetection/issues/10720)。

- **模型训练过程偶现`AssertionError`，导致模型训练中断**

    答：该问题重新拉起训练即可解决，具体问题原因及解决方法可参考[开源社区Issue：Assertion Error On Finiteness](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus/issues/4)。

- **训练过程报错：`KeyError: 'road_plane'`**

    答：该报错需修改`tools/cfgs/kitti_models/pointpillar.yaml`，将文件中`USE_ROAD_PLANE`设置为`False`。

- **训练时报错：`RuntimeError: The server socket has failed to listen on any local network address`**

    答：该报错表示默认端口已被占用，自行修改`mmdetection3d`源码文件`tools/dist_train.sh`下的`PORT`默认值即可。

- **训练时报错：`RuntimeError: ACL stream synchronize failed, error code:507018`**

    答：该报错大概率是有残余进程或者其他程序在预处理数据集时占用，全局清理进程并重跑即可。

- **在各参数配置一致的情况下NPU训练结果与GPU差异较大**

    答：可能是由于EMA模型初始化过程中随机数生成器状态不一致导致，NPU对该过程进行了优化。原始逻辑中，EMA模型会先在CPU上随机初始化再被主模型预训练权重覆盖；而NPU版本优化后跳过该冗余步骤，直接加载预训练权重，加快训练启动速度。可对GPU也应用相同修改，统一随机数状态，且不影响训练效果。

- **多机训练运行出现HCCL报错且error code为1或7**

    答：优先考虑进程残留没杀干净，可以参考如下命令终止服务器上其余进程：

    ```shell
    pkill -9 python; pkill -9 torchrun
    ```

- **训练启动中卡死**

    答：若遇到训练启动卡死，需要检查是否环境中存在网络代理等，关闭代理后重试。

## 环境变量及配置错误

- **模型训练过程报错：`ImportError:/usr/local/gcc-7.5.0/lib64/libgomp.so.1:cannot allocate memory in static TLS block`**

    答：该问题由glibc版本兼容性引起，可升级glibc版本或者手动导入环境变量：

    ```shell
    export LD_PRELOAD=/usr/local/gcc-7.5.0/lib64/libgomp.so.1
    ```

- **模型训练过程报错：`ImportError: {conda_env_path}/bin/../lib/libgomp.so.1:cannot allocate memory in static TLS block`**

    答：该问题与6.1中的错误类似，可手动导入环境变量：

    ```shell
    export LD_PRELOAD={conda_env_path}/bin/../lib/libgomp.so.1:$LD_PRELOAD
    ```

- **模型训练过程报错：`ImportError: {conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`**

    答：该问题与6.1、6.2中的错误类似，可手动导入环境变量:

    ```shell
    export LD_PRELOAD={conda_env_path}/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
    ```

- **训练时报错：`Environment variable [HCCL_IF_IP] is invalid`**

    答：遇到该报错时，将环境变量设置脚本`test/env_npu.sh`中的`export HCCL_IF_IP=...`注释即可。

- **设置环境变量`TORCH_HCCL_ZERO_COPY`报错**    

    答：当前该环境变量以支持A3为主，A2暂不建议开启。

## LTO及PGO编译优化错误

- **为什么要使用编译优化，能否不使用编译优化？**

    答：编译优化技术在数据库、分布式存储等数据和计算密集型前端瓶颈较高的场景效果显著，性能可得到显著的提升。通过毕昇编译器对源码构建编译Python、PyTorch、TorchNPU三个组件，可以有效提升模型性能。如果不需要追求编译优化后的更高模型性能，那么可以不使用编译优化。

- **编译优化`TorchNPU`时，报错：`ImportError: .../torch_npu/lib/libtorch_npu.so: undefined symbol`**

    答：该问题是由于编译优化对于GCC等编译依赖的版本要求较高，推荐使用PyTorch和TorchNPU编译优化专有镜像编译，具体镜像使用和编译优化步骤请参考《TorchNPU框架特性》的[编译优化](https://www.hiascend.com/document/detail/zh/Pytorch/2610/devguide/fwfeatures/docs/zh/framework_feature_guide_pytorch/comp_opt_intro.md)章节。

## 其他问题

- **模型所需的预训练权重文件因网络问题下载失败如何解决？**

    答：预训练权重文件下载失败，可以根据报错链接，手动下载，拷贝到用户名对应目录：

    ```shell
    wget $ckpt_url # 将预训练权重文件的链接记为$ckpt_url
    cp $ckpt_file $HOME/.cache/torch/hub/checkpoints/$pretrain_ckpt # 将下载好的权重文件名称记为$ckpt_file, 将用户根目录记为$HOME, 将原下载失败的文件名称记为$pretrain_ckpt
    ```

- **模型训练配置是否可以自行更改？**

    答：推荐按照模型README文件中提供的训练配置进行模型训练。

- **训练不同模型时，必须为每个模型新建一个环境吗？可以使用同一套环境管理所有模型吗？**

    答：不建议使用同一套环境管理所有模型。每个模型所使用的组件和依赖版本不完全相同，且部分模型应用tcmalloc高性能内存库、编译优化技术提升模型的性能，若使用同一套环境，可能影响未应用tcmalloc高性能内存库和编译优化技术的模型性能。

- **无网络或设有防火墙的环境下如何下载预训练权重？**

    答：无网络情况下，用户可以自行下载预训练权重文件，将下载好的权重拷贝至对应目录。例如下载`resnet50`预训练权重：

    ```shell
    # 手动下载预训练权重文件
    wget https://download.pytorch.org/models/resnet50-0676ba61.pth
    # 将预训练权重拷贝至TorchHub缓存目录
    cp resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    ```

    若使用`SwinTransformer`预训练权重，可自行下载[swin_base_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)，拷贝至`~/.cache/torch/hub/checkpoints/`目录。

- **在无法访问HuggingFace hub的情况下运行模型报错**

    答：用户可以使用HuggingFace镜像源或ModelScope在有网络的情况下自主下载模型与数据集，按照模型README中的文件结构组织文件即可。下载后设置环境变量：

    ```shell
    export HF_HOME="/{path_to_caches}/caches/"
    export HUGGINGFACE_HUB_CACHE="/{path_to_caches}/caches/"
    ```

- **在无网络环境下无法自动下载Paligemma权重**

    答：可自行下载[paligemma权重](https://huggingface.co/google/paligemma-3b-pt-224)，将权重路径记作`paligemma_weights`，再执行以下命令使用脚本将本地权重路径替换进模型代码：

    ```shell
    bash test/paligemma_weights_mod.sh ${paligemma_weights}
    ```
