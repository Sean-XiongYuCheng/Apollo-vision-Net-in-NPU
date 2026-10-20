
# Driving SDK Security Statement

## System Security Hardening

1. When configuring the system, you are advised to enable **address space layout randomization** (ASLR) level 2 to improve system security. Run the following command to enable it:

    ```shell
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

2. Because you need to build Driving SDK yourself, you are advised to run `strip` on the generated `.so` files. `strip` is also known as **removing debug symbols**. Use the following command:

    ```shell
    strip -s <so_file>
    ```

   The relevant `.so` files are as follows:
   - `mx_driving/packages/vendors/customize/op_api/lib/libcust_opapi.so`
   - `mx_driving/packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so`
   - `mx_driving/packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so`

## Recommended Runtime User

For security and least-privilege considerations, you are advised not to use `root` or other administrator accounts to run Driving SDK.

## File Permission Control

When using Driving SDK, you may perform profiling, debugging, and other operations. You are advised to apply permission controls to the relevant directories and files to ensure file security.

1. When using Driving SDK, you are advised to set `umask` to `0027` or a more restrictive value, so that newly created directories have maximum default permissions of `750` and newly created files have maximum default permissions of `640`.
2. You are advised to properly control permissions for sensitive content such as personal data, commercial assets, source files, and all types of files saved during training. See the table below for secure permission settings.

### File Permission Reference

| Type                              | Maximum Linux Permission|
| ---------------------------------- | ------------------- |
| Home directory                        | 750 (rwxr-x---)   |
| Program files (including scripts and libraries)    | 550 (r-xr-x---)   |
| Program file directory                      | 550 (r-xr-x---)   |
| Configuration files                          | 640 (rw-r-----)   |
| Configuration file directory                      | 750 (rwxr-x---)   |
| Log files (recorded or archived)    | 440 (r--r-----)   |
| Log files (being recorded)                | 640 (rw-r-----)   |
| Log file directory                      | 750 (rwxr-x---)   |
| Debug files                         | 640 (rw-r-----)   |
| Debug file directory                     | 750 (rwxr-x---)   |
| Temporary file directory                      | 750 (rwxr-x---)   |
| Maintenance and upgrade file directory                  | 770 (rwxrwx---)   |
| Service data files                      | 640 (rw-r-----)   |
| Service data file directory                  | 750 (rwxr-x---)   |
| Key components, private keys, certificates, and ciphertext file directory| 700 (rwx------)   |
| Key components, private keys, certificates, and ciphertext files    | 600 (rw-------)   |
| APIs and scripts for encryption and decryption            | 500 (r-x------)   |

## Build Security Statement

When you compile and install Driving SDK from source, the build process generates intermediate files. You are advised to apply permission controls to these intermediate files after the build is complete to ensure file security.

## Runtime Security Statement

1. You are advised to write training scripts based on the available environment resources. If the training script does not match the available resources, such as dataset load that exceeds available memory or a training script that generates data locally beyond drive capacity, errors may occur and the process may exit unexpectedly.
2. If Driving SDK runs abnormally, such as due to input validation errors (see the API documentation), environment variable configuration errors, or operator execution failures, the process will exit and print an error message. This is expected. You are advised to locate the specific cause based on the error message, for example by enabling synchronous operator execution, checking CANN logs, or parsing generated core dump files.

## Public Network Address Statement

The Driving SDK code contains a public address statement as shown in the table below:

### Public Network Addresses

| Type        | Open-Source Code Address                                                           | File                                                         | Public IP Address / Public URL / Domain / Email Address                                             | Purpose                                                   |
| ----------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| In-house    | N/A                                                                                | ci/docker/ARM/Dockerfile                                     | <https://mirrors.aliyun.com/pypi/simple>                                                            | Docker configuration file used to configure the pip index |
| In-house    | N/A                                                                                | ci/docker/X86/Dockerfile                                     | <https://mirrors.aliyun.com/pypi/simple>                                                            | Docker configuration file used to configure the pip index |
| In-house    | N/A                                                                                | ci/docker/ARM/install_cann.sh                                | <https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN>                                         | CANN package download address                             |
| In-house    | N/A                                                                                | ci/docker/x86/install_cann.sh                                | <https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN>                                         | CANN package download address                             |
| In-house    | N/A                                                                                | ci/docker/ARM/install_obs.sh                                 | <https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz> | OBS download link                                         |
| In-house    | N/A                                                                                | ci/docker/X86/install_obs.sh                                 | <https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz> | OBS download link                                         |
| Open-source | <https://gitee.com/it-monkey/protocolbuffers.git>                                  | ci/docker/ARM/build_protobuf.sh                              | <https://gitee.com/it-monkey/protocolbuffers.git>                                                   | Used to build protobuf                                    |
| Open-source | <https://gitee.com/it-monkey/protocolbuffers.git>                                  | ci/docker/X86/build_protobuf.sh                              | <https://gitee.com/it-monkey/protocolbuffers.git>                                                   | Used to build protobuf                                    |
| Open-source | `https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip`              | model_examples/CenterNet/CenterNet.patch                     | `https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip`                               | Download link for invalid source model data               |
| Open-source | <https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip> | model_examples/CenterNet/CenterNet.patch                     | <https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip>                  | Download link for required model data                     |
| Open-source | <https://download.pytorch.org/whl/torch_stable.html>                               | model_examples/Diffusion-Planner/diffusionPlanner.patch      | <https://download.pytorch.org/whl/torch_stable.html>                                                | Model dependency package download                         |
| Open-source | <https://pypi.tuna.tsinghua.edu.cn/simple>                                         | model_examples/Diffusion-Planner/diffusionPlanner.patch      | <https://pypi.tuna.tsinghua.edu.cn/simple>                                                          | Model dependency package download                         |
| Open-source | <https://github.com/Pointcept/Pointcept>                                           | model_examples/PointTransformerV3/Ptv3.patch                 | <xiaoyang.wu.cs@gmail.com>                                                                          | Model author email declaration                            |
| Open-source | <https://www.argoverse.org/av2.html>                                               | model_examples/QCNet/patch/qcnet.patch                       | <https://www.argoverse.org/av2.html>                                                                | Model dataset introduction                                |
| Open-source | <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0>                 | model_examples/Cosmos-Predict2/cosmos_predict2.patch         | <https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0>                                  | Model dependency package download                         |
| Open-source | <https://github.com/yifanlu0227/viser>                                             | model_examples/Cosmos-Drive-Dreams/Cosmos-Drive-Dreams.patch | <https://github.com/yifanlu0227/viser>                                                              | Model dependency package download                         |
| Open-source | <https://github.com/vllm-project/vllm-ascend/pull/1482>                            | model_examples/Cosmos-Reason1/cosmos-rl.patch                | <https://github.com/vllm-project/vllm-ascend/pull/1482>                                             | Model dependency package download                         |
| Open-source | <https://github.com/huggingface/transformers/pull/38366>                           | model_examples/Cosmos-Reason1/cosmos-rl.patch                | <https://github.com/huggingface/transformers/pull/38366>                                            | Model dependency package download                         |

## Public Interface Statement

Refer to the [API List](./docs/en/api/README.md). Driving SDK provides custom interfaces that are exposed externally. If a function appears in the documentation, that interface is public. Otherwise, before using the function, you can consult the community on whether it is truly public or has been unintentionally exposed, because interfaces that are not exposed may be changed or removed in the future.

## Communication Security Hardening

Driving SDK depends on PyTorch and TorchNPU at runtime. You should pay attention to communication security hardening. For details, see [TorchNPU Communication Security Hardening](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.en.md#communication-security-hardening).

## Communication Matrix

Driving SDK depends on PyTorch and TorchNPU at runtime and involves a communication matrix. For details, see [TorchNPU Communication Matrix](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.en.md#communication-matrix).

## Vulnerability Response

We place high importance on the security of the community version. The Mind open-source community receives, investigates, and discloses security vulnerabilities related to this community. <!--For details, see [Ascend Vulnerability Response](https://gitcode.com/Ascend/community/blob/master/docs/security.md).-->
