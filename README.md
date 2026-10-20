# Apollo-Vision-Net 昇腾 NPU FP32 交付仓

本仓库是 Apollo-Vision-Net 在昇腾 NPU 上完成迁移、适配和真实训练验证后的 FP32 统一交付版本。

仓库包含 Apollo 主工程、昇腾适配后的 OpenMMLab 依赖、参考源码、预训练权重、Docker 交付文件、适配指导书和训练验证脚本。

> 当前发布范围仅包括已经验证的 FP32 路线，不包含 FP16 实验改动。

## 已完成验证

当前版本已经完成以下验证：

- `torch_npu` 可正常识别并使用昇腾 NPU；
- MMCV 的 NPU 二进制扩展可正常加载；
- Apollo BEVFormer 检测与占用模型可正常构建；
- 单卡 FP32 真实训练第一步通过；
- 八卡 FP32 分布式训练入口可用；
- nuScenes 数据集可在 `torch_npu` 多进程 DataLoader 下序列化；
- 无速度检测框可以正常执行 nuScenes 评估转换；
- Docker 镜像构建成功；
- 三份预训练权重通过 SHA256 校验；
- 权重已经使用 Git LFS 上传。

已验证的真实训练日志包含：

```text
Epoch [1][1/28130]
memory: 10447
loss: 46.6250
grad_norm: 34.1578
```

## 仓库结构

```text
Apollo-Vision-Net/          Apollo主工程及预训练权重
DrivingSDK/                 昇腾DrivingSDK源码
mmcv/                       Apollo使用的MMCV昇腾适配版本
mmdetection_2.19/           MMDetection 2.19适配版本
mmsegmentation_0.20/        MMSegmentation 0.20适配版本
mmdetection3d_0.18.1/       MMDetection3D 0.18.1适配版本

reference/                  迁移过程中使用的参考源码
extras/                     校验脚本、辅助脚本和训练看板
docs/                       NPU迁移、适配和Docker交付指导书
scripts/                    正式环境检查、冒烟测试和八卡训练脚本
manifests/                  源码版本及Python环境清单
docker/                     Dockerfile、容器入口和启动脚本
```

`reference/` 目录用于迁移对照，不是 Apollo FP32 运行时依赖。

## 核心版本

| 组件 | 版本或提交 |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post6 |
| mx_driving | 26.1.0.post2 |
| MMCV | 1.7.2 昇腾适配版 |
| MMDetection | 2.19.0 |
| MMSegmentation | 0.20.0 |
| MMDetection3D | 0.18.1 |
| Apollo-Vision-Net | `540ad5e` |
| DrivingSDK | `ad53cd15` |
| MMCV源码 | `ac5854e8` |

完整版本见：

```text
COMPONENT_VERSIONS.md
manifests/source_versions.txt
manifests/pip_freeze.txt
manifests/conda_explicit.txt
```

## 克隆仓库

预训练权重使用 Git LFS 管理，因此必须先安装 Git LFS：

```bash
git lfs install

git clone \
  https://github.com/Sean-XiongYuCheng/Apollo-vision-Net-in-NPU.git

cd Apollo-vision-Net-in-NPU

git lfs pull
```

不要只下载 GitHub 的 Source code ZIP，因为 ZIP 不保证包含完整的 Git LFS 权重对象。

## 校验预训练权重

仓库包含三份预训练权重：

```text
Apollo-Vision-Net/ckpts/r101_dcn_fcos3d_pretrain.pth
Apollo-Vision-Net/ckpts/resnet50-0676ba61.pth
Apollo-Vision-Net/ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth
```

执行校验：

```bash
sha256sum -c WEIGHTS.sha256
```

三行全部显示 `OK`，表示权重下载完整。

## 数据集

数据集约 547GB，不提交到 Git 仓库，也不写入 Docker 镜像。

宿主机应单独准备数据目录，并挂载到容器内：

```text
/opt/apollo/workspace/Apollo-Vision-Net/data
```

例如：

```bash
export APOLLO_DATA_DIR=/data/apollo-data
export APOLLO_OUTPUT_DIR=/data/apollo-output
```

训练输出应挂载到：

```text
/opt/apollo/workspace/Apollo-Vision-Net/work_dirs
```

这样删除容器不会删除数据集和训练结果。

## Docker环境

基础镜像：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10
```

本地已经验证的镜像：

```text
apollo-npu-fp32:v1
```

启动前设置：

```bash
export APOLLO_IMAGE=apollo-npu-fp32:v1
export APOLLO_DATA_DIR=/data/apollo-data
export APOLLO_OUTPUT_DIR=/data/apollo-output

bash docker/run.sh
```

`docker/run.sh` 会配置：

- Ascend 驱动和固件挂载；
- `--privileged`；
- host 网络；
- host IPC；
- 数据集只读挂载；
- 训练输出持久化挂载。

### 关于Docker环境压缩包

Dockerfile 使用已经验证的 Conda 环境压缩包：

```text
docker/context/env2_apollo.tar.gz
```

该压缩包体积较大，不提交到 Git。

其他服务器应优先拉取后续发布的完整 Docker 镜像。需要从 Dockerfile 重建时，必须单独取得该环境压缩包，并放入上述目录。

## 环境检查

进入容器后执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
bash /opt/apollo/workspace/scripts/check_environment.sh
```

成功标志：

```text
APOLLO_NPU_ENV_CHECK=PASS
```

该脚本会检查：

- Python和核心模块版本；
- `torch_npu`；
- MMCV NPU扩展；
- NPU可用状态；
- NPU张量计算结果。

## 单卡一步训练验证

使用 0 号卡执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
bash /opt/apollo/workspace/scripts/smoke_train_fp32.sh
```

脚本会完成第一个 iteration 后主动停止。

成功标志：

```text
APOLLO_FP32_SMOKE_TRAIN=PASS
```

## 八卡FP32正式训练

默认使用 0–7 号卡：

```bash
bash /opt/apollo/workspace/scripts/train_8p_fp32.sh
```

使用 8–15 号卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
RUN_DIR=work_dirs/apollo_dla34_fp32_8p \
PORT=29511 \
bash /opt/apollo/workspace/scripts/train_8p_fp32.sh
```

默认参数：

```text
训练进程数=8
单卡batch=1
DataLoader workers/卡=4
OMP_NUM_THREADS=1
```

这些参数都可以通过同名环境变量覆盖。

## 主要文档

- `docs/Apollo-Vision-Net_昇腾NPU适配指导书.md`
- `docs/Apollo-Vision-Net_昇腾NPU迁移理解.md`
- `docs/Apollo-Vision-Net_FP32代码仓与Docker镜像交付指导书.md`
- `docker/VALIDATION.md`
- `COMPONENT_VERSIONS.md`
- `WEIGHTS.sha256`

## 正式脚本

| 脚本 | 用途 |
|---|---|
| `scripts/check_environment.sh` | 检查Python、NPU和MMCV扩展 |
| `scripts/smoke_train_fp32.sh` | 单卡执行一个真实训练 iteration |
| `scripts/train_8p_fp32.sh` | 八卡FP32正式训练 |

## 不纳入Git的内容

以下内容不会提交：

- 547GB数据集；
- 训练输出 `work_dirs/`；
- profiler 原始数据；
- Python字节码缓存；
- 本地备份文件；
- Docker环境压缩包；
- FP16实验代码和日志。

## GitHub仓库

```text
https://github.com/Sean-XiongYuCheng/Apollo-vision-Net-in-NPU
```
