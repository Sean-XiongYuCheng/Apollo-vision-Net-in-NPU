# Apollo-Vision-Net FP32 代码仓与 Docker 镜像交付指导书

版本：v1.0 · 代码仓＋运行镜像方案 · 2026-10-20

本文只交付当前已经跑通的 **Apollo DLA-34 检测＋Occupancy FP32 路线**。不包含 FP16 适配，不使用 FP16 配置。

交付结果固定为两部分：

1. 一个代码仓库：`apollo-npu-workspace`。
2. 一个运行镜像：`apollo-npu-fp32:<版本>`。

目标服务器的使用方式固定为：

```bash
git clone <代码仓地址> apollo-npu-workspace
docker pull <镜像仓地址>/apollo-npu-fp32:<版本>
docker run ... <镜像仓地址>/apollo-npu-fp32:<版本>
```

镜像内部已经安装并固定原 `/data/apollo-npu-workspace/env2_apollo` 的 Python 环境及编译依赖。目标服务器不再创建、复制或激活 `/data/apollo-npu-workspace/env2_apollo`。

## 1. 交付边界

### 1.1 代码仓包含什么

新代码仓名称固定为：

```text
apollo-npu-workspace
```

结构如下：

```text
apollo-npu-workspace/
├── Apollo-Vision-Net/          FP32 模型、配置、训练和测试入口
├── DrivingSDK/                 NPU 算子源码和对应说明
├── mmcv/                       已应用 NPU 适配的 MMCV 源码
├── mmdetection_2.19/           当前运行版本
├── mmsegmentation_0.20/        当前运行版本
├── mmdetection3d_0.18.1/       当前运行版本及本项目适配
├── reference/                  仅用于对照、不安装的其他源码
│   ├── BEVFormer/
│   ├── detectron2/
│   ├── mmdetection_2.24/
│   └── mmdetection3d_1.0rc4/
├── extras/                     原工作根中的辅助代码，不进入运行环境
│   ├── apollo_training_dashboard/
│   └── jiaoyan/
├── docs/
│   ├── Apollo-Vision-Net_昇腾NPU适配指导书.md
│   └── 本指导书
├── docker/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── artifacts/              本地构建材料，不提交 Git
├── scripts/
│   ├── check_environment.sh
│   ├── smoke_train_fp32.sh
│   └── train_8p_fp32.sh
├── manifests/
│   ├── source_versions.txt
│   ├── pip_freeze.txt
│   ├── runtime_modules.txt
│   └── data_and_weights.md
├── .dockerignore
├── .gitignore
└── README.md
```

### 1.2 代码仓不包含什么

以下内容不提交 Git：

```text
env2_apollo/
Apollo-Vision-Net/data/
Apollo-Vision-Net/work_dirs/
Apollo-Vision-Net/test/
Apollo-Vision-Net/ckpts/*.pth
profiler 输出
训练日志
Conda 环境压缩包
```

Conda 环境只用于构建 Docker 镜像；数据和初始权重通过授权文件存储单独提供。

### 1.3 Docker 镜像包含什么

镜像包含：

- DrivingSDK A3 基础镜像中的 CANN 用户态组件、torch 2.7.1、torch_npu 2.7.1.post6 和 mx_driving。
- 从当前 `env2_apollo` 打包并迁入镜像的 Python 环境。
- 从 `apollo-npu-workspace` 固定提交构建的 MMCV、mmdet、mmseg 和 mmdet3d。
- 同一提交的 Apollo 源码、配置和启动脚本。
- 自动设置好的 PATH、PYTHONPATH 和工作目录。

镜像不包含宿主机驱动、固件、训练数据和未经授权的权重。目标机器仍需满足原《Apollo-Vision-Net 昇腾 NPU适配指导书》第 2、4 章的硬件、驱动和容器运行条件。

## 2. 先固定 FP32 源码基线

### 2.1 选择 Apollo FP32 提交

在源服务器执行：

```bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
git status --short
git log --oneline --graph --decorate -20
```

第一行查看未提交修改；第二行查看最近提交。选择最后一个已经验证 FP32 训练可运行、尚未包含 FP16 适配的提交，将其编号记为 `FP32_COMMIT`。

检查该提交：

```bash
git show --stat FP32_COMMIT
git show FP32_COMMIT:projects/configs/bevformer/bev_tiny_det_occ_apollo.py | head -50
```

把 `FP32_COMMIT` 替换为实际提交号。确认配置使用普通 `BEVFormer`、普通 `EpochBasedRunner`，没有启用 FP16。

创建标签：

```bash
git tag -a apollo-npu-fp32-v1 FP32_COMMIT \
  -m "Apollo DLA34 detection+occupancy FP32 verified baseline"
```

该标签是新代码仓中 Apollo 目录的唯一来源。后续 FP16 提交可以继续留在原仓历史中，但不会进入新仓工作树。

### 2.2 固定其余运行仓库

以下五个仓库必须把运行所需修改提交到各自 Git 历史：

```text
/data/apollo-npu-workspace/DrivingSDK
/data/apollo-npu-workspace/mmcv
/data/apollo-npu-workspace/mmdetection_2.19
/data/apollo-npu-workspace/mmsegmentation_0.20
/data/apollo-npu-workspace/mmdetection3d_0.18.1
```

逐个执行：

```bash
git status --short
git rev-parse HEAD
git log -1 --oneline
```

`status` 非空时，先区分源码修改与缓存、编译输出。运行所需源码修改必须形成提交；新代码仓通过 `git archive` 导出已提交内容，不会自动携带未提交文件。

## 3. 创建统一代码仓 apollo-npu-workspace

### 3.1 创建空目录

在源服务器执行：

```bash
export SOURCE_ROOT=/data/apollo-npu-workspace
export WORKSPACE_REPO=/data/apollo-npu-workspace/apollo-npu-workspace
test ! -e "$WORKSPACE_REPO"
mkdir -p "$WORKSPACE_REPO"
git -C "$WORKSPACE_REPO" init -b main
```

- `SOURCE_ROOT` 指向现有全量工作区。
- `WORKSPACE_REPO` 在源服务器暂存于现有容器已挂载的目录中；真正的仓库名称仍是 `apollo-npu-workspace`，上传后不依赖这个源路径。
- `test ! -e` 防止覆盖已有目录；失败时换一个暂存位置，不删除不确定的目录。
- 最后一行初始化 `main` 分支。

### 3.2 导出 Apollo FP32 代码

```bash
mkdir -p "$WORKSPACE_REPO/Apollo-Vision-Net"
git -C "$SOURCE_ROOT/Apollo-Vision-Net" \
  archive apollo-npu-fp32-v1 \
  | tar -x -C "$WORKSPACE_REPO/Apollo-Vision-Net"
```

`git archive` 只导出 FP32 标签所对应的已提交文件；管道右侧将内容展开到新仓的 Apollo 目录。原仓中后续 FP16 文件和修改不会进入这里。

### 3.3 导出五个运行依赖仓库

```bash
for repo in \
  DrivingSDK \
  mmcv \
  mmdetection_2.19 \
  mmsegmentation_0.20 \
  mmdetection3d_0.18.1; do
  mkdir -p "$WORKSPACE_REPO/$repo"
  git -C "$SOURCE_ROOT/$repo" archive HEAD \
    | tar -x -C "$WORKSPACE_REPO/$repo"
done
```

循环逐个导出当前已提交版本。新仓中没有嵌套 `.git`，六套源码由外层 `apollo-npu-workspace` 统一管理。

### 3.4 导出对照源码

对照代码放入 `reference`，不会安装到运行环境：

```bash
for repo in \
  BEVFormer \
  detectron2 \
  mmdetection_2.24 \
  mmdetection3d_1.0rc4; do
  mkdir -p "$WORKSPACE_REPO/reference/$repo"
  git -C "$SOURCE_ROOT/$repo" archive HEAD \
    | tar -x -C "$WORKSPACE_REPO/reference/$repo"
done
```

如果某个目录不是 Git 仓库，不要改用整目录盲拷贝；先确认它是否需要交付以及如何记录版本。

### 3.5 核对原工作根中的其余代码

先生成一级目录清单：

```bash
find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
```

将每一项明确归入以下一类：运行源码、`reference` 对照源码、`extras` 辅助代码、Conda 环境、数据/权重、训练输出或缓存。不能因为本指导书没有预先列出某个目录就静默遗漏。

本地工作区中还存在 `apollo_training_dashboard` 和 `jiaoyan`。若服务器也有且需要随全量代码交付，优先从它们各自的 Git 提交导出；若它们不是 Git 仓库，在确认只含源码和展示材料后执行：

```bash
mkdir -p "$WORKSPACE_REPO/extras"
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='node_modules/' \
  --exclude='dist/' \
  "$SOURCE_ROOT/apollo_training_dashboard/" \
  "$WORKSPACE_REPO/extras/apollo_training_dashboard/"
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$SOURCE_ROOT/jiaoyan/" \
  "$WORKSPACE_REPO/extras/jiaoyan/"
```

这些目录不会由 Dockerfile 安装。若服务器不存在其中某项，跳过对应命令并在 README 记录；不要创建空目录假装完成迁移。`env2_apollo` 明确属于镜像构建材料，不进入代码仓。

### 3.6 复制 FP32 指导书

```bash
mkdir -p "$WORKSPACE_REPO/docs"
cp "$SOURCE_ROOT/apollo_comment/Apollo-Vision-Net_昇腾NPU适配指导书.md" \
   "$WORKSPACE_REPO/docs/"
cp "$SOURCE_ROOT/apollo_comment/Apollo-Vision-Net_FP32代码仓与Docker镜像交付指导书.md" \
   "$WORKSPACE_REPO/docs/"
```

只复制 FP32 主指导书和本交付指导书，不复制 FP16 指导书。

## 4. 记录新仓的来源

### 4.1 源码版本清单

```bash
mkdir -p "$WORKSPACE_REPO/manifests"
{
  printf '[Apollo-Vision-Net]\n'
  git -C "$SOURCE_ROOT/Apollo-Vision-Net" rev-parse 'apollo-npu-fp32-v1^{}'
  for repo in DrivingSDK mmcv mmdetection_2.19 \
              mmsegmentation_0.20 mmdetection3d_0.18.1; do
    printf '\n[%s]\n' "$repo"
    git -C "$SOURCE_ROOT/$repo" rev-parse HEAD
  done
} > "$WORKSPACE_REPO/manifests/source_versions.txt"
```

`^{}` 将带注释标签解析为真正的 Apollo commit。清单记录新单仓库内每个目录原来对应的上游提交。

### 4.2 环境清单

进入当前跑通容器：

```bash
docker exec -it apollo_dev bash
conda activate /data/apollo-npu-workspace/env2_apollo
```

执行：

```bash
python -m pip freeze \
  > /data/apollo-npu-workspace/apollo-npu-workspace/manifests/pip_freeze.txt

python - <<'PY' \
  > /data/apollo-npu-workspace/apollo-npu-workspace/manifests/runtime_modules.txt
import torch
import torch_npu
import mmcv
import mmdet
import mmseg
import mmdet3d
import mx_driving

for module in [torch, torch_npu, mmcv, mmdet, mmseg, mmdet3d, mx_driving]:
    print(module.__name__, getattr(module, '__version__', 'unknown'),
          getattr(module, '__file__', 'unknown'))
print('npu_available', torch.npu.is_available())
print('npu_count', torch.npu.device_count())
PY
```

这些文件用于审计镜像内容。`runtime_modules.txt` 中出现旧 `/data/apollo-npu-workspace` 路径是源环境事实；正式镜像构建后需要再次生成镜像内清单，路径应指向 `/opt/apollo`。

## 5. 配置 Git 忽略规则

新建 `$WORKSPACE_REPO/.gitignore`：

```gitignore
# Dataset and runtime outputs
Apollo-Vision-Net/data/
Apollo-Vision-Net/work_dirs/
Apollo-Vision-Net/test/
Apollo-Vision-Net/msprof/

# Model artifacts
Apollo-Vision-Net/ckpts/*.pth
*.pth
*.onnx

# Python/build caches
**/__pycache__/
*.py[cod]
*.so
*.egg-info/
build/
dist/

# Local Docker build artifact
docker/artifacts/*
!docker/artifacts/.gitkeep

# Logs and editor files
*.log
.vscode/
.idea/
```

说明：`.so` 不提交 Git；MMCV NPU 扩展在构建 Docker 镜像时由源码重新编译。Conda 环境压缩包只作为本机构建输入，不进入 Git。

新建 `$WORKSPACE_REPO/.dockerignore`：

```dockerignore
.git
Apollo-Vision-Net/data
Apollo-Vision-Net/work_dirs
Apollo-Vision-Net/test
reference
docs
*.log
```

这里没有排除 `docker/artifacts/env2_apollo.tar.gz`，因为 Dockerfile 构建镜像时需要读取它。该文件由 `.gitignore` 排除，因此不会进入代码仓提交。

## 6. 将当前 env2_apollo 做成镜像构建材料

### 6.1 使用 conda-pack

在当前跑通容器里检查：

```bash
conda pack --help | head
```

如果命令不存在，在基础 Conda 环境或专门的打包环境安装 `conda-pack`；不要为此升级 `env2_apollo` 的训练依赖。

创建构建材料目录：

```bash
mkdir -p /data/apollo-npu-workspace/apollo-npu-workspace/docker/artifacts
```

打包当前环境：

```bash
conda pack \
  -p /data/apollo-npu-workspace/env2_apollo \
  -o /data/apollo-npu-workspace/apollo-npu-workspace/docker/artifacts/env2_apollo.tar.gz \
  --ignore-editable-packages
```

- `-p` 指定当前已跑通环境。
- `-o` 生成镜像构建材料。
- `--ignore-editable-packages` 允许源环境中存在指向本地源码的 editable 包。

editable 包不会直接沿用旧路径；Dockerfile 会从新代码仓源码重新安装。若 conda-pack 报包文件缺失，不要使用 `--ignore-missing-files` 直接掩盖，应先查明 pip/conda 文件冲突。

## 7. 编写运行镜像

### 7.1 Dockerfile

新建 `$WORKSPACE_REPO/docker/Dockerfile`：

```dockerfile
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10

ARG APOLLO_WORKSPACE_COMMIT=unknown

LABEL org.opencontainers.image.title="Apollo Vision Net FP32 on Ascend"
LABEL org.opencontainers.image.revision="${APOLLO_WORKSPACE_COMMIT}"

COPY docker/artifacts/env2_apollo.tar.gz /tmp/env2_apollo.tar.gz

RUN mkdir -p /opt/apollo/env && \
    tar -xzf /tmp/env2_apollo.tar.gz -C /opt/apollo/env && \
    /opt/apollo/env/bin/conda-unpack && \
    rm -f /tmp/env2_apollo.tar.gz

ENV PATH=/opt/apollo/env/bin:${PATH}
ENV PYTHONNOUSERSITE=1

COPY Apollo-Vision-Net /opt/apollo/workspace/Apollo-Vision-Net
COPY DrivingSDK /opt/apollo/workspace/DrivingSDK
COPY mmcv /opt/apollo/workspace/mmcv
COPY mmdetection_2.19 /opt/apollo/workspace/mmdetection_2.19
COPY mmsegmentation_0.20 /opt/apollo/workspace/mmsegmentation_0.20
COPY mmdetection3d_0.18.1 /opt/apollo/workspace/mmdetection3d_0.18.1
COPY manifests /opt/apollo/workspace/manifests
COPY scripts /opt/apollo/workspace/scripts
COPY docker/entrypoint.sh /opt/apollo/entrypoint.sh

RUN cd /opt/apollo/workspace/mmcv && \
    MMCV_WITH_OPS=1 MAX_JOBS=8 \
    python -m pip install --no-deps --no-build-isolation --force-reinstall .

RUN python -m pip install --no-deps --no-build-isolation --force-reinstall \
      /opt/apollo/workspace/mmdetection_2.19 && \
    python -m pip install --no-deps --no-build-isolation --force-reinstall \
      /opt/apollo/workspace/mmsegmentation_0.20 && \
    python -m pip install --no-deps --no-build-isolation --force-reinstall \
      /opt/apollo/workspace/mmdetection3d_0.18.1

RUN ! grep -R '/data/apollo-npu-workspace' \
      /opt/apollo/env/lib/python3.10/site-packages/*.pth && \
    python -c "import mmcv,mmdet,mmseg,mmdet3d; \
assert all('/data/apollo-npu-workspace' not in m.__file__ \
for m in [mmcv,mmdet,mmseg,mmdet3d])"

RUN chmod +x /opt/apollo/entrypoint.sh \
             /opt/apollo/workspace/scripts/*.sh

WORKDIR /opt/apollo/workspace/Apollo-Vision-Net
ENTRYPOINT ["/opt/apollo/entrypoint.sh"]
CMD ["bash"]
```

关键点：

- 环境统一安装到镜像内部 `/opt/apollo/env`，不依赖目标服务器用户名路径。
- 六套运行源码统一放入 `/opt/apollo/workspace`。
- MMCV 在镜像构建阶段重新编译 NPU 扩展，不使用原工作区的未跟踪 `.so`。
- mmdet、mmseg、mmdet3d 使用普通非 editable 安装，运行时不再引用旧 `/data/apollo-npu-workspace`。
- 构建阶段检查 `.pth` 和核心包实际路径；仍残留旧 editable 路径时直接构建失败，不发布带隐患的镜像。
- Apollo 本身从固定工作目录加载 `projects` 插件。

### 7.2 entrypoint.sh

新建 `$WORKSPACE_REPO/docker/entrypoint.sh`：

```bash
#!/usr/bin/env bash
set -e

export PATH=/opt/apollo/env/bin:${PATH}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/apollo/workspace/Apollo-Vision-Net:${PYTHONPATH:-}

cd /opt/apollo/workspace/Apollo-Vision-Net
exec "$@"
```

逐行含义：遇到启动错误立即退出；固定使用镜像内 Python；禁用宿主机用户 site-packages；让 Apollo 的 `projects` 包可导入；进入 Apollo 目录；最后执行 `docker run` 传入的命令。

## 8. 编写检查和训练脚本

### 8.1 check_environment.sh

新建 `$WORKSPACE_REPO/scripts/check_environment.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import mmcv, mmdet, mmseg, mmdet3d, mx_driving
from mmcv.ops import MultiScaleDeformableAttention

for module in [torch, torch_npu, mmcv, mmdet, mmseg, mmdet3d, mx_driving]:
    print(module.__name__, getattr(module, '__version__', 'unknown'),
          getattr(module, '__file__', 'unknown'))

assert torch.npu.is_available()
x = torch.ones(2, 2).npu()
print((x + x).cpu())
print(MultiScaleDeformableAttention)
PY
```

该脚本检查核心包来源、NPU 张量计算和 MMCV 扩展导入。输出全 2 矩阵才说明基本计算成功。

### 8.2 smoke_train_fp32.sh

新建 `$WORKSPACE_REPO/scripts/smoke_train_fp32.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${ASCEND_RT_VISIBLE_DEVICES:?请先指定一张空闲 NPU}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

RUN_DIR=${RUN_DIR:-/outputs/fp32_smoke}
mkdir -p "$RUN_DIR"

PORT=${PORT:-29531} bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
  1 \
  --work-dir "$RUN_DIR" \
  --seed 0 \
  --no-validate \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=4 \
    log_config.interval=1 \
    checkpoint_config.interval=1 \
  2>&1 | tee "$RUN_DIR/train.log"
```

该脚本只使用普通 FP32 配置。它最多运行一轮；短检查可在连续若干次正常迭代后在前台按一次 Ctrl+C。要验证 checkpoint 保存与恢复时，需要让独立测试完整完成一轮。

正式 8 卡脚本以原指导书第 9.4 节命令为准，保存为 `scripts/train_8p_fp32.sh`，不得引用 FP16 入口或配置。

## 9. 提交并上传统一代码仓

先检查：

```bash
cd /data/apollo-npu-workspace/apollo-npu-workspace
git status --short
git diff --check
```

然后只在确认内容正确后提交：

```bash
git add --all
git commit \
  -m "release: package Apollo FP32 NPU workspace" \
  -m "Consolidate the verified Apollo, DrivingSDK, MMCV, MMDetection, MMSegmentation and MMDetection3D sources. Add the reproducible Docker runtime and FP32 validation scripts. FP16 changes are excluded."
```

给统一仓创建版本标签：

```bash
git tag -a apollo-npu-fp32-workspace-v1 \
  -m "Verified Apollo FP32 NPU workspace v1"
```

添加你有权限的远端并上传：

```bash
git remote add origin <代码仓地址>
git push -u origin main
git push origin apollo-npu-fp32-workspace-v1
```

将 `<代码仓地址>` 替换为真实私有仓地址。推送前检查仓库中没有数据、权重、环境包、访问令牌或内部敏感信息。

## 10. 构建并验证 Docker 镜像

取得统一仓提交号：

```bash
cd /data/apollo-npu-workspace/apollo-npu-workspace
export WORKSPACE_COMMIT=$(git rev-parse HEAD)
```

构建镜像：

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --build-arg APOLLO_WORKSPACE_COMMIT="$WORKSPACE_COMMIT" \
  -t <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  .
```

这一步会重新编译 MMCV，耗时取决于服务器。构建成功后先在源服务器检查，不直接推送未经验证的镜像。

环境检查：

```bash
docker run --rm -it \
  --network host \
  --ipc host \
  --privileged \
  <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  /opt/apollo/workspace/scripts/check_environment.sh
```

确认代码版本标签已写入镜像：

```bash
docker image inspect \
  <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
```

输出应等于 `WORKSPACE_COMMIT`。

## 11. 数据和权重采用固定挂载

目标服务器统一准备：

```text
/data/apollo_assets/data/nuscenes
/data/apollo_assets/data/occ_gt_release_v1_0
/data/apollo_assets/ckpts
/data/apollo_outputs
```

实际配置要求的目录名必须与原指导书第 8 章一致。建议将数据根整体挂载到容器内：

```text
/opt/apollo/workspace/Apollo-Vision-Net/data
```

将初始权重挂载到：

```text
/opt/apollo/workspace/Apollo-Vision-Net/ckpts
```

将训练输出挂载到：

```text
/outputs
```

数据和权重的下载地址、许可证、目录结构和 SHA256 写入 `manifests/data_and_weights.md`。不把它们放入 Git 或通用运行镜像。

## 12. 用真实数据完成镜像验收

确认设备 8 空闲后执行：

```bash
docker run --rm -it \
  --name apollo_fp32_smoke \
  --network host \
  --ipc host \
  --privileged \
  -e ASCEND_RT_VISIBLE_DEVICES=8 \
  -e OMP_NUM_THREADS=1 \
  -e RUN_DIR=/outputs/fp32_smoke_01 \
  -v /data/apollo_assets/data:/opt/apollo/workspace/Apollo-Vision-Net/data:ro \
  -v /data/apollo_assets/ckpts:/opt/apollo/workspace/Apollo-Vision-Net/ckpts:ro \
  -v /data/apollo_outputs:/outputs \
  <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  /opt/apollo/workspace/scripts/smoke_train_fp32.sh
```

各项含义：

- `--rm`：容器退出后删除容器实例，不删除镜像和挂载输出。
- `--network host`、`--ipc host`、`--privileged`：沿用原指导书已验证的专用开发环境运行方式。
- 三个 `-e`：指定空闲 NPU、CPU 线程数和输出目录。
- 数据、权重使用只读挂载；输出目录可写。
- 镜像内代码和 Python 环境均为构建时固定版本。

通过标准：能进入训练迭代；loss 和 `grad_norm` 为有限值；没有持续跳过优化器更新；日志写入宿主机 `/data/apollo_outputs/fp32_smoke_01`。单卡通过后，再运行原指导书规定的多卡一轮训练和 checkpoint 恢复检查。

## 13. 推送 Docker 镜像

登录镜像仓库：

```bash
docker login <镜像仓地址>
```

推送已通过检查的镜像：

```bash
docker push <镜像仓地址>/apollo-npu-fp32:20261020-v1
```

建议同时记录不可变 digest：

```bash
docker image inspect \
  <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  --format '{{json .RepoDigests}}'
```

正式交付时优先提供 digest，而不只依赖可能被覆盖的 tag。

## 14. 另一台服务器如何拉取并运行

### 14.1 拉取代码

```bash
cd /data
git clone <代码仓地址> apollo-npu-workspace
cd apollo-npu-workspace
git switch --detach apollo-npu-fp32-workspace-v1
```

代码仓用于查看源码、配置、版本和重建镜像。直接运行默认使用镜像内同一提交的代码，避免宿主机依赖污染。

### 14.2 拉取环境镜像

```bash
docker pull <镜像仓地址>/apollo-npu-fp32:20261020-v1
```

如交付了 digest，进一步确认拉取结果与 manifest 一致。

### 14.3 准备数据和输出目录

```bash
mkdir -p /data/apollo_assets/data
mkdir -p /data/apollo_assets/ckpts
mkdir -p /data/apollo_outputs
```

按 `manifests/data_and_weights.md` 放置并校验数据和权重。

### 14.4 启动容器

```bash
docker run --rm -it \
  --name apollo_fp32 \
  --network host \
  --ipc host \
  --privileged \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=1 \
  -v /data/apollo_assets/data:/opt/apollo/workspace/Apollo-Vision-Net/data:ro \
  -v /data/apollo_assets/ckpts:/opt/apollo/workspace/Apollo-Vision-Net/ckpts:ro \
  -v /data/apollo_outputs:/outputs \
  <镜像仓地址>/apollo-npu-fp32:20261020-v1 \
  bash
```

进入后不需要 `conda activate`，直接执行：

```bash
/opt/apollo/workspace/scripts/check_environment.sh
```

然后运行 smoke test：

```bash
RUN_DIR=/outputs/fp32_smoke \
  /opt/apollo/workspace/scripts/smoke_train_fp32.sh
```

## 15. 修改代码后的规则

镜像中代码与环境是一套固定发布物。接收方修改统一代码仓后，不应把宿主机整个仓库直接覆盖挂载到镜像 `/opt/apollo/workspace`，因为 mmcv、mmdet 和 mmdet3d 的安装版本可能与修改后源码不一致。

正确流程是：

1. 修改 `apollo-npu-workspace`。
2. 提交代码并生成新标签。
3. 使用同一 Dockerfile 构建新镜像标签。
4. 完成环境检查和 FP32 smoke test。
5. 推送新的镜像 tag/digest。

这样代码提交和镜像版本始终能够一一对应。

## 16. 完成交付判定

- [ ] Apollo 目录来自已验证 FP32 标签，不包含 FP16 配置和改动。
- [ ] 原 `/data/apollo-npu-workspace` 下本路线全部必要源码已经纳入 `apollo-npu-workspace`。
- [ ] 新仓记录了六个运行源码目录的原始 commit。
- [ ] 数据、权重、环境包、训练输出没有进入 Git。
- [ ] Docker 镜像内 Python 环境位于 `/opt/apollo/env`，不依赖旧用户路径。
- [ ] MMCV NPU 扩展在镜像构建阶段成功编译并导入。
- [ ] 镜像 label 中的 revision 与统一代码仓提交一致。
- [ ] 源服务器单卡 FP32 smoke test 通过。
- [ ] 镜像已推送并记录 digest。
- [ ] 目标服务器能够完成 `git clone`、`docker pull`、数据挂载和 FP32 smoke test。

满足以上条件后，接收方不需要重建 `/data/apollo-npu-workspace/env2_apollo`，也不需要重新按照依赖章节逐包安装；只需使用固定代码标签、固定镜像和经过校验的数据与权重即可运行。
