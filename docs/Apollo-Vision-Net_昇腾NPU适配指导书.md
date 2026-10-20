# Apollo-Vision-Net 昇腾 NPU 适配指导书

版本：v8.0 · 执行顺序重构版 · 2026-10-15

本书只指导一条明确的路线：将 Apollo 的 **DLA-34 检测 + Occupancy 配置**迁移到当前昇腾环境，完成单卡训练检查、8 卡训练、权重保存和验证集评估。

**阅读顺序：认识模型 → 准备环境 → 完成全部代码适配 → 准备数据 → 训练得到权重 → 推理与评估。**

- 第 1～3 章：理解迁移范围，确认机器、代码和初始权重。
- 第 4～6 章：准备容器、env2、mmcv 和旧版依赖；这里解决安装、导入兼容问题。
- [第 7 章](#7-一次完成-apollo-代码适配)：一次完成训练、数据加载、推理和评估所需的代码修改。
- [第 8 章](#8-准备数据与索引)：准备数据及索引。
- [第 9 章](#9-训练检查与正式训练)：训练检查与正式训练，产出 checkpoint。
- [第 10 章](#10-有权重后再推理与评估)：加载 checkpoint，推理、评估并阅读指标。
- 第 11 章：确认完成程度，保存可复现材料。

已经跑通的服务器不需要重新安装。核对第 6、7 章是否与服务器一致后，可直接使用第 9、10 章。**本书不会要求先执行一个必然失败的命令，再补救。**

本文依据本地源码、DrivingSDK 随仓补丁及此前服务器日志整理；本次文档核对没有重新运行服务器训练。此前日志已证明 8 卡训练和完整验证集评估可以运行，但是否完成 24 epoch、是否达到公开精度，仍以对应训练目录中的最终日志和指标为准。

## 1. 先认识模型：迁移的究竟是什么

### 1.1 唯一配置入口

服务器上的配置文件是：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/configs/bevformer/bev_tiny_det_occ_apollo.py
~~~

它实际选择 `img_backbone.type='DLA'`，不是文件开头旧注释中的 ResNet-50。阅读配置时以生效的字段为准。

模型依次完成以下工作：

1. 接收 6 路相机图像，训练时还使用历史帧信息。
2. **DLA-34** 提取图像特征，**SECONDFPNV2** 融合多尺度特征。
3. **BEVFormer Encoder** 将图像特征转换到鸟瞰空间，并融合历史 BEV。
4. 检测 Decoder 和 Occupancy 分支利用 BEV 特征，分别预测 3D 目标框与空间体素类别。

配置中这两项任务组织在 `pts_bbox_head.type='BEVFormerOccupancyHeadApollo'` 中，不存在需要另外配置的 `bbox_head`、`occupancy_head` 两个顶层字段。

### 1.2 为什么选 DLA-34，不处理 DCNv3

这是 Apollo README 展示的主要优化路线，当前配置、深度预训练权重和既有训练记录都与它对应。先完成这条路线，才能把结果与相应的公开配置作比较。

仓库同时保留了 InternImage 骨干。**DCNv3 是其使用的可变形卷积算子，不是 DLA-34 之后又串联的一个骨干。** 它与 BEVFormer 中的 MSDeformAttn 也不是同一种算子。

对应代码位于 Apollo 仓：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/backbones/internimage.py
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3/
~~~

当前配置不构建 InternImage，因此不要求适配 DCNv3。但 Python 包有时会在导入阶段提前导入未使用的分支，所以第 7.1 节还要核对骨干导入开关。这与“模型实际运行了 DCNv3”是两回事。

本书不声称 DCNv3 无法适配，也不声称 DrivingSDK 覆盖仓库所有分支；它只是明确本次工作的边界。

### 1.3 仓库代码怎么找

以下均属于 **Apollo-Vision-Net 仓**，按运行顺序阅读即可：

- 模型入口：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`。
- 图像骨干：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/models/backbones/dla.py`。
- 特征融合：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/models/necks/second_fpnv2.py`。
- BEV 特征转换：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/transformer.py`。
- Encoder、时序注意力、空间注意力及检测 Decoder：同一 `modules` 目录中的 `encoder.py`、`temporal_self_attention.py`、`spatial_cross_attention.py`、`decoder.py`。
- 检测与 Occupancy 输出、损失：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`。
- 数据读取及 Occupancy 评估：`/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`。

这里提供的是“读代码的地图”，不是要求逐个修改。真正需要改的文件和位置集中在第 6、7 章。

### 1.4 从底层到上层，各个包负责什么

从下往上理解：

1. **驱动、固件、CANN**：让系统和运行时能够操作 NPU。
2. **PyTorch + torch_npu**：张量计算、自动求导及 NPU 后端；HCCL 负责多卡通信。
3. **mx_driving 与 mmcv**：前者提供 DrivingSDK 的 NPU 加速算子；后者提供配置、训练运行器、并行封装和一组编译算子。
4. **mmdet、mmseg、mmdet3d**：提供检测/分割组件、3D 框、数据集、评估和注册机制。
5. **Apollo 插件与配置**：组合 DLA、BEVFormer、检测/OCC 任务，决定本次实际运行哪些模块。
6. **训练和测试入口**：加载数据、启动训练、保存权重、计算指标。

`timm` 是图像模型组件库。本地 InternImage 文件使用了它的 `DropPath`、`trunc_normal_`；DLA 文件没有直接使用它。保留 `timm==0.4.12` 是兼容原仓依赖，不代表当前训练启用了 InternImage。

**为什么主要编译 mmcv？** 因为它包含本路线要加载的 C++/设备扩展。mmdet、mmseg 主要安装 Python 组件；mmdet3d 原本也会编译点云 CUDA 扩展，但本书对当前相机路线关闭这些扩展的构建，并处理相应导入。不能把“这次不编译”理解为“这些包永远没有原生扩展”。

### 1.5 什么是算子，官方补丁能保证到哪一步

“算子”可以简单理解为一次计算：卷积、矩阵乘法、旋转采样、注意力等。大部分普通 PyTorch 运算由 torch_npu 承接；少数代码直接调用 CUDA 扩展，需要改成 NPU 实现。

本路线最明确的替换是 **MSDeformAttn（多尺度可变形注意力）**：

- 时序注意力中调用一次这种计算；
- 空间交叉注意力中也调用这种计算；
- 检测 Decoder 中还调用这种计算。

这是**三个代码调用位置，使用同一种底层运算**，不是三个完全不同的算子。第 7.2 节会逐个指出修改位置。

DrivingSDK 的 BEVFormer 补丁面向其指定的 BEVFormer 版本，不是 Apollo 的完整补丁。Apollo 加入了 DLA、不同的特征融合与任务头，不能直接整包应用模型补丁，也不能据此证明所有运算数值完全等价。

“算子级验收”是开发者单独比较某个算子输出、梯度与参考实现的专项测试；它不等于整网能训练，也不能用一个固定的误差阈值覆盖所有输入。本执行流程不再放置只有注释、不能运行的此类测试。首次落地先完成明确修改，再通过第 9、10 章验证真实训练与评估；需要做精度归因时，再按相同输入、dtype、前向和反向分别设计专项对比。

## 2. 确认机器、版本与路径

### 2.1 本项目的版本基线

这是此前服务器使用的组合，不是宣称所有机器只能使用这一种组合：

- 系统：openEuler 24.03 LTS SP1，aarch64。
- 驱动/固件记录：HDK 26.1.1；CANN 9.1.0。
- 镜像：DrivingSDK 26.1.0，**a3** 变体，Python 3.10。
- PyTorch 2.7.1，torch_npu 2.7.1.post6。
- mx-driving 26.1.0.post2。
- mmcv 1.7.2，应用对应 NPU 补丁后编译。
- mmdet 2.19.0、mmseg 0.20.0、mmdet3d 0.18.1。
- NumPy 1.23.5、Numba 0.56.4、YAPF 0.40.1、timm 0.4.12。

旧记录采用 Atlas A3/910C 环境。不要仅凭 `npu-smi` 中的“Ascend910”字符串重新判断具体型号，也不要把板卡数量、芯片数量、进程可见设备数量混为一谈。本文正式训练使用 **8 个可见 NPU 设备、8 个训练进程**。

### 2.2 在宿主机检查，不安装新的 Python

~~~bash
npu-smi info
docker --version
docker ps -a
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
df -h /data
~~~

确认设备健康、驱动与镜像配套、数据和权重存储空间充足。解压数据时还会同时占用压缩包和展开数据的空间，不能只按下载包大小预留。

本工作区 README 有设备占用提醒。**先确认获准使用的设备号，不要停止他人的任务，也不要默认 0～7 都空闲。** 后文设备号仅作命令示例，必须换成实际分配结果。

### 2.3 统一路径约定

- 本地 Windows 仓库根目录：`D:\apollo-npu-workspace`。
- 服务器及容器内工作根目录：`/data/apollo-npu-workspace/`。
- 本书命令除特别说明外，均在**服务器容器内的 Bash** 执行，不是在本地 PowerShell 执行。
- 环境目录：`/data/apollo-npu-workspace/env2_apollo`。
- 运行训练、转换数据、评估时，当前目录统一为 `/data/apollo-npu-workspace/Apollo-Vision-Net`。

本地修改文件不会自动同步到服务器。将需要的源文件同步到相同相对位置后，才会影响服务器运行。本文列出的代码修改是指导操作，不表示本次文档重构已经替你修改了服务器。

## 3. 准备代码与初始权重

### 3.1 本路线需要哪些仓库

服务器目录应包括：

~~~text
/data/apollo-npu-workspace/
├── Apollo-Vision-Net/       模型、配置、训练和测试工具
├── DrivingSDK/              NPU 算子库说明、BEVFormer 参考补丁
├── mmcv/                    1.7.2 源码，编译 NPU 扩展
├── mmdetection_2.19/         安装为 Python 包 mmdet
├── mmsegmentation_0.20/      安装为 Python 包 mmseg
├── mmdetection3d_0.18.1/     安装为 Python 包 mmdet3d
└── env2_apollo/             后续创建
~~~

已有目录先查看版本和本地改动，**不要重新 clone、覆盖或删除已经适配的仓库**：

~~~bash
cd /data/apollo-npu-workspace
git -C Apollo-Vision-Net status --short
git -C Apollo-Vision-Net log -1 --format="%H %ci"
git -C DrivingSDK log -1 --format="%H %ci"
git -C mmcv status --short
git -C mmcv log -1 --format="%H %ci"
~~~

只有缺少某个目录时，才单独执行对应命令：

~~~bash
cd /data/apollo-npu-workspace
git clone https://github.com/ApolloAuto/Apollo-Vision-Net.git
git clone https://gitcode.com/Ascend/DrivingSDK.git
git clone -b v1.7.2 https://github.com/open-mmlab/mmcv.git
git clone -b v2.19.0 https://github.com/open-mmlab/mmdetection.git mmdetection_2.19
git clone -b v0.20.0 https://github.com/open-mmlab/mmsegmentation.git mmsegmentation_0.20
git clone -b v0.18.1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d_0.18.1
~~~

已有 mmcv 若来自 `1.x` 分支且实际版本为 1.7.2，保留当前已验证提交，不要为对齐命令再切换分支。DrivingSDK 也要保存实际提交号，不能把不断变化的 master 当成永久固定基线。

BEVFormer、detectron2、mmdetection_2.24、mmdetection3d_1.0rc4 可作官方示例参考，**不是这条 Apollo env2 安装流程的必经步骤**。

### 3.2 三个 ckpts 权重分别是什么

Apollo 的目录是 `ckpts`，不是 `ckpt`：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/ckpts/
~~~

- `depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`：当前 DLA 骨干的深度预训练初始化权重。
- `resnet50-0676ba61.pth`：ResNet-50 初始化权重，供另一类配置使用。
- `r101_dcn_fcos3d_pretrain.pth`：R101-DCN/FCOS3D 预训练权重，供相应 base 配置使用。

它们都不能替代“当前 Apollo 检测 + OCC 整网已经训练好”的 checkpoint。当前流程只需要第一项；不要把另外两个权重混合加载到 DLA 配置。

~~~bash
test -s /data/apollo-npu-workspace/Apollo-Vision-Net/ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth
~~~

本地 README 没有提供该 DLA 多任务整网的成品权重链接。第 9 章训练得到的 `epoch_N.pth` 才是第 10 章评估使用的权重。

## 4. 使用容器

### 4.1 已有 apollo_dev：直接进入

在宿主机执行：

~~~bash
docker ps -a --filter name=apollo_dev
docker exec -it apollo_dev bash
~~~

如果容器处于停止状态，确认没有维护操作后先 `docker start apollo_dev`。不要为了更新指导书删除或重建现有容器。

### 4.2 仅新机器：创建容器

使用以下镜像：

~~~text
swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10
~~~

下列模板参照 DrivingSDK 仓的 `/data/apollo-npu-workspace/DrivingSDK/docs/zh/installation/installation.md`。它使用 host 网络、host IPC 和 privileged 权限，适合本项目已获批准的专用开发环境；共享或受管服务器必须先按管理员要求确定权限与设备隔离方案。**privileged 不能作为所有设备或线程错误的通用解决办法。**

~~~bash
docker run -it -d \
  --name apollo_dev \
  --ipc=host --network=host --privileged -u=root \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /data/apollo-npu-workspace:/data/apollo-npu-workspace \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10 \
  /bin/bash
~~~

创建前核实模板中的宿主机文件确实存在。该 privileged 模板会扩大设备访问权限，列出 `davinci0` 不构成“只允许使用 0 号卡”的隔离保证；训练时仍须选择获准的设备。

### 4.3 容器内确认基础环境

~~~bash
npu-smi info
conda env list
conda activate torch2.7.1
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
~~~

再选择一张获准使用的设备做一次基础计算。示例设备号为 0，应按分配结果调整：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=0
python - <<'PY'
import torch, torch_npu
assert torch.npu.is_available()
torch.npu.set_device(0)
x = torch.ones((2, 2), device="npu")
print((x @ x).cpu())
PY
~~~

输出应是各元素均为 2 的矩阵。这只检查设备能实际执行计算，不是 Apollo 模型测试。若设备枚举正常但计算失败，先核对芯片对应镜像、CANN/驱动配套和容器权限，不要进入模型安装阶段。


若 shell 没有初始化 conda，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。若缺少 CANN 动态库环境，检查镜像初始化，并按实际安装位置加载 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，不要随意将其他版本库加入搜索路径。

## 5. 建立独立的 env2 环境

### 5.1 首次创建

在容器内执行；目录已经存在时跳过创建：

~~~bash
conda create -y -p /data/apollo-npu-workspace/env2_apollo --clone torch2.7.1
conda activate /data/apollo-npu-workspace/env2_apollo
python -c "import sys; print(sys.executable)"
python -m pip --version
~~~

两项输出都应指向 `/data/apollo-npu-workspace/env2_apollo`。不要向镜像的 `bevformer` 对照环境安装 Apollo 老版本依赖。

### 5.2 后续每次打开新终端

~~~bash
docker exec -it apollo_dev bash
conda activate /data/apollo-npu-workspace/env2_apollo
cd /data/apollo-npu-workspace/Apollo-Vision-Net
~~~

第一行在宿主机执行，后两行在容器内执行。env2 位于挂载目录中，环境文件可保留；但容器的系统库和驱动配套仍是运行前提，不能只复制 conda 目录就认为换机器也可运行。

## 6. 一次完成依赖安装与导入兼容

本章解决“软件能安装、Python 能导入”。下一章解决“模型及训练评估流程如何在 NPU 上执行”。请完成两章再准备运行模型。

### 6.1 固定基础依赖，避免自动升级破坏旧框架

在服务器新建 `/data/apollo-npu-workspace/apollo_constraints.txt`，内容如下：

~~~text
torch==2.7.1
torch-npu==2.7.1.post6
numpy==1.23.5
numba==0.56.4
yapf==0.40.1
timm==0.4.12
~~~

然后在 env2 安装依赖：

~~~bash
python -m pip install -c /data/apollo-npu-workspace/apollo_constraints.txt \
  numpy==1.23.5 numba==0.56.4 yapf==0.40.1 timm==0.4.12 \
  nuscenes-devkit pyquaternion terminaltables pycocotools \
  matplotlib seaborn prettytable plyfile scikit-image shapely \
  lyft_dataset_sdk trimesh tensorboard networkx ipython fvcore \
  scikit-learn tqdm
~~~

这些依赖解决的不是同一件事：

- **NumPy 1.23.5**：保留旧框架使用的部分 NumPy API；不要笼统理解为“PyTorch 2.7 不支持 NumPy 2”。
- **Numba 0.56.4**：配合 Python 3.10、NumPy 1.23；不要照搬旧 requirements 中的 Numba 0.48。
- **YAPF 0.40.1**：保留 mmcv 1.x 输出配置时使用的接口，避免格式化配置阶段失败。
- **nuscenes-devkit、pyquaternion**：数据解析、坐标变换和检测评估。
- **terminaltables、prettytable、pycocotools、lyft_dataset_sdk、plyfile、trimesh 等**：旧版框架在导入、数据处理或评估时需要的配套包。
- **TensorBoard**：记录和显示训练、验证曲线；不要求安装 TensorFlow。
- **timm**：保留原仓兼容依赖，作用见第 1.4 节。

这不是完整的版本锁文件。未列出版本的配套包可能随时间变化；服务器已跑通时，应优先保留其现有环境，并按第 11 章导出完整清单。部分包包含原生代码，不应把整个列表称为“纯 Python 包”。

### 6.2 编译并安装 mmcv NPU 版本

所属仓库：**mmcv**。参考补丁来自 **DrivingSDK**：

~~~text
源码：/data/apollo-npu-workspace/mmcv/
补丁：/data/apollo-npu-workspace/DrivingSDK/model_examples/BEVFormer/mmcv_config.patch
~~~

先核对版本，再应用补丁：

~~~bash
cd /data/apollo-npu-workspace/mmcv
git status --short
git apply --check /data/apollo-npu-workspace/DrivingSDK/model_examples/BEVFormer/mmcv_config.patch
~~~

仅当检查成功，且尚未应用过时，执行：

~~~bash
git apply /data/apollo-npu-workspace/DrivingSDK/model_examples/BEVFormer/mmcv_config.patch
python -m pip install -c /data/apollo-npu-workspace/apollo_constraints.txt -r requirements/runtime.txt
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
~~~

如果 `--check` 失败，先区分“已经应用”和“源码版本不匹配”：可用 `git apply --reverse --check <补丁完整路径>` 检查是否已应用。不要反复打补丁，也不要默认用 `--reject` 留下半成功状态。编译内存不足可调低 `MAX_JOBS`。

编译前后保持同一个 env2 环境，不再安装 PyPI 的普通 mmcv/mmcv-full 覆盖它。检查当前环境实际加载的 mmcv：

~~~bash
python - <<'PY'
import mmcv
import mmcv._ext as ext
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.ops import points_in_boxes_part, sigmoid_focal_loss

print("mmcv:", mmcv.__version__, mmcv.__file__)
assert mmcv.__version__ == "1.7.2"
assert hasattr(ext, "ms_deform_attn_forward")
assert hasattr(ext, "ms_deform_attn_backward")
print("mmcv 扩展和 Apollo 所需导入可用")
PY
~~~

**这里检查的是扩展加载和符号，不是所有算子的 NPU 数值覆盖。** `points_in_polygons` 与 `points_in_boxes` 也不是同一个算子，不能拿前者的补丁名证明后者已全部支持。

如果 `mmcv._ext` 加载失败，必须在本步修正环境或编译产物；不能在 Apollo 中加空壳模块或忽略异常。修改、重编后重新检查，避免 Python 实际导入了另一个环境里的旧 mmcv。

### 6.3 安装三个旧版框架前，先调整版本检查和构建入口

分别打开以下文件，只调整上限，保留下限：

1. **mmdetection_2.19 仓**：`/data/apollo-npu-workspace/mmdetection_2.19/mmdet/__init__.py`，设为 `mmcv_maximum_version = '1.7.2'`。
2. **mmsegmentation_0.20 仓**：`/data/apollo-npu-workspace/mmsegmentation_0.20/mmseg/__init__.py`，设为 `MMCV_MAX = '1.7.2'`。
3. **mmdetection3d_0.18.1 仓**：`/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/__init__.py`，设为 `mmcv_maximum_version = '1.7.2'`。

原因：这些旧版本的上限早于本次使用的 mmcv。放宽检查只允许组合继续加载，不代表自动证明所有 API 兼容。

再打开 **mmdetection3d_0.18.1 仓**的：

~~~text
/data/apollo-npu-workspace/mmdetection3d_0.18.1/setup.py
~~~

在文件底部 `setup(...)` 的参数中，将 `ext_modules=get_extensions()` 改为 `ext_modules=[]`。其他构建参数不变。

原因：当前相机路线不使用这批 mmdet3d 自带的点云 CUDA 扩展；NPU 环境无需为它们安装 CUDA。**这一修改不适用于需要 voxel、点云 NMS 等算子的其他模型。**

### 6.4 处理 mmdet3d 的可选 CUDA 扩展导入

不编译扩展后，旧框架仍会在导入包时尝试加载它们。因此要把**以下指定导入**改成可选；不是把所有 ImportError 都屏蔽。

以下文件均属于 **mmdetection3d_0.18.1 仓**。公共目录为：

~~~text
/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/ops/
~~~

在各文件顶部找到下面的扩展名，按示例包一层 `try/except ImportError`。已是相同形式的不要重复包裹：

- `ball_query/ball_query.py`：`ball_query_ext`。
- `furthest_point_sample/furthest_point_sample.py`：`furthest_point_sample_ext`。
- `gather_points/gather_points.py`：`gather_points_ext`。
- `group_points/group_points.py`：`group_points_ext`。
- `interpolate/three_nn.py`、`interpolate/three_interpolate.py`：`interpolate_ext`。
- `iou3d/iou3d_utils.py`：`iou3d_cuda`。
- `knn/knn.py`：`knn_ext`。
- `paconv/assign_score.py`：`assign_score_withk_ext`。
- `roiaware_pool3d/points_in_boxes.py`、`roiaware_pool3d/roiaware_pool3d.py`：`roiaware_pool3d_ext`。
- `spconv/ops.py`：`sparse_conv_ext`。

例如完整文件 `/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/ops/ball_query/ball_query.py` 中：

~~~python
# 原来：from . import ball_query_ext
try:
    from . import ball_query_ext
except ImportError:
    ball_query_ext = None
~~~

另外两个 voxel 文件导入的是函数，不能照抄同一段：

**文件一**：`/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/ops/voxel/scatter_points.py`，顶部改为：

~~~python
try:
    from .voxel_layer import (
        dynamic_point_to_voxel_backward,
        dynamic_point_to_voxel_forward,
    )
except ImportError:
    dynamic_point_to_voxel_backward = None
    dynamic_point_to_voxel_forward = None
~~~

**文件二**：`/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/ops/voxel/voxelize.py`，顶部改为：

~~~python
try:
    from .voxel_layer import dynamic_voxelize, hard_voxelize
except ImportError:
    dynamic_voxelize = None
    hard_voxelize = None
~~~

本地核对发现 `voxelize.py` 曾误复制成前一个文件的导入名，应按上面分别处理。

**文件三**：`/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/core/bbox/structures/base_box3d.py`，顶部对应导入改为：

~~~python
try:
    from mmdet3d.ops.iou3d import iou3d_cuda
except ImportError:
    iou3d_cuda = None
~~~

这些改动只让**当前没有执行的分支**不阻塞包导入，不会把 CUDA 运算变成 NPU 运算。后续若切换到使用它们的模型，需要重新适配对应实现，不能继续依赖 `None`。mmcv 的 `_ext` 是必需导入，不属于这里的可选范围。

### 6.5 处理注册冲突和 Numba 导入

仍在 **mmdetection3d_0.18.1 仓**：

- 文件 `/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/ops/spconv/conv.py`：将该文件中的 10 处 `@CONV_LAYERS.register_module()` 改为 `@CONV_LAYERS.register_module(force=True)`。用于处理本版本组合中同名卷积注册冲突，不要全仓替换所有注册器。
- 文件 `/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/datasets/pipelines/data_augment_utils.py`：将 `NumbaPerformanceWarning` 的旧导入改为 `from numba.core.errors import NumbaPerformanceWarning`。

### 6.6 安装并确认导入来源

完成前述修改后执行：

~~~bash
python -m pip install --no-deps --no-build-isolation -e /data/apollo-npu-workspace/mmdetection_2.19
python -m pip install --no-deps --no-build-isolation -e /data/apollo-npu-workspace/mmsegmentation_0.20
python -m pip install --no-deps --no-build-isolation -e /data/apollo-npu-workspace/mmdetection3d_0.18.1
python - <<'PY'
import torch, torch_npu, mx_driving
import mmcv, mmdet, mmseg, mmdet3d, numpy, numba, yapf, timm
from importlib.metadata import version

for module in (torch, torch_npu, mmcv, mmdet, mmseg, mmdet3d, numpy, numba, timm):
    print(module.__name__, getattr(module, "__version__", "?"), module.__file__)
print("mx-driving:", version("mx-driving"), mx_driving.__file__)
print("yapf:", version("yapf"))
assert callable(mx_driving.multi_scale_deformable_attn)
print("依赖导入通过")
PY
~~~

`--no-deps` 防止旧包安装流程重新拉取另一版 mmcv；`--no-build-isolation` 使用当前已具备 torch 的构建环境。它们不替代第 6.1 节的依赖安装。

**本章完成标志**：版本正确、三个框架来自指定源码目录、mmcv 扩展可加载、mx_driving 接口可导入。此时才进入模型代码适配。


## 7. 一次完成 Apollo 代码适配

本章结束后，训练、worker 加载、权重评估所需的已知兼容修改都应完成。**第 9、10 章不再要求临时改代码。**

**本章统一采用 diff 阅读方式：**

- `-` 开头：删除的旧代码。
- `+` 开头：新增的代码，写入文件时不带这个 `+`。
- 空格开头：定位上下文，保持不变。
- `@@`：说明修改位置，不是要写入源码的内容。

每项先给出**所属仓库、完整路径、定位位置**，再列出修改。已与 `+` 侧目标一致的地方不用重复修改。片段用于逐处对照，不是可直接整包 `git apply` 的补丁；尤其不要把 DrivingSDK 的 BEVFormer 模型补丁整包应用到 Apollo。修改前保留文件备份。

### 7.1 初始化 NPU，统一单卡和多卡入口

**A. 训练入口：将 NPU 初始化移到框架导入之前。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/train.py
~~~

定位：文件顶部导入区。以当前文件为基础，移动已有的 NPU 初始化，不保留两份：

~~~diff
@@ 顶部导入区，保留前面的 from __future__ import division
 import argparse
 import copy
+import torch
+import torch_npu
+from torch_npu.contrib import transfer_to_npu
+
+torch.npu.config.allow_internal_format = False
+
 import mmcv
 import os
 import time
-import torch
 import warnings
@@ 下方原有初始化处，删除已经上移的代码
 from mmcv.utils import TORCH_VERSION, digit_version

-import torch_npu
-from torch_npu.contrib import transfer_to_npu
-torch.npu.config.allow_internal_format = False
~~~

如果使用尚未添加 NPU 初始化的原始文件，只做第一段；第二段不存在就不需要删除。作用是让后续框架导入和模型执行使用 NPU 兼容层。

**B. 测试入口：同样显式初始化 NPU。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/test.py
~~~

定位：文件顶部导入区：

~~~diff
@@ 导入区
 import argparse
 from unittest import result
+import torch
+import torch_npu
+from torch_npu.contrib import transfer_to_npu
+
+torch.npu.config.allow_internal_format = False
+
 import mmcv
 import os
-import torch
 import warnings
~~~

启用 transfer_to_npu 后，旧代码中的 `.cuda()` 和部分 CUDA 接口由兼容层承接；不用全仓机械替换“cuda”字符串。

**C. 不提前导入当前不用的 InternImage。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/__init__.py
~~~

定位：文件末尾的 backbones 导入。当前已经注释则不再改：

~~~diff
@@ 文件末尾
 from .runner import *
 from .hooks import *
-from .backbones import *
+# from .backbones import *
~~~

这里控制的是 BEVFormer 子目录中的 InternImage 注册入口，不是 DLA 所在的 `models/backbones`。其他导入保持原状。

**D. 通信后端改为 HCCL。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/configs/_base_/default_runtime.py
~~~

定位：`dist_params`：

~~~diff
@@ dist_params
-dist_params = dict(backend='nccl')
+dist_params = dict(backend='hccl')
 log_level = 'INFO'
~~~

HCCL 负责 NPU 进程间通信。单卡也走同一个分布式入口，只启动一个进程，避免进入本仓未完善的非分布式分支。

### 7.2 将三个注意力调用改为 mx_driving

以下三个文件都属于 **Apollo-Vision-Net 仓**。每个文件均有“添加导入”和“替换 forward 内调用”两处修改。下面的 `-` 展示迁移前 CUDA 写法；如果已经是 `mx_driving` 调用，核对参数一致即可。

**A. 检测 Decoder。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/decoder.py
~~~

定位：顶部 `ext_module` 后，以及 `CustomMSDeformableAttention.forward`：

~~~diff
@@ 顶部扩展导入后
 ext_module = ext_loader.load_ext(
     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
+import mx_driving
@@ CustomMSDeformableAttention.forward
         if torch.cuda.is_available() and value.is_cuda:
-
-            # using fp16 deformable attention is unstable because it performs many sum operations
-            if value.dtype == torch.float16:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            else:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            output = MultiScaleDeformableAttnFunction.apply(
-                value, spatial_shapes, level_start_index, sampling_locations,
-                attention_weights, self.im2col_step)
+            output = mx_driving.multi_scale_deformable_attn(
+                value, spatial_shapes, level_start_index,
+                sampling_locations, attention_weights)
         else:
             output = multi_scale_deformable_attn_pytorch(
                 value, spatial_shapes, sampling_locations, attention_weights)
~~~

**B. 空间交叉注意力中的多尺度采样。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
~~~

定位：顶部 `ext_module` 后，以及 **`MSDeformableAttention3D.forward`**，不是外层 `SpatialCrossAttention.forward`：

~~~diff
@@ 顶部扩展导入后
 ext_module = ext_loader.load_ext(
     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
+import mx_driving
@@ MSDeformableAttention3D.forward
         if torch.cuda.is_available() and value.is_cuda:
-            if value.dtype == torch.float16:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            else:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            output = MultiScaleDeformableAttnFunction.apply(
-                value, spatial_shapes, level_start_index, sampling_locations,
-                attention_weights, self.im2col_step)
+            output = mx_driving.multi_scale_deformable_attn(
+                value, spatial_shapes, level_start_index,
+                sampling_locations, attention_weights)
         else:
             output = multi_scale_deformable_attn_pytorch(
                 value, spatial_shapes, sampling_locations, attention_weights)
~~~

**C. 历史 BEV 与当前 BEV 的时序融合。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
~~~

定位：顶部 `import math` 后，以及 `TemporalSelfAttention.forward`：

~~~diff
@@ 顶部导入区
 import math
+import mx_driving
@@ TemporalSelfAttention.forward
         if torch.cuda.is_available() and value.is_cuda:
-
-            # using fp16 deformable attention is unstable because it performs many sum operations
-            if value.dtype == torch.float16:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            else:
-                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
-            output = MultiScaleDeformableAttnFunction.apply(
-                value, spatial_shapes, level_start_index, sampling_locations,
-                attention_weights, self.im2col_step)
+            output = mx_driving.multi_scale_deformable_attn(
+                value, spatial_shapes, level_start_index,
+                sampling_locations, attention_weights)
         else:

             output = multi_scale_deformable_attn_pytorch(
                 value, spatial_shapes, sampling_locations, attention_weights)
~~~

三处修改的目的都是让设备分支调用 DrivingSDK 的 NPU 算子。**不要传入旧接口的 `self.im2col_step`；不要改动周围的投影、reshape、残差和 CPU 回退。** 判断条件中的“cuda”名称依赖第 7.1 节已启用的兼容层，不是在 NPU 上执行 CUDA 内核。

**本节不需要修改或删除的内容：**

- 三个文件原有的 `MultiScaleDeformableAttnFunction*`、`ext_loader` 导入，以及 `ext_module = ext_loader.load_ext(...)`。
- 公共封装文件 `/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py`。

只替换实际执行的函数，不同时重整导入关系。**`mmcv._ext` 仍必须能加载；“加载失败也无害”是错误说法。** 扩展加载检查必须已在第 6.2 节通过。

### 7.3 调整 BEV 坐标投影、形状常量和历史旋转

本节处理注意力周围的坐标计算和输入准备。所有文件均属于 **Apollo-Vision-Net 仓**。

**A. 修改坐标投影。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/encoder.py
~~~

定位：`BEVFormerEncoder.point_sampling` 中计算 `reference_points_cam` 的语句：

~~~diff
@@ BEVFormerEncoder.point_sampling
-        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
-                                            reference_points.to(torch.float32)).squeeze(-1)
+        reference_points_cam = torch.mul(
+            lidar2img.to(torch.float32),
+            reference_points.to(torch.float32).transpose(-1, -2)
+        ).sum(-1, keepdim=True).squeeze(-1)
         eps = 1e-5
~~~

保留前面的 reshape/repeat 和后面的深度除法、可见性 mask。这只是该处广播计算的适配，不是全仓替换 matmul。

**B. 在同一 encoder.py 中缓存时序注意力的固定形状。**

定位一：`BEVFormerLayer.__init__` 的参数和初始化：

~~~diff
@@ BEVFormerLayer.__init__ 的参数
     def __init__(self,
+                 bev_h,
+                 bev_w,
                  attn_cfgs,
                  feedforward_channels,
@@ BEVFormerLayer.__init__ 中，super 初始化之后
             **kwargs)
         self.fp16_enabled = False
+        self.level_start_index = torch.tensor([0]).npu()
+        self.spatial_shapes = torch.tensor([[bev_h, bev_w]]).npu()
~~~

定位二：`BEVFormerLayer.forward` 中 **`if layer == 'self_attn':`** 内的注意力调用：

~~~diff
@@ BEVFormerLayer.forward 的 self_attn 分支
                     reference_points=ref_2d,
-                    spatial_shapes=torch.tensor(
-                        [[bev_h, bev_w]], device=query.device),
-                    level_start_index=torch.tensor([0], device=query.device),
+                    spatial_shapes=self.spatial_shapes,
+                    level_start_index=self.level_start_index,
                     **kwargs)
~~~

只改 self_attn；后面的 cross_attn 仍使用原有输入。两个常量在建层时放到当前 NPU，必须先绑定设备再建模型；第 7.4 节的分布式入口满足这一顺序。

**C. 配置同步传入 BEV 大小。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/configs/bevformer/bev_tiny_det_occ_apollo.py
~~~

定位：`model → pts_bbox_head → transformer → encoder → transformerlayers`：

~~~diff
@@ encoder.transformerlayers，不是 decoder.transformerlayers
                 transformerlayers=dict(
                     type='BEVFormerLayer',
+                    bev_h=bev_h_,
+                    bev_w=bev_w_,
                     attn_cfgs=[
~~~

两项各保留一次。上层 head 中同名参数不能替代这里的传参；重复添加会造成重复关键字错误。

**D. 历史 BEV 旋转指定双线性插值。**

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/transformer.py
~~~

定位：顶部 torchvision 导入及 `PerceptionTransformer.get_bev_features` 内的 rotate 调用：

~~~diff
@@ 顶部导入区
-from torchvision.transforms.functional import rotate
+from torchvision.transforms.functional import InterpolationMode, rotate
@@ PerceptionTransformer.get_bev_features
                     tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
+                                          interpolation=InterpolationMode.BILINEAR,
                                           center=self.rotate_center)
~~~

这是本方案的旋转采样设置，可能影响数值，精度对比时应记录；不是所有 torchvision 旋转都必须修改。

### 7.4 改用 NPU 并行封装和 torchrun

**A. 训练包装层：导入和四处包装类名一起改。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py
~~~

定位：顶部导入，以及 `custom_train_detector` 的模型包装部分：

~~~diff
@@ 顶部导入区
-from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
+from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
@@ custom_train_detector：distributed 分支的 model
-        model = MMDistributedDataParallel(
+        model = NPUDistributedDataParallel(
             model.cuda(),
@@ custom_train_detector：distributed 分支的 eval_model
         if eval_model is not None:
-            eval_model = MMDistributedDataParallel(
+            eval_model = NPUDistributedDataParallel(
                 eval_model.cuda(),
@@ custom_train_detector：非 distributed 分支的 model 和 eval_model
     else:
-        model = MMDataParallel(
+        model = NPUDataParallel(
             model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
         if eval_model is not None:
-            eval_model = MMDataParallel(
+            eval_model = NPUDataParallel(
                 eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
~~~

仅换包装类，`device_ids`、`broadcast_buffers=False`、`find_unused_parameters` 和 `.cuda()` 保持原状。

**B. 测试包装层。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/test.py
~~~

定位：顶部并行类导入，以及 `main` 中的 distributed 分支：

~~~diff
@@ 顶部并行类导入
-from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
+from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
@@ main：distributed 分支的模型包装
     else:
-        model = MMDistributedDataParallel(
+        model = NPUDistributedDataParallel(
             model.cuda(),
             device_ids=[torch.cuda.current_device()],
             broadcast_buffers=False)
~~~

当前本地文件已经有 NPU 并行类导入及包装；这种情况下不再重复增加，只去掉多余的旧 MM 并行类导入即可。不是把文件中的所有 `else` 都替换，只定位上述模型包装位置。

**C. 训练启动脚本。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/dist_train.sh
~~~

定位：整个短脚本。下面以**当前已换成 torchrun 的本地脚本**为 `-` 侧，整理成第 9 章使用的写法。如果服务器仍使用 `python -m torch.distributed.launch`，该旧启动行也由下面的 `+` 侧 torchrun 行替换。

~~~diff
@@ dist_train.sh
 #!/usr/bin/env bash
-
-CONFIG=$1
-GPUS=$2
-PORT=${PORT:-28509}
-
-PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
-torchrun --nproc_per_node=$GPUS --master_port=$PORT \
-    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
+set -e
+CONFIG="$1"
+NPROC="$2"
+PORT="${PORT:-29509}"
+SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
+export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"
+torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
+  "$SCRIPT_DIR/train.py" "$CONFIG" --launcher pytorch "${@:3}" --deterministic
~~~

**D. 测试启动脚本。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/dist_test.sh
~~~

同样以当前本地脚本为 `-` 侧；保留评估参数 `iou bbox`：

~~~diff
@@ dist_test.sh
 #!/usr/bin/env bash
-
-CONFIG=$1
-CHECKPOINT=$2
-GPUS=$3
-PORT=${PORT:-29503}
-
-PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
-torchrun --nproc_per_node=$GPUS --master_port=$PORT \
-    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval iou bbox
+set -e
+CONFIG="$1"
+CHECKPOINT="$2"
+NPROC="$3"
+PORT="${PORT:-29503}"
+SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
+export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"
+torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
+  "$SCRIPT_DIR/test.py" "$CONFIG" "$CHECKPOINT" \
+  --launcher pytorch "${@:4}" --eval iou bbox
~~~

两个文件保存为 Linux LF 换行。NPROC 表示进程数，单卡设 1、8 卡设 8；测试脚本已附带 `--eval iou bbox`，后面不要重复追加。

### 7.5 提前完成 DataLoader worker 兼容

`workers_per_gpu` 是**每个训练进程用于读取、解码和增强数据的 CPU 子进程数**，不是 NPU 数量，也不是每卡 batch。8 卡、每卡 4 worker，通常共 32 个加载 worker。设为 0 是主进程自己加载，合法但可能更慢。

**A. 修正 NPU pin memory 参数。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/datasets/builder.py
~~~

定位：`build_dataloader` 中创建 DataLoader 之前。**当前本地已存在下面的旧 kwargs 赋值**，将它替换为单项设置：

~~~diff
@@ build_dataloader：构造 DataLoader 之前
         seed=seed) if seed is not None else None
-    kwargs = {"pin_memory_device": "npu"} if torch.__version__ >= "2.0" else {}
+    kwargs["pin_memory_device"] = "npu"
     data_loader = DataLoader(
~~~

若使用原始 CUDA 版本、根本没有旧 kwargs 行，则只在同一位置加入 `+` 行。目的是指定 NPU pin memory，同时不丢弃调用者传入的其他 kwargs。

同一次 DataLoader 调用中，若仍为 `pin_memory=False`，还要改这一处；当前已为 True 则跳过：

~~~diff
@@ build_dataloader：DataLoader 参数
         collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
-        pin_memory=False,
+        pin_memory=True,
         worker_init_fn=init_fn,
         **kwargs)
~~~

保留原有导入和其他参数，不重复添加 `pin_memory`。本方案固定为 PyTorch 2.7.1，无需字符串比较版本号。

**B. 让数据集可以交给 spawn worker。**

所属仓库：**Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py
~~~

定位：`CustomNuScenesDataset.__init__` 中 `super().__init__` 之后：

~~~diff
@@ CustomNuScenesDataset.__init__
         super().__init__(*args, **kwargs)
+        self.eval_detection_configs.class_names = list(
+            self.eval_detection_configs.class_names
+        )
         self.queue_length = queue_length
         self.overlap_test = overlap_test
~~~

原因：评估配置中的 `dict_keys` 不能被 spawn worker 序列化；转为 list 保留类别名和顺序。不要改成强制 fork 来掩盖问题。

这两项必须在训练前完成。配置默认每卡 4 worker；第 9 章先用 1 worker 检查，再回到 4。已验证每卡 8 worker 合适的服务器也可用 8，但不是越多越好。

### 7.6 提前修复无速度 7D 框的评估

所属仓库：**mmdetection3d_0.18.1，不是 Apollo-Vision-Net**。文件：

~~~text
/data/apollo-npu-workspace/mmdetection3d_0.18.1/mmdet3d/datasets/nuscenes_dataset.py
~~~

定位：`output_to_nusc_box(detection)` 中遍历框的循环：

~~~diff
@@ output_to_nusc_box
     for i in range(len(box3d)):
         quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
-        velocity = (*box3d.tensor[i, 7:9], 0.0)
+        if box3d.tensor.shape[1] >= 9:
+            velocity = (*box3d.tensor[i, 7:9], 0.0)
+        else:
+            velocity = (0.0, 0.0, 0.0)
~~~

后面的 `NuScenesBox(..., velocity=velocity)` 不变。

当前配置 `with_velocity=False`，解码后为没有 vx/vy 的 7D 框。旧切片得到空值，再补一个 0 后只有一维，坐标旋转就会报 `shapes (3,3) and (1,) not aligned`。

此处为无速度框补合法的三维零速度，**不是增加速度预测能力**。不需要修改模型的 with_velocity；mAVE、NDS 和属性指标的含义见第 10 章。每轮训练后的自动验证也用这个函数，所以必须在训练前完成。

### 7.7 固定运行配置，避开未实现的测试选项

文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/projects/configs/bevformer/bev_tiny_det_occ_apollo.py
~~~

**本节只核对配置，不是新增补丁。** 当前配置已经包含下列设置，不要重复追加；不要把早期短跑参数永久写进正式配置：

~~~python
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, pipeline=test_pipeline)
find_unused_parameters = True
~~~

在原有 `data` 字典中保持 `samples_per_gpu=1`、`workers_per_gpu=4`，保留 train/val/test 中的 `with_velocity=False`。不要用只含两个字段的新 `data` 字典覆盖数据集定义。

两点在启动前就要明确：

- 只覆盖 `runner.max_epochs=1` 而不覆盖 `total_epochs`，会触发训练器的一致性断言。第 9 章短跑命令会同时设置两者；正式训练使用 24/24。
- 当前 `tools/test.py` 的非分布式分支、`--out` 分支仍是未实现分支，含 `assert False`。本路线不扩展这些功能：**单卡测试也由 dist_test.sh 启动，评估命令不带 --out，不带 --gpu-collect。** 当前自定义结果收集器的 gpu_collect 路径也不适合作为完整 OCC 结果收集入口。

这不是关闭评估：测试脚本仍会计算 OCC 指标，调用 nuScenes 检测评估，并写出检测 JSON/指标文件。只是不用未实现的“额外整包 pickle 导出”选项。此前“跑完 6019 帧后在 --out 断言处失败”的情况由此在启动命令层面避免，不应通过删除所有 assert 来绕过。

### 7.8 整理已有 tuili.py：保留用途，不再作为必经阶段

所属目录：**jiaoyan，位于 Apollo 仓之外**。文件：

~~~text
/data/apollo-npu-workspace/jiaoyan/tuili.py
~~~

定位：文件开头的导入区。以当前本地文件为基础：

~~~diff
@@ 文件开头
-import torch, torch_npu
-import projects.mmdet3d_plugin              # 插件注册
+import torch
+import torch_npu
+from torch_npu.contrib import transfer_to_npu
+
+torch.npu.config.allow_internal_format = False
+torch.npu.set_device(0)
+
 from mmcv import Config
-from mmcv.parallel import collate, scatter
+from mmcv.parallel import collate
+from mmcv.device import scatter
+import projects.mmdet3d_plugin
 from mmdet3d.datasets import build_dataset
 from mmdet3d.models import build_model
~~~

模型和样本构建代码不变。运行时从 Apollo 根目录启动，并使该根目录位于 PYTHONPATH；设备 0 是进程可见的第一个设备。

此脚本只读取一个验证样本做前向，没有加载训练好的整网 checkpoint，不能用其输出评判精度。保留它方便局部检查，但第 9 章直接验证真实训练，不再把它作为必经阶段。

### 7.9 本章结束检查：不需要数据，也不训练

先在获准使用的设备上设置 `ASCEND_RT_VISIBLE_DEVICES`，然后执行：

~~~bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
bash -n tools/dist_train.sh
bash -n tools/dist_test.sh
python - <<'PY'
from pathlib import Path
import torch, torch_npu
from torch_npu.contrib import transfer_to_npu

torch.npu.config.allow_internal_format = False

for base in ("tools", "projects"):
    for path in Path(base).rglob("*.py"):
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")

import projects.mmdet3d_plugin
from mmcv import Config
from mmcv.device.npu import NPUDataParallel, NPUDistributedDataParallel
from mmcv.device import scatter
from mmdet3d.models import build_model

cfg = Config.fromfile("projects/configs/bevformer/bev_tiny_det_occ_apollo.py")
assert cfg.model.img_backbone.type == "DLA"
assert cfg.total_epochs == cfg.runner.max_epochs == 24
assert cfg.dist_params.backend == "hccl"
assert cfg.data.samples_per_gpu == 1
assert cfg.data.train.with_velocity is False
assert cfg.data.val.with_velocity is False
assert cfg.data.test.with_velocity is False
print("语法、插件导入、运行配置检查通过；尚未训练")
PY
~~~

本检查用于尽早发现缩进、重复参数、错误导入和配置不一致，不证明模型精度。检查通过后，第 7 章结束；后面不再边运行边追加这些已知兼容修改。


## 8. 准备数据与索引

第 7 章完成后再处理数据。训练和评估共同使用这里的固定目录；不要把路径问题拖到训练中临时处理。

### 8.1 下载哪几部分

需要三类数据：

1. **nuScenes v1.0-trainval**：trainval 元数据和完整对应传感器文件。若下载页拆成多个 part，需全部下载并合并到同一目录，不能把其中一个 part 当成全量训练集。mini 只适合另外配置的小规模实验，不替代本文的 trainval。
2. **CAN bus 扩展数据**：单独下载解压到 `data/can_bus`。mmdetection3d 的 tools 是处理代码，不代表仓库已附带 CAN bus 数据。
3. **Apollo 所用 occ_gt_release_v1_0**：0.5 m 分辨率、16 类的 Occupancy/flow 标注，包含 train 和 val。

下载入口与标注链接记录在 Apollo 仓的：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/docs/prepare_dataset.md
~~~

该文件给出的 OccGT 百度网盘提取码为 `hgk2`，压缩包约 15 GB；下载链接是否仍有效以访问时为准。

nuScenes 下载项后的美国/亚洲是文件服务区域，不是训练数据的不同地理子集。选可访问且下载稳定的一个区域，避免重复下载相同内容。

当前配置 `use_map=False`，不使用高精地图扩展作为模型输入；**Maps expansion 不是本路线必装项**。保留 nuScenes 正常目录结构中的基础 maps 文件，不要为了开始训练再去挑一套无关的地图扩展版本。

虽然模型只输入相机，**本仓的索引转换器仍会检查/读取 nuScenes 的 LiDAR 样本信息和文件路径**。不能据此只下载相机图片、遗漏转换器需要的 LIDAR_TOP 文件。本流程使用完整 trainval，不要求下载没有公开标注的 test 集。

### 8.2 解压后的目录

容器内应能看到：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/data/
├── can_bus/
├── nuscenes/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
└── occ_gt_release_v1_0/
    ├── train/
    ├── val/
    ├── occ_gt_train.json
    └── occ_gt_val.json
~~~

多个 nuScenes 包要合并相同的 samples、sweeps 目录，不要形成 `data/nuscenes/part01/samples` 或重复的 `nuscenes/nuscenes` 层级。数据可以放在公共存储后链接到此结构，但容器必须能访问链接的真实目标路径。

### 8.3 只生成 trainval 索引

先说明为什么不用原文档的整条 create_data.py 命令：**本地 `tools/create_data.py` 在 `--version v1.0` 时会依次转换 trainval 和 test**。没有下载 test 时，trainval 转完仍会在后半段失败。

这里直接调用同一个转换函数，只执行 trainval，不修改转换算法，也无需为此下载 test：

~~~bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
python - <<'PY'
from tools.data_converter.nuscenes_converter import create_nuscenes_infos

create_nuscenes_infos(
    root_path="./data/nuscenes",
    out_path="./data/nuscenes",
    can_bus_root_path="./data",
    info_prefix="nuscenes",
    version="v1.0-trainval",
    max_sweeps=10,
)
PY
python tools/create_data_with_occ.py
~~~

第一步生成：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/data/nuscenes/nuscenes_infos_temporal_train.pkl
/data/apollo-npu-workspace/Apollo-Vision-Net/data/nuscenes/nuscenes_infos_temporal_val.pkl
~~~

第二步按场景和 sample token 合并 OCC 标注，生成训练配置实际使用的文件：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/data/occ_gt_release_v1_0/nuscenes_infos_temporal_train_occ_gt.pkl
/data/apollo-npu-workspace/Apollo-Vision-Net/data/occ_gt_release_v1_0/nuscenes_infos_temporal_val_occ_gt.pkl
~~~

第一次转换可能耗时较长；已有匹配当前数据目录的完整索引时不必重做。不要把别的机器生成、内部路径失效的 pkl 直接当作可用数据。

### 8.4 训练前一次检查数据完整性

在 Apollo 根目录执行。此命令检查样本数及每帧相机/OCC 文件是否存在，不做模型计算：

~~~bash
python - <<'PY'
from pathlib import Path
import mmcv

root = Path("data/occ_gt_release_v1_0")
for split, expected in (("train", 28130), ("val", 6019)):
    obj = mmcv.load(root / f"nuscenes_infos_temporal_{split}_occ_gt.pkl")
    infos = obj["infos"]
    assert len(infos) == expected, (split, len(infos), expected)
    missing = []
    for info in infos:
        assert len(info["cams"]) == 6, info["token"]
        paths = [cam["data_path"] for cam in info["cams"].values()]
        for key in ("occ_gt_path", "occ_invalid_path", "flow_gt_path"):
            if key in info:
                paths.append(info[key])
        assert "occ_gt_path" in info, info["token"]
        for name in paths:
            if not Path(name).is_file():
                missing.append(str(name))
    if missing:
        print(split, "缺失文件数:", len(missing))
        print("\n".join(missing[:20]))
        raise SystemExit("请先补齐数据或修正索引来源，再开始训练")
    print(split, len(infos), "文件路径检查通过")
PY
~~~

这里的检查不能替代解码校验，但能提前发现大部分漏包和路径错误。train 28130、val 6019 是本仓全量 trainval 的预期样本数；不要为了通过断言修改这个数字来掩盖下载不完整。

## 9. 训练检查与正式训练

本章只执行训练，不再修改适配代码。已经完成单卡、多 worker、8 卡及整轮验证的服务器可直接到第 9.4 节；不要为了遵循新排版重跑旧检查。

### 9.1 先理解训练的几个数字

- **1 iter**：各训练进程完成一次取数据、前向、计算损失、反向及参数更新。
- **1 epoch**：训练采样器遍历一轮训练集；分布式采样可能为对齐各卡补齐少量样本。
- **8 卡**：8 个进程协同训练同一个模型，不是每卡训练一个独立模型，也不是把同一轮变成 8 个 epoch。
- **每卡 samples_per_gpu=1**：8 卡时每步合计处理 8 个样本；一个样本内部又包含多相机和历史帧，不能理解为只有一张图片。
- **24 epoch**：原配置的完整训练计划。不是 NPU 必须训练 24 轮才能运行，但复现该训练方案应使用相同计划。

此前 8 卡日志的 `Epoch [1][2000/3517]` 表示第 1 轮完成了约 2000 次迭代，**第 1 轮还没结束**。默认按 epoch 保存，所以当时还没有 `epoch_1.pth` 是正常现象。

单卡能够运行同一模型，不会因配置源自 8 卡而天然冲突；但单卡的全局 batch、每轮迭代数和训练行为不同。本文单卡用于功能检查，不把它直接当作 8 卡精度复现实验。

### 9.2 第一次迁移：单卡短训练检查

前提：选择一张已分配的空闲设备。下面的 0 仅为示例：

~~~bash
conda activate /data/apollo-npu-workspace/env2_apollo
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
PORT=29509 bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py 1 \
  --work-dir work_dirs/npu_check_1card \
  --no-validate \
  --cfg-options total_epochs=1 runner.max_epochs=1 \
    data.workers_per_gpu=1 log_config.interval=10 \
    checkpoint_config.by_epoch=False checkpoint_config.interval=50 \
    checkpoint_config.max_keep_ckpts=2
~~~

这条命令**不会在第 100 iter 自动退出**。观察约 100 iter，确认保存了 `iter_50.pth` 或 `iter_100.pth` 后，可用 Ctrl+C 结束自己的检查进程；若不停止，它会训练完整 1 epoch。手动中断出现 launcher 的终止提示不等同于模型计算失败。

检查的是四件事：真实数据能加载、loss 能计算、反向更新能执行、checkpoint 能保存。loss 与 grad_norm 应为有限数值，多个迭代能持续推进；不要求每一步 loss 单调下降。

`--no-validate` 只在这个短训练检查中使用，避免尚未形成有用权重就做完整验证；它不关闭反向训练，也不代表正式流程可以省略评估。

### 9.3 第一次迁移：8 卡完成一轮训练和验证

确认上一项进程已退出，且获准使用全部 8 个示例设备后执行：

~~~bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
PORT=29509 bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py 8 \
  --work-dir work_dirs/npu_check_8card_1epoch \
  --cfg-options total_epochs=1 runner.max_epochs=1 \
    data.workers_per_gpu=4 log_config.interval=10
~~~

这次**没有 --no-validate**。训练结束应保存 `epoch_1.pth`，并完成 6019 帧验证，输出 OCC 和检测指标。它一次检查多 worker、HCCL、整轮训练、保存和自动评估，不再拆成互不衔接的“冒烟”章节。

一轮只能证明功能闭环及初期训练行为，不能证明已充分收敛。这个 1 epoch 实验使用较短的学习率计划，权重不作为下面正式 24 epoch 复现的起点；正式实验使用独立目录，从同一 DLA 初始化权重开始。

### 9.4 正式 8 卡、24 epoch 训练指令

第 7 章已固定 24 epoch、每轮保存和每轮验证。下面命令不覆盖轮数、不跳过验证、不沿用短跑的按 iter 保存设置。

`RUN_DIR` 使用一个**新的、未存放旧实验结果**的目录名。使用既有训练继续跑时，应改用第 9.6 节，不要重新启动覆盖记录。

~~~bash
conda activate /data/apollo-npu-workspace/env2_apollo
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
RUN_DIR=work_dirs/npu_dla34_8card_24e_w4
mkdir -p "$RUN_DIR"
set -o pipefail

PORT=29509 bash tools/dist_train.sh \
  projects/configs/bevformer/bev_tiny_det_occ_apollo.py 8 \
  --work-dir "$RUN_DIR" \
  --seed 0 \
  --cfg-options data.workers_per_gpu=4 \
  2>&1 | tee "$RUN_DIR/train.log"
~~~

如已验证服务器每卡 8 worker 更合适，可把命令中的 4 改为 8，同时使用清楚标识 w8 的新实验目录。不要在同一精度对比实验中随意同时改 batch、学习率、分辨率、轮数和损失。

正式训练建议在服务器现有的持久会话工具中运行，避免 SSH 断开导致训练被终止。不要关闭已有的他人会话。

本配置初始学习率 2e-4、骨干学习率倍率 0.1、warmup 500 iter、梯度裁剪阈值 35。保留这些设置建立基线；NPU 利用率优化应在结果可复现后另做实验。

### 9.5 何时保存权重，保存多少份

默认 `checkpoint_config.interval=1` 且按 epoch 保存：

- 第 1 轮结束：`epoch_1.pth`。
- 第 2 轮结束：`epoch_2.pth`。
- 正常跑完第 24 轮：`epoch_24.pth`。
- `latest.pth` 指向或复制最近一次保存的 checkpoint，**不代表验证精度最好**。

默认没有限制保留数量时，完整 24 轮会有 24 个按轮编号的 checkpoint，另有 latest 的入口；不等于保存 25 个不同模型。若配置了 `max_keep_ckpts`，旧文件会被相应清理，数量就会不同。

判断是否跑完不能只看存在 latest.pth。要看最终日志轮数、对应 `epoch_24.pth` 和最后一轮验证输出。早期目录名 `npu_dla_8card_worker8` 只说明命名习惯，不包含轮数保证；究竟跑了几轮，应看该目录保存的配置和日志。

### 9.6 中断后继续训练

只对**原本就是 24 epoch 计划**的正式实验使用下面的续训命令：

~~~bash
conda activate /data/apollo-npu-workspace/env2_apollo
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
RUN_DIR=work_dirs/npu_dla34_8card_24e_w4
set -o pipefail

if test -f "$RUN_DIR/latest.pth"; then
  PORT=29509 bash tools/dist_train.sh \
    projects/configs/bevformer/bev_tiny_det_occ_apollo.py 8 \
    --work-dir "$RUN_DIR" \
    --resume-from "$RUN_DIR/latest.pth" \
    --seed 0 \
    --cfg-options data.workers_per_gpu=4 \
    2>&1 | tee -a "$RUN_DIR/resume.log"
else
  echo "没有找到续训权重，请先核对 RUN_DIR"
fi
~~~

`--resume-from` 恢复权重、优化器和训练进度；只加载模型权重不是完整续训。按 epoch 保存时，中途断电通常只能回到最近保存的一轮，而不是自动回到最后一条日志的迭代。

### 9.7 怎么看训练日志

先把训练损失与验证指标区分开：**loss 反映优化目标；验证集 mAP/mIoU 才用于判断对未参与更新的数据表现如何。**

主要训练字段：

- `loss_cls`：最终检测 Decoder 层的目标分类损失。
- `loss_bbox`：最终层的框回归损失；是加权训练量，不是直接以米表示的定位误差。
- `d0.loss_cls/d0.loss_bbox` 至 `d4.*`：前 5 个 Decoder 层的辅助监督，不是不同设备的损失。
- `loss_occupancy`：体素语义预测损失，当前配置为加权 FocalLoss。
- `loss_sem_scal`：语义类别层面的统计约束，辅助改善类别预测。
- `loss_geo_scal`：几何占用约束；当前配置权重为 0，日志为 0 不是漏训练。
- `loss_flow`：体素运动/flow 分支损失；当前配置未启用该训练输出，0 不表示速度预测完美。
- `lovasz_softmax`：可选的 Lovász 损失，当前未启用时输出 0。
- `loss`：训练器汇总用于优化的损失。各分项权重和尺度不同，不能直接按数值大小判断哪个任务更重要。
- `grad_norm`：梯度范数；用于观察异常放大/非有限值，不是准确率。

运行状态字段：

- `lr`：当前学习率；warmup 和余弦调度会改变它。
- `time`：日志统计窗口的平均迭代时间，不是本轮累计耗时。
- `data_time`：等待数据的时间指标，可帮助判断加载瓶颈，不代表全部 CPU 耗时。
- `memory`：框架统计的峰值已分配设备内存，和 npu-smi 的当前整卡占用不是同一口径。
- `eta`：估算剩余时间，受预热、验证、I/O 等影响，不是完工承诺。

不要把稀疏日志中的 `time` 简单累加当作全程耗时，也不要只看一次 npu-smi 截图判断性能。少量初始慢迭代正常；持续 NaN/Inf、worker 退出、通信超时或迭代长时间不前进才需要暂停自己的实验并保留完整日志。

### 9.8 查看曲线：重点保留验证指标

当前配置已启用 TextLoggerHook 和 TensorboardLoggerHook，不需要为看 loss 安装 TensorFlow。

服务器容器内执行：

~~~bash
tensorboard \
  --logdir /data/apollo-npu-workspace/Apollo-Vision-Net/work_dirs/npu_dla34_8card_24e_w4 \
  --host 127.0.0.1 --port 6006
~~~

在本地终端建立 SSH 转发，将 `服务器IP` 换成真实地址：

~~~bash
ssh -N -L 6006:127.0.0.1:6006 root@服务器IP
~~~

浏览器访问 `http://127.0.0.1:6006`。这里假定使用第 4 章 host 网络容器；其他网络模式要按实际端口映射处理。不必为了访问曲线把 TensorBoard 无认证暴露到公网。

实际查看旧实验时，把 logdir 指向**那个实验目录**，例如 `/data/apollo-npu-workspace/Apollo-Vision-Net/work_dirs/npu_dla_8card_worker8`。目录下只有 `tf_logs` 并不代表异常，应进一步确认其中有 `events.out.tfevents.*` 文件及 scalar 事件。TensorFlow 未安装的 reduced feature set 提示通常不影响这些标量曲线。

本地已有展示页面：

~~~text
D:\apollo-npu-workspace\apollo_training_dashboard\index.html
~~~

可用它读取服务器导出的 `.log.json`。此文件是一行一个记录，不是一个完整 JSON 数组；训练记录和 val 记录都应保留。验证行中的个别 NaN 需要按“该指标无有效值”展示，**不能把整行验证结果丢弃**。训练曲线点密集时用线图，验证点较少时用可缩放、点击查看数据的点线图。

验证指标通常要到首轮结束后才出现；不要在第 1 轮中途仅因没有 val 曲线就判断日志器失效。

## 10. 有权重后再推理与评估

推理是“加载权重得到预测”，评估是“将预测与标注比较得到指标”。它们不更新参数。Apollo 是视觉检测/OCC 模型，直接使用本仓测试脚本即可，不需要 vLLM。

训练期间的每轮自动验证已经在计算同类指标；独立测试用于检查某个指定 checkpoint、复核结果或导出展示材料。

### 10.1 先选择一个已完整保存的 checkpoint

建议明确指定 `epoch_N.pth`，而不是在训练保存的同时读取不断变化的 latest.pth。配置必须与权重对应；不能把 DLA 权重配给 R101 或其他任务配置。

只有早期实验的 latest.pth 也能评估，但结果代表它实际训练到的状态，不代表完整 24 epoch 成果。

### 10.2 先用 1 张 NPU 完整评估

选择空闲且获准的设备，下面以 0 为例。这里评估 `epoch_1.pth`；有其他权重时只调整 CKPT 的值：

~~~bash
conda activate /data/apollo-npu-workspace/env2_apollo
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
RUN_DIR=work_dirs/npu_dla34_8card_24e_w4
CKPT="$RUN_DIR/epoch_1.pth"
set -o pipefail

if test -f "$CKPT"; then
  PORT=29503 bash tools/dist_test.sh \
    projects/configs/bevformer/bev_tiny_det_occ_apollo.py "$CKPT" 1 \
    2>&1 | tee "$RUN_DIR/eval_epoch_1_1card.log"
else
  echo "权重不存在，请核对 CKPT 路径"
fi
~~~

这里的 1 表示启动一个分布式进程。**不要改成直接 python tools/test.py，也不要加 --out。** 第 7 章已解释并处理本路线的入口约束。

通过标准不是进度条到 6019/6019 就结束，而是其后依次出现：

1. `thre_0.25` 下的 OCC 分类 IoU、mIoU、IoU、Precision、Recall。
2. nuScenes 检测评估结果，包括 mAP、NDS 和各项误差。
3. 进程正常退出，没有后续 Traceback。

前向完成但评估异常属于“推理成功、评估未完成”，不能记为完整通过。

### 10.3 单卡通过后，再按需用 8 卡评估

只在 8 张设备均可用时执行，不要与正在占用相同设备的训练任务重叠：

~~~bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RUN_DIR=work_dirs/npu_dla34_8card_24e_w4
CKPT="$RUN_DIR/epoch_1.pth"
set -o pipefail

if test -f "$CKPT"; then
  PORT=29503 bash tools/dist_test.sh \
    projects/configs/bevformer/bev_tiny_det_occ_apollo.py "$CKPT" 8 \
    2>&1 | tee "$RUN_DIR/eval_epoch_1_8card.log"
else
  echo "权重不存在，请核对 CKPT 路径"
fi
~~~

多卡把样本分给不同进程，最后汇总结果。当前测试采样器给各进程分配连续片段，不能自行换成按索引交错分配的普通采样器，否则会改变历史 BEV 使用方式及结果顺序。

即使同一权重，单卡和多卡在片段起点的历史状态也可能不同；因此不承诺逐位一致。严肃比较精度时固定卡数、采样器、数据顺序、阈值和配置。确认多卡结果与单卡合理一致后，再将多卡作为常规评估方式。

### 10.4 去哪里找结果

在 Apollo 根目录运行测试后，检测输出位于：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/test/bev_tiny_det_occ_apollo/<本次测试时间>/pts_bbox/
~~~

重点保留：

- `results_nusc.json`：送入 nuScenes 评估器的检测预测。
- `metrics_summary.json`、`metrics_details.json`：整体和细分检测指标，以本次评估输出为准。
- 第 10.2/10.3 节保存的评估日志：其中同时包含 OCC 指标和检测指标。

训练中的自动验证输出通常位于 Apollo 根目录的 `val/` 下；确切位置看当次日志打印的路径。不要把不同 work_dir、不同 epoch 或不同测试时间的文件混在一起。

### 10.5 Occupancy 指标怎么读

本配置将空间划分为 `200 × 200 × 16` 个体素，每格 0.5 m，预测 16 类语义；没有被判为这 16 类的体素按空体素处理。指标只在评估实现规定的有效区域上统计。

- **miou/car、miou/driveable_surface 等单类项**：虽然打印前缀叫 miou，实际每一项是该类别自己的 IoU。公式为 TP / (TP + FP + FN)，同时惩罚多预测和漏预测。
- **miou**：16 个语义类别 IoU 的平均值，越高越好；不同类别对平均值各占一份，不是把所有体素简单计算一次正确率。
- **iou**：不区分类别，只判断空间是否被占用的 IoU；预测“车”为“行人”会影响语义 mIoU，但可能不影响这一项占用判断。
- **Precision**：预测为占用的体素中，有多少真的被占用。偏低意味着虚假占用较多。
- **Recall**：真实占用体素中，有多少被检出。偏低意味着漏掉较多物体/空间结构。
- **thre_0.25**：生成占用预测采用的分数阈值，不是 IoU 的匹配阈值。改变它会改变预测数量及 Precision/Recall，不能用不同阈值的结果直接比较。

本仓输出中，单类 IoU、mIoU、IoU 使用百分数形式，Precision/Recall 使用 0～1 小数。例如此前结果：

~~~text
miou: 13.91       → 13.91%
iou: 29.45        → 29.45%
Precision: 0.4024 → 40.24%
Recall: 0.5233    → 52.33%
~~~

这说明早期权重已有一定空间预测能力，但语义分类和空间覆盖都还有提升空间，不能据此断言模型已收敛。

16 类依次涵盖：小汽车、卡车、拖挂车、公交车、工程车辆、自行车、摩托车、行人、交通锥、隔离栏、可行驶路面、其他平面、人行道、地形、人造结构、植被。小目标与少样本类别低于道路等大面积类别并不罕见，应结合可视化和训练进展分析，而不是只看总 mIoU。

### 10.6 3D 检测指标怎么读

nuScenes 检测有 10 个目标类别，其 mAP 与 OCC 的 mIoU 不是同一种指标。

- **mAP，越高越好**：各类别平均精度的汇总，反映不同置信度下的检出与误检权衡。nuScenes 使用地面平面上的中心距离阈值进行匹配，而不是这里的 OCC IoU。日志若为 `0.3194`，换算为 31.94%。
- **NDS，越高越好**：结合 mAP 与定位、尺寸、朝向、速度、属性等误差的综合评分；同样常用 0～1 小数。
- **mATE，越低越好，单位 m**：匹配目标的位置平移误差。
- **mASE，越低越好，无量纲**：尺寸误差；可理解为对齐位置和朝向后，框的尺寸重合程度对应的误差。
- **mAOE，越低越好，单位 rad**：朝向误差，不是角度制；1 rad 约等于 57.3°。
- **mAVE，越低越好，单位 m/s**：速度误差。
- **mAAE，越低越好，无量纲**：属性分类误差，例如部分物体的运动状态等。

单类别 AP 可帮助定位哪类物体表现差；若输出 `AP_dist_0.5`、`AP_dist_1.0` 等，后缀是中心距离阈值，单位米。

**当前 DLA 配置不预测速度。** 第 7.6 节补零速度是为了符合评估输入格式，不是速度学习。由此 mAVE 及 NDS 不能按“完整速度预测模型”理解，基于速度推断的属性也会受影响。要研究带速度的模型属于新的配置与训练任务，不能仅将评估文件中的零改成别的数。

某些类别没有适用的速度/属性等指标，细分输出可能是 NaN。这不同于训练 loss 为 NaN；展示时保留该验证记录并标为“不适用/无有效值”。

### 10.7 怎样向 README 的公开精度靠近

本地 Apollo README 给出的该方法指标是：

- 3D 检测 mAP：**31.94%**。
- Occupancy mIoU：**21.87%**。

这些是参考目标，不是“命令运行成功”自动得到的结果。对齐时依次核对：

1. 使用相同 DLA 配置、深度预训练权重、完整 trainval 和同一版 OccGT。
2. 使用相同分辨率、历史帧设置、任务开关、全局 batch 与优化器设置。
3. 完成原 24 epoch 训练计划，检查每轮验证曲线，而不是只评估短跑权重。
4. 固定评估阈值、卡数和采样方式，明确比较的是哪个 epoch。
5. 如果训练正常但精度仍有差距，再对比 CPU/GPU/NPU 的相关算子、插值、dtype 和梯度，不随意删损失或改真值来“提高指标”。

24 epoch 是原训练预算，不保证恰好达到同一个数值。最优 checkpoint 要按事先选定的验证指标确定；检测与 OCC 最优可能出现在不同轮，报告时应注明“同一 checkpoint 的两项结果”还是“分别最佳”。

官方 BEVFormer R101 配置与 Apollo DLA 多任务模型不是同一模型，其结果可以帮助检查环境，但不能直接作为 Apollo 的数值对齐答案。

### 10.8 可视化属于结果展示，不是训练前置依赖

需要保存 Occupancy 预测和真值用于查看时，在已经通过的测试命令中追加：

~~~bash
--show_dir work_dirs/npu_dla34_8card_24e_w4/visual_epoch_1
~~~

参数名是 `--show_dir`，不是 `--show-dir`。这段是追加参数，不是独立运行的一条命令。当前实现保存排序后的前 5 个场景，输出按 `thre_0.25/<scene>/occ_gts`、`occ_preds` 等目录组织；它不自动生成完整 3D 动画。

Apollo 仓中的：

~~~text
/data/apollo-npu-workspace/Apollo-Vision-Net/tools/occ_visualization/visualize_occ_gt.py
~~~

是**真值可视化脚本**，不是自动加载 latest.pth 做推理的工具。其他 Mayavi 展示脚本还有本地路径、图形环境和图像输入要求，不能把“执行一次脚本”写成已完成适配的通用命令。

建议将需要展示的预测/真值拷到具备图形环境的独立环境查看；Mayavi/VTK 不作为 env2 训练依赖。当前索引没有 `surround_image_path` 时，导出过程不会凭空生成拼接相机图，组合 RGB/OCC 视频还需单独准备图像素材。

## 11. 完成判定与交付记录

### 11.1 整个计划是否完成

分开判断三个目标，避免把“能跑”和“精度复现”混为一谈：

- **迁移功能闭环**：能加载真实数据、前向、反向、多 worker、8 卡训练、保存权重，再加载权重完成 OCC 和检测评估。此前提供的服务器记录已支持这些步骤跑通。
- **完整训练完成**：指定正式目录确实完成 24 epoch，具有最终 checkpoint 和各轮日志。是否完成需要查看该次实验的证据，不能从环境配置或一个 latest.pth 推断。
- **精度目标完成**：同口径验证结果达到约定目标。需保留对应 checkpoint、配置、评估日志；不能仅以 loss 降低或设备利用率较高判定。

本次文档重构根据本地源码纠正了若干与服务器记录不一致的地方。**本地源码存在待核对项，不等于服务器一定没改；也不能因为服务器跑通，就把本地每个文件都标成已适配。**

### 11.2 保存能够复现的材料

在已跑通的服务器 env2 中，为当前实验保存完整环境清单：

~~~bash
cd /data/apollo-npu-workspace/Apollo-Vision-Net
RUN_DIR=work_dirs/npu_dla34_8card_24e_w4
python -m pip freeze > "$RUN_DIR/pip-freeze.txt"
conda list > "$RUN_DIR/conda-list.txt"
git rev-parse HEAD > "$RUN_DIR/apollo-commit.txt"
git diff > "$RUN_DIR/apollo-changes.patch"
git -C /data/apollo-npu-workspace/mmcv diff > "$RUN_DIR/mmcv-changes.patch"
git -C /data/apollo-npu-workspace/mmdetection_2.19 diff > "$RUN_DIR/mmdet-changes.patch"
git -C /data/apollo-npu-workspace/mmsegmentation_0.20 diff > "$RUN_DIR/mmseg-changes.patch"
git -C /data/apollo-npu-workspace/mmdetection3d_0.18.1 diff > "$RUN_DIR/mmdet3d-changes.patch"
~~~

同时记录其他仓库的提交号、DrivingSDK 补丁版本、镜像标签、设备型号和启动命令。git diff 不包含未跟踪的新文件，应另行保存这些文件；不要只留补丁就删除工作目录。

每个实验至少保留：生效配置、训练日志、验证日志、所选 checkpoint、环境清单、源码版本及改动。再使用本地展示页面汇总 loss 和验证曲线，就能将“怎么跑通”和“跑到了什么结果”清楚地区分。

### 11.3 本次重构的依据与边界

主要依据是本工作区中的 Apollo README/配置/训练与评估源码、DrivingSDK 的 BEVFormer 参考补丁、mmcv 的扩展加载和分布式实现，以及 mmdetection3d 0.18.1 的数据集和扩展导入代码。

本书的必需修改已经集中在第 6、7 章；第 8 章只做数据，第 9 章只做训练，第 10 章只做推理评估。原稿作为历史资料保留在同目录的 `backups/Apollo-Vision-Net_昇腾NPU适配指导书_v7_重构前备份.md`，不再作为当前执行流程。

适配成立的范围是本文固定的 DLA 配置和软件组合。增加 InternImage/DCNv3、点云分支、速度预测、其他精度模式或新的部署后端时，需要重新确认依赖和算子路径，不能直接套用“当前路线已通过”的结论。
