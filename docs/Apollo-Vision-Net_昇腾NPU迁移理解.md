# Apollo-Vision-Net 昇腾 NPU 迁移理解

> 目标：在 Ascend 910C（Atlas A3）上，让 Apollo-Vision-Net 的 **DLA-34 + BEVFormer + 3D 检测 + Occupancy** 主线完成单卡推理、单卡训练，以及 8/16 卡训练。
>
> 本文汇总当前迁移讨论并重构为易读版本。它首先说明适配边界与证据，再说明后续动作。详细命令、逐行补丁和故障排查仍以《Apollo-Vision-Net 昇腾 NPU 适配指导书》为准。

---

## 1. 先记住结论

本项目的迁移方案**可行**，但支持范围与结论强度必须明确：

- **当前目标**：DLA-34 主线，配置为 `bev_tiny_det_occ_apollo.py`。
- **核心改造**：将 Apollo 中 CUDA 版的多尺度可变形注意力（MSDeformAttn）改为 `mx_driving.multi_scale_deformable_attn`。
- **暂不支持的路线**：InternImage / `*_intern_s.py` / DCNv3。
- **不做的事**：不为 NPU 编译 CUDA 扩展，不把所有 `.cuda()` 逐处改为 `.npu()`。

截至当前静态审查，DLA-34 的实际代码路径中，**没有发现 Apollo 新增且没有 NPU 等价实现的自定义 CUDA 算子**。这不是“已保证全模型可跑”，而是“尚不存在由缺少 CUDA 算子替代品导致的明确阻塞”。后续风险集中在设备语义、普通 PyTorch 算子支持、shape/dtype、反向传播和分布式运行。

一句话理解：

> 使用华为已适配好的 NPU 环境与算子库；保留 Apollo 的业务模型逻辑，只替换其 CUDA 专属算子并完成少量训练运行时适配。

---

## 2. 从底层到上层：整套系统各自做什么

```text
Ascend 910C NPU
  └─ 驱动 + CANN
      └─ 容器镜像（DrivingSDK A3）
          └─ PyTorch + torch_npu
              ├─ transfer_to_npu：多数 CUDA 语义自动兼容到 NPU
              ├─ mx_driving：NPU 高性能算子
              └─ mmcv：OpenMMLab 的编译型基础算子
                  └─ mmdet / mmseg / mmdet3d：检测、分割、3D 框架
                      └─ Apollo-Vision-Net：BEV、检测、占用业务模型
                          └─ DLA-34 配置：当前要验证和训练的主线
```

### 2.1 硬件与系统层

- **Ascend 910C**：实际执行计算的 NPU。
- **驱动 / 固件 / CANN**：让容器和 Python 程序能够使用 NPU。
- **DrivingSDK A3 镜像**：为 910C 准备的容器基础环境。必须使用带 `a3` 的镜像；不要将 910B 镜像混用到 910C。

这一层由官方维护，迁移时只做环境检查，不修改。

### 2.2 PyTorch 运行时层

- **PyTorch 2.7.1 + torch_npu 2.7.1.post6**：模型、张量、自动求导和 NPU 执行环境。
- **`transfer_to_npu`**：兼容层。导入后会将大量 `.cuda()`、`torch.cuda.*` 和分布式后端语义自动重定向到 NPU/HCCL。

因此，Apollo 中大部分 `.cuda()` 不需要逐个手改；但 NPU 并行类、分布式启动方式等不属于它的覆盖范围，仍要手工处理。

### 2.3 算子层

- **`mx_driving`**：DrivingSDK 的 NPU 原生算子库。当前最重要的是 `multi_scale_deformable_attn`。
- **`mmcv`**：OpenMMLab 的底层训练工具和编译型算子库，提供 `points_in_boxes_part`、`sigmoid_focal_loss` 等。

`mmcv` 需要以 `FORCE_NPU=1` 编译，因为它含有需要随硬件编译的二进制算子。`mmdet`、`mmseg`、`mmdet3d` 主要是 Python 框架代码，不需要各自编译 NPU 内核。

### 2.4 OpenMMLab 框架层

| 包 | 版本 | 在 Apollo 中的职责 |
| --- | --- | --- |
| `mmcv` | 1.7.2，NPU 编译版 | 基础训练工具与自定义算子 |
| `mmdet` | 2.19.0 | 2D 检测组件、训练基础设施 |
| `mmsegmentation` | 0.20.0 | 分割相关基础组件 |
| `mmdet3d` | 0.18.1 | 3D 检测、数据集、Runner |
| `timm` | 0.4.12 | 视觉模型的辅助依赖 |

三个旧版 OpenMMLab 包对旧 `mmcv` 有版本上限限制。必须放宽上限后，以源码 editable + `--no-deps` 安装，避免 pip 拉取旧 `mmcv-full` 覆盖已经编译好的 NPU 版 MMCV。

### 2.5 Apollo 应用层

Apollo-Vision-Net 的工作过程是：

```text
多相机图像
  → 图像骨干提取特征
  → BEV Transformer 融合为鸟瞰图特征
  → 3D 检测头输出目标框
  → Occupancy 头输出体素占用预测
```

当前迁移只针对 DLA-34 主线：

```text
多相机图像 → DLA-34 → BEVFormer → 检测 + Occupancy
```

---

## 3. DLA、InternImage、DCNv3 和 MSDeformAttn 不要混淆

### 3.1 DLA-34：当前使用的图像骨干

DLA-34 负责从每个相机图像中提取视觉特征。当前起步配置使用 DLA-34，不依赖 DCNv3。

### 3.2 InternImage：仓内另一条可选骨干路线

Apollo 是研究型仓库，保留了多种骨干网络。`*_intern_s.py` 配置会选择 InternImage，例如：

- `bev_tiny_occ_intern_s.py`
- `hybrid_tiny_occ_intern_s.py`

InternImage 的内部核心算子是 DCNv3。

### 3.3 DCNv3：当前不迁移

`ops_dcnv3` 是仓内的纯 CUDA 扩展，使用 `CUDAExtension` 和 `.cu` 源码编译。

DrivingSDK 有 `deform_conv2d` 与 `modulated_deform_conv2d`，但它们**不是 DCNv3 的一对一替代**：输入接口、张量布局、offset/mask 逻辑与运行限制不同。因此不能直接替换。

结论：

> 只要不选择 `*_intern_s.py`，DLA-34 主线不会调用 DCNv3；当前无需编译、修改或验证 DCNv3。

### 3.4 先理解“算子”是什么

算子是计算图中一个具体的张量计算动作，不等于一个完整模块：

```text
DLA、FPN、BEVFormer、Occupancy Head  = 模块（由许多算子组成）
Conv、MatMul、Softmax、MSDeformAttn   = 算子（实际执行的计算动作）
```

NPU 迁移时应把代码按来源分为三类：

```text
普通 PyTorch 算子
  例：torch.matmul、F.softmax、nn.Conv2d、torch.cat、reshape
  → 由 torch_npu 承接；不应机械替换

MMCV 扩展算子
  例：points_in_boxes_part、sigmoid_focal_loss
  → 由已用 FORCE_NPU=1 编译的 mmcv 提供；需确认实现可在 NPU 上执行

自定义 CUDA 扩展算子
  典型标记：CUDAExtension、.cu、load_ext、ext_module.xxx、torch.ops
  → 若目标路径会调用，必须有 NPU 等价实现或进行重写
```

此外，`.cuda()`、`device='cuda'`、`torch.cuda.*` 不是算子；它们是把代码绑定到 CUDA 的设备语义，同样需要审查。

### 3.5 MSDeformAttn：当前迁移必须处理的核心算子

MSDeformAttn 是 BEVFormer 中融合多相机特征的关键算子，与 DCNv3 完全不同。

MSDeformAttn 是 BEVFormer 中融合多相机特征的关键算子，与 DCNv3 完全不同。Apollo 当前 DLA-34 主线有三个调用点：

- [decoder.py](../Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/decoder.py)：检测 Decoder 的跨尺度可变形注意力；
- [spatial_cross_attention.py](../Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py)：相机特征投影到 BEV 的空间交叉注意力；
- [temporal_self_attention.py](../Apollo-Vision-Net/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py)：历史 BEV 与当前 BEV 的时序注意力。

三处都加载 MMCV 的 `_ext`，并在 CUDA 条件分支中调用：

```python
MultiScaleDeformableAttnFunction.apply(...)
```

该调用最终走 `ms_deform_attn_forward/backward` CUDA 扩展。迁移后，NPU 分支应改为：

```python
mx_driving.multi_scale_deformable_attn(...)
```

```python
mx_driving.multi_scale_deformable_attn(...)
```

静态扫描这三份文件的 `load_ext`、`ext_module`、`CUDAExtension`、`torch.ops`、`.cu` 等标记后，没有发现 Apollo 在这三份文件中额外加入第二种自定义 CUDA 扩展算子。因此，当前 DLA Transformer 主干中已识别的直接 CUDA 扩展需求只有 MSDeformAttn。

### 3.6 官方 patch：能证明什么，不能证明什么

DrivingSDK 提供的是针对官方 BEVFormer 某一基线版本的 **差异补丁**，不是 Apollo 当前文件的完整副本。因此，不能因为路径或文件名相同，就认定内容完全一致，也不应将整个 patch 直接应用到 Apollo。

对三个文件的逐文件 `git apply --check` 结果是：

```text
decoder.py                    可以干净匹配
spatial_cross_attention.py   可以干净匹配
temporal_self_attention.py   不能干净匹配
```

最后一个文件的导入区已有 Apollo 自己的演进（例如增加了 `torch.nn.functional as F`），故 patch 在上下文匹配时失败；这不表示 MSDeformAttn 的替换不可用，只表示不能整段套用。

正确结论是：

```text
文件完全相同                         × 不能证明，且已有反例
官方 patch 的 MSDeformAttn 替换逻辑   √ 三个调用点都适用
可将整个官方 patch 直接应用到 Apollo  × 不可取
Apollo 新增了另一种 CUDA 算子          × 当前三文件静态审查未发现
整个 Apollo DLA 已保证可运行           × 仍须后续分级验证
```

人工移植应只保留最小改动：为文件导入 `mx_driving`，并将 CUDA 分支中的 `MultiScaleDeformableAttnFunction.apply(...)` 替换为 `mx_driving.multi_scale_deformable_attn(...)`。保留原始 PyTorch 回退分支，不复制、删除或覆盖 Apollo 自有的其他逻辑。

### 3.7 DLA 路线的算子覆盖结论

```text
DLA-34 Backbone / SECONDFPNV2
  → 未发现 Apollo 自定义 CUDA 扩展
  → 普通 PyTorch / torch_npu 范畴

BEVFormer Encoder、Decoder、空间注意力、时序注意力
  → 发现 MSDeformAttn CUDA 扩展
  → mx_driving 有对应 NPU 实现
  → 官方 patch 提供了三个调用点的替换模式

Apollo Occupancy Head
  → 没有发现新的 Apollo CUDA 扩展
  → 依赖普通 PyTorch 与 MMCV 算子

points_in_boxes_part / sigmoid_focal_loss
  → MMCV 扩展算子，而非 Apollo 自写 CUDA 算子
  → 依赖已编译的 NPU MMCV；不能仅凭 import 成功就断言训练和反向完全无误
```

因此，本阶段可使用的范围结论是：

> DLA-34 路线没有发现“缺少 NPU 等价实现的 Apollo 自定义 CUDA 算子”这一阻塞项；但 Apollo Head、损失、MMCV 算子以及各动态分支不是官方 BEVFormer patch 自动覆盖的内容，必须独立管理。

---

## 4. 当前环境状态

env2 已通过以下验收：

- `torch 2.7.1+cpu`、`torch_npu 2.7.1.post6`、`mx_driving` 可导入；
- `mmcv 1.7.2`、`mmdet 2.19.0`、`mmseg 0.20.0`、`mmdet3d 0.18.1`、`timm` 可在同一进程导入；
- NPU 张量 `.npu()` 计算成功；
- `points_in_boxes_part`、`sigmoid_focal_loss` 和 `multi_scale_deformable_attn_pytorch` 可导入。

`torch==2.7.1+cpu` 是 Ascend 包的版本标识，不表示模型在 CPU 回退。NPU 张量已计算成功才是有效验证。

以下警告当前不构成阻断：

- `transfer_to_npu` 的替换提示；
- AMP API 的 FutureWarning；
- `mx_driving.fused` 的弃用提示。

注意：`transfer_to_npu` 会禁用 `torch.jit.script`，当前迁移和训练流程不要依赖 TorchScript。

---

## 5. Apollo 需要做的适配

### 5.1 算子适配

将三个 MSDeformAttn 调用点从 CUDA 扩展切换到 `mx_driving.multi_scale_deformable_attn`。

保留纯 PyTorch 回退路径，用于 CPU 对照和数值验证；不要删除与旧扩展相关的无害导入，只要主 NPU 路径不再调用 CUDA 实现即可。

### 5.2 设备与训练运行时适配

按官方 BEVFormer NPU patch 的思路逐项迁移：

1. `tools/train.py`、`tools/test.py` 提前导入 `torch_npu` 与 `transfer_to_npu`；
2. `MMDataParallel` / `MMDistributedDataParallel` 改为 `NPUDataParallel` / `NPUDistributedDataParallel`；
3. 分布式 backend 使用 `hccl`；
4. 旧 `torch.distributed.launch` 改为 `torchrun`；
5. DataLoader 使用 NPU 对应的 `pin_memory` 配置；
6. 修正 BEV Encoder 中的 NPU 张量缓存和部分矩阵运算；
7. 补齐配置中 BEV 层所需的参数；
8. 修正 torchvision 新版本下 rotate 的插值参数。

这些改动应逐项迁移和验证，不要对 Apollo 直接整体套用 BEVFormer patch；Apollo 已经有自己的检测头、占用头和数据逻辑。

---

## 6. 推荐执行顺序

### 阶段 A：冻结已验证的环境

保存 Conda 与 pip 快照，确保后续问题可复现：

```bash
conda activate /data/apollo-npu-workspace/env2_apollo
conda list --explicit > /data/apollo-npu-workspace/env2_apollo-spec.txt
pip freeze > /data/apollo-npu-workspace/env2_apollo-pip-freeze.txt
```

### 阶段 B：建立官方基准

在镜像自带 `bevformer` 环境中，用官方 BEVFormer 跑通单卡 50 iteration。它是环境、数据、NPU 和 DrivingSDK 的对照基线。

### 阶段 C：准备真实数据

准备 nuScenes、occupancy 标注和 temporal pkl。BEVFormer 时序模型需要真实的 `img_metas`、`can_bus` 等信息，不能以随机张量代替整图验证。

### 阶段 D：迁移最小代码范围

仅针对 `bev_tiny_det_occ_apollo.py` 与 DLA-34 主线，按第 5 节完成 MSDeformAttn 和运行时改动。

### 阶段 E：分级验收

1. **算子对齐**：比较 `mx_driving` NPU 版与 MMCV 纯 PyTorch 回退；FP32 最大绝对误差目标 `< 1e-3`；
2. **单卡前向**：使用真实验证集首样本，输出结构完整且无设备错误；
3. **单卡训练**：运行 50–100 iteration，确认 loss、反向传播与 checkpoint 保存；
4. **8 卡训练**：确认 HCCL、数据加载和 loss 稳定；
5. **16 卡训练**：最后扩展，观察吞吐、HBM 和收敛趋势。

### 阶段 F：多卡训练里程碑

Apollo 在 NPU 上采用**数据并行（DDP）**：每张 NPU 都保存完整模型，各自处理不同 mini-batch，再通过 HCCL 同步梯度。

多卡不是“把卡数改大”即可，以下条件必须同时满足：

1. **通信后端**：运行时 `dist_params` 使用 `backend='hccl'`，不能使用 CUDA 的 `nccl`；
2. **启动方式**：`tools/dist_train.sh` 和 `tools/dist_test.sh` 使用 `torchrun`，不能继续依赖已过时的 `torch.distributed.launch`；
3. **并行封装**：训练/测试代码使用 `NPUDataParallel` 或 `NPUDistributedDataParallel`，这不由 `transfer_to_npu` 自动替换；
4. **数据切分**：使用 `DistributedSampler`，确保不同 rank 获得不同训练样本；
5. **多任务分支**：当前检测 + Occupancy 配置保留 `find_unused_parameters=True`，避免条件分支造成 DDP 梯度同步报错；
6. **数据加载与日志**：DataLoader 使用 NPU 对应的 pin-memory 配置；仅 rank 0 写 checkpoint、主日志和评估结果。

总 batch size 会随卡数变化：

```text
总 batch size = 每卡 samples_per_gpu × NPU 卡数
```

因此扩卡后必须重新观察 loss 和显存，不应默认单卡学习率与多卡训练完全等价。

| 里程碑 | NPU 数 | 放行标准 |
| --- | ---: | --- |
| M1：单卡前向 | 1 | 真实样本前向成功，输出结构完整，无 CUDA/NPU 设备错误 |
| M2：单卡短训 | 1 | 50–100 iteration 可反向、loss 正常、checkpoint 可保存 |
| M3：8 卡短跑 | 8 | `torchrun` + HCCL 正常启动；无 rank 卡死；不同卡均有计算和数据读取 |
| M4：8 卡稳定训练 | 8 | loss 曲线稳定；吞吐、HBM 与 checkpoint 正常 |
| M5：16 卡短跑 | 16 | 扩容后无通信、Sampler、未使用参数或内存错误 |
| M6：16 卡正式训练 | 16 | 完整训练收敛，产出可用于正式推理与评估的 checkpoint |

> 固定顺序：单卡前向 → 单卡短训 → 8 卡短跑 → 8 卡稳定 → 16 卡短跑 → 16 卡正式训练。不要跳级。

---

## 7. 边界与风险清单

- 不要把“仓里存在 DCNv3”理解为“当前训练会用 DCNv3”。是否触发由配置决定。
- 不要选择 `*_intern_s.py`；它会进入 InternImage/DCNv3 路线。
- 不要让 pip 自动处理 OpenMMLab 依赖；它可能覆盖 NPU 编译版 MMCV。
- 不要跳过官方 BEVFormer 基准；它是后续定位问题的最短路径。
- 不要先上多卡；单卡真实样本前向和单卡反向必须先通过。
- 服务器上若发现 Git 对象损坏，应保留工作树后恢复或重新克隆 Git 元数据，避免无法 diff、回退和追踪补丁基线。

---

## 8. 最终范围声明

本阶段的成功定义是：

> 在 Ascend 910C 上，以 DLA-34 配置完成 Apollo-Vision-Net 的单卡推理、单卡训练和多卡训练；MSDeformAttn 使用 DrivingSDK 的 NPU 实现；InternImage/DCNv3 不属于本阶段交付范围。
