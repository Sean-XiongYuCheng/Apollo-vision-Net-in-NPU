# nms_rotated_dist[beta]

## 接口原型

```python
mx_driving.nms_rotated_dist(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor
```

## 功能描述

旋转框非极大值抑制（NMS），基于投影距离（projection distance）判定旋转框之间的重叠关系并剔除冗余框。

按`scores`从高到低排序后，依次以当前框为基准，将后续框中心点相对当前框中心的距离投影到当前框的局部坐标系（沿`angle`方向与垂直方向）上；当投影距离超出当前框的长宽范围时，认为两框不重叠、予以保留，否则抑制得分较低的框。相较于基于精确 IoU 的 NMS，该方式以投影距离近似重叠判定，计算开销更低，适合对精度要求不极端、追求性能的旋转框检测后处理场景。

## 参数说明

- `boxes (Tensor)`：旋转框张量，数据类型为`float32`。shape为`[N, 5]`。其中`5`分别代表`x_center, y_center, width, height, angle`，`x_center, y_center`代表box的中心点坐标，`width, height`代表box的长宽，`angle`代表box的弧度制旋转角。输入需在NPU上。
- `scores (Tensor)`：评分张量，数据类型为`float32`。shape为`[N]`。输入需在NPU上。
- `iou_threshold (float)`：抑制阈值。由于采用投影距离判定，当阈值小于`0.5`时对判定为重叠的框进行抑制；当阈值大于等于`0.5`时不进行抑制。

## 返回值

- `output (Tensor)`：NMS后保留框的索引张量，数据类型为`int64`，shape为`[K]`，其中`K`为保留框的数量，索引按`scores`从高到低的顺序返回。

## 约束说明

- `boxes`与`scores`必须在NPU设备上，否则会抛出`ValueError`。
- `boxes`的数据类型仅支持`float32`。
- `boxes`的最后一维必须为`5`。
- `boxes`与`scores`的第`0`维`N`必须一致。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>

## 调用示例

```python
import torch, torch_npu
from mx_driving import nms_rotated_dist
boxes = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5],
                      [0.0, 2.0, 2.0, 5.0, 0.3]], dtype=torch.float32).npu()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).npu()
keep = nms_rotated_dist(boxes, scores, 0.2)
```
