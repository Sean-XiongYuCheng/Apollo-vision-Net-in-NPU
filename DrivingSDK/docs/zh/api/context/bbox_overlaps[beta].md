# bbox_overlaps[beta]

## 接口原型

```python
mx_driving.npu_bbox_overlaps(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool is_aligned=False, float eps=1e-6) -> Tensor
```

## 功能描述

计算两个边界框的IoU、IoF或GIoU。

## 参数说明

- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 4]`。其中`4`分别代表`x1, y1, x2, y2`，表示box左上角和右下角的坐标。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 4]`。其中`4`分别代表`x1, y1, x2, y2`，表示box左上角和右下角的坐标。
- `mode (str)`：取值为`"iou"`时，计算IoU（intersection over union）；取值为`"iof"`时，计算IoF（intersection over foreground）；取值为`"giou"`时，计算GIoU（generalized intersection over union）。默认值为`"iou"`。
- `is_aligned (bool)`：取值为`True`时，只计算配对的box之间的结果；取值为`False`时，计算每对box之间的结果。默认值为`False`。
- `eps (float)`：加在分母上以保证数值稳定性的小值。默认值为`1e-6`。

## 返回值

- `ious (Tensor)`：包含两组bounding boxes的IoU、IoF或GIoU的张量，数据类型为`float32`。shape为`[M]`（`is_aligned=True`）或`[M, N]`（`is_aligned=False`）。

## 约束说明

- `mode`的取值范围为`{'iou', 'iof', 'giou'}`。
- `boxes_a`和`boxes_b`的维度必须至少为2-D。
- `boxes_a`和`boxes_b`最后一维必须为4。
- `boxes_a`和`boxes_b`的batch维度必须一致。
- 当`is_aligned=True`时，`M`必须等于`N`。
- 当`is_aligned=False`时，`boxes_a`数量`M`与`boxes_b`数量`N`的乘积不超过9亿。

## 支持的型号

- <term>Atlas A2训练系列产品</term>
- <term>Atlas A3训练系列产品</term>
- <term>Ascend 950PR&950DT系列产品</term>

## 调用示例

```python
import torch, torch_npu
from mx_driving import npu_bbox_overlaps
boxes_a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=torch.float32).npu()
# 非对齐模式，shape 为 [2, 2]
ious = npu_bbox_overlaps(boxes_a, boxes_b, mode="iou", is_aligned=False)
# 对齐模式，shape 为 [2]
ious = npu_bbox_overlaps(boxes_a, boxes_b, mode="giou", is_aligned=True)
```
