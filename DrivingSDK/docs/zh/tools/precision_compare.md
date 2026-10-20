# 精度对比工具

## 痛点分析

NPU与CPU执行相同计算时，结果可能因数据类型、计算顺序和硬件实现产生差异。直接使用完全相等判断，可能把正常的浮点舍入误差误判为错误；只查看单个最大误差，又无法判断少量异常是否影响整体结果。因此，需要同时比较误差大小和通过比例。

## 工具能力概述

`precision_compare`是DrivingSDK提供的精度对比工具，用于逐点比较NPU输出与CPU标杆结果。NPU输出是待验证的实际结果，CPU标杆结果是作为参照的结果，工具会据此计算绝对误差、相对误差和整体通过率。

工具支持NumPy数组、PyTorch Tensor、list、tuple和数值标量，可用于本地调试和自动化测试。比较前应确保两组数据元素数量相同，且相同位置的元素含义一致。

## 后续内容概览

[快速使用](#快速使用)说明如何准备数据并完成首次比较；[判定规则](#判定规则)解释通过和失败的判断方式；[参数配置](#参数配置)说明默认阈值和显式阈值；[结果处理](#结果处理)介绍输出阅读和失败定位；[接口参考](#接口参考)集中列出参数与返回字段。

如果是第一次使用，建议按照[快速使用](#快速使用)、[判定规则](#判定规则)、[参数配置](#参数配置)和[结果处理](#结果处理)的顺序阅读。已经熟悉比较流程的用户，可以直接进入[接口参考](#接口参考)查询参数和返回字段。

## 快速使用

首次使用时建议采用默认阈值，先确认输入和比较流程是否正确。只有两组数据的元素语义和位置一一对应时，误差才有意义，因此调用前应先准备NPU输出和CPU标杆结果，并确认两者元素数量相同。工具会将输入展平后比较。

按以下步骤准备数据：

1. 准备NPU输出和CPU标杆结果。
2. 确认两者包含相同数量的元素。
3. 确认展平后相同位置代表相同业务含义。
4. 如果原始shape也必须一致，在调用前自行检查shape。

完成数据准备后，使用默认阈值执行首次比较。工具会根据第一个参数`npu_pred`的dtype推断默认阈值，并打印比较详情。各数据类型对应的默认值请参见[参数配置](#参数配置)。

按以下步骤执行：

1. 调用`precision_compare`获取比较结果。
2. 读取`passed`判断是否通过。
3. 读取`pass_rate`了解整体通过比例。

```python
from mx_driving.tools.precision_compare import precision_compare

result = precision_compare(npu_pred, benchmark)
print(result.passed, result.pass_rate)
```

`passed`为`True`表示满足当前精度标准；为`False`时，可通过错误索引和误差字段进一步定位问题。

工具提供了一系列便捷接口，可根据调用场景选择。

| 接口 | 返回或失败行为 | 适用场景 |
| --- | --- | --- |
| `precision_compare` | 返回完整`CompareResult` | 查看统计和错误明细 |
| `assert_precision` | 返回`True`或`False` | 由调用方处理比较结果 |
| `check_precision` | 通过时返回`None`，失败时抛出`AssertionError` | 单元测试和CI |

例如，在自动化测试中使用`check_precision`，可以在精度未通过时直接终止测试：

```python
from mx_driving.tools.precision_compare import check_precision

check_precision(npu_pred, benchmark, verbose=False)
```

需要注意，`assert_precision`只返回布尔值，不会自动抛出异常。

## 判定规则

工具先判断每个数据点是否通过，再结合整体通过率和最大相对误差生成最终结果。理解这一过程，有助于解释结果并正确设置阈值。

对于浮点数据，满足以下任一条件即可通过单点检查：

- 绝对误差不大于`abs_threshold`；
- 相对误差不大于`err_threshold`。

对于整数数据，NPU输出必须与CPU标杆结果完全相同。

完成逐点检查后，工具按以下步骤判定整体结果：

1. 计算实际通过率`pass_rate`。
2. 检查实际通过率是否达到`(1 - err_ratio) × 100%`。
3. 对浮点数据检查最大相对误差；达到或超过`max_diff_hard`时直接判定失败。

其中，`err_threshold`控制单点误差，`err_ratio`控制允许未通过的数据比例，两者含义不同。

## 参数配置

默认阈值适合快速验证。默认值依据第一个参数`npu_pred`的dtype推断：

| 数据类型 | `err_threshold` | `err_ratio` | 最低通过率 |
| --- | --- | --- | --- |
| float16 | `1e-3` | `1e-3` | `99.9%` |
| 其他可转换浮点类型 | `1e-4` | `1e-4` | `99.99%` |
| 整数类型 | `0` | `0` | `100%` |

如果项目已有明确验收标准，应显式设置阈值，包括`err_threshold`和`err_ratio`。如果只指定`err_threshold`或`err_ratio`，另一个参数仍按dtype使用默认值。需要自定义标准时，建议同时设置两者：

```python
result = precision_compare(
    npu_pred,
    benchmark,
    err_threshold=1e-3,
    err_ratio=1e-3,
)
```

该配置允许最多`0.1%`的数据点未通过，并要求其余数据点满足设置的误差条件。

当标杆结果接近零时，相对误差容易被放大，可单独设置绝对误差阈值：

```python
result = precision_compare(
    npu_pred,
    benchmark,
    err_threshold=1e-4,
    err_ratio=1e-4,
    abs_threshold=1e-6,
)
```

这样既保留普通数值的相对误差约束，也能使用绝对误差判断近零值。`abs_threshold` 未设置时，默认等于 `err_threshold`。

## 结果处理

默认情况下，工具会先输出逐点比较数据，再输出阈值、通过率和整体结论。下面的示例表示3个数据点均满足精度要求，整体结果为`Pass`：

```text
+=====================================================================+
| total_count:3;                                                      |
+---------------------------------------------------------------------+
|     Loop     ExpectOut       RealOut        FpDiff      RateDiff    |
|---------------------------------------------------------------------|
| 00000001     1.0000000     1.0000000     0.0000000     0.0000000   |
| 00000002     2.0000000     2.0005000     0.0005000     0.0002500   |
| 00000003     3.0000000     3.0001000     0.0001000     0.0000333   |
+---------------------------------------------------------------------+
|  DiffThd        PctThd        PctRlt    Result                      |
|---------------------------------------------------------------------|
|   0.0010         99.90%   100.000000%      Pass                     |
+=====================================================================+
```

示例中的逐点字段用于查看每个位置的CPU标杆结果、NPU输出和误差，汇总字段用于判断整体是否满足精度要求。各字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `DiffThd` | 相对误差阈值 |
| `PctThd` | 要求的最低通过率 |
| `PctRlt` | 实际通过率 |
| `Result` | 整体结论 |
| `ExpectOut` | CPU标杆结果 |
| `RealOut` | NPU输出 |
| `FpDiff` | 绝对误差 |
| `RateDiff` | 相对误差 |

阅读输出时，先查看`Result`获取整体结论，再比较`PctRlt`与`PctThd`，最后根据`Loop`和逐点误差定位具体数据。

### 精度未通过的输出示例

精度未通过时，不要先放宽阈值，应先确认输入对应关系，再根据汇总信息和错误点判断偏差来源。输出除了逐点比较表和`Failed`汇总行，还会追加最大相对误差和错误行。例如，下面共有3个数据点，其中第3个点未通过：

```text
+=====================================================================+
| total_count:3;                                                      |
+---------------------------------------------------------------------+
|     Loop     ExpectOut       RealOut        FpDiff      RateDiff    |
|---------------------------------------------------------------------|
| 00000001     1.0000000     1.0000000     0.0000000     0.0000000   |
| 00000002     2.0000000     2.0005000     0.0005000     0.0002500   |
| 00000003     3.0000000     3.1000000     0.1000000     0.0322581   |
+---------------------------------------------------------------------+
|  DiffThd        PctThd        PctRlt    Result                      |
|---------------------------------------------------------------------|
|   0.0010         99.90%    66.666667%    Failed                     |
|---------------------------------------------------------------------|
| Maximum relative error is: 0.0322581.                              |
| Error Line                                                         |
|---------------------------------------------------------------------|
|     Loop     ExpectOut       RealOut        FpDiff      RateDiff    |
|---------------------------------------------------------------------|
| 00000003     3.0000000     3.1000000     0.1000000     0.0322581   |
+=====================================================================+
```

阅读失败输出时先看`Result`，再比较`PctRlt`和`PctThd`，最后根据`Error Line`定位未通过的数据点。

比较失败时，工具会输出错误行。也可以直接读取返回值中的错误信息：

```python
result = precision_compare(npu_pred, benchmark, verbose=False)

if not result.passed:
    print(result.error_indices)
    print(result.error_benchmarks)
    print(result.error_predictions)
    print(result.error_rel_diffs)
```

`error_indices` 是从零开始的展平索引，适合定位原始数据。批量测试时设置 `verbose=False`，可以关闭控制台详情但保留完整返回结果。

精度未通过时，建议按以下顺序检查：

1. 确认输入的元素数量、原始形状和元素顺序一致。
2. 检查数据类型和实际使用的阈值。
3. 查看NaN、Inf和错误索引是否集中在特定位置。
4. 对比最大绝对误差和最大相对误差，判断偏差特征。
5. 根据既定精度标准调整参数，避免仅为通过测试而放宽阈值。

### NaN或Inf输出示例

任一输入包含NaN或Inf时，工具会在逐点数据前输出警告，并分别统计NPU输出和CPU标杆结果中的特殊值数量：

```text
+=====================================================================+
| total_count:4;                                                      |
+---------------------------------------------------------------------+
| Warning: npu_pred contains 1 NaN, 2 Inf; benchmark contains 0 NaN, 1 Inf. |
+---------------------------------------------------------------------+
|     Loop     ExpectOut       RealOut        FpDiff      RateDiff    |
|---------------------------------------------------------------------|
| ...                                                                 |
```

警告表示输入中存在特殊值，不等同于整体结果一定失败。NaN和Inf使用单独规则处理：相同位置的NaN、同符号Inf视为通过，其他不匹配情况视为未通过。若业务不允许特殊值，应额外检查返回值中的NaN和Inf数量是否为零。

## 接口参考

本节汇总参数和返回字段，便于日常查阅。

### 参数说明

```python
precision_compare(
    npu_pred,
    benchmark,
    err_threshold=None,
    err_ratio=None,
    *,
    abs_threshold=None,
    max_diff_hard=0.1,
    display_lines=30,
    verbose=True,
)
```

| 参数 | 是否必选 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| `npu_pred` | 是 | NumPy数组、PyTorch Tensor、list、tuple或数值标量 | NPU输出 |
| `benchmark` | 是 | NumPy数组、PyTorch Tensor、list、tuple或数值标量 | CPU标杆结果 |
| `err_threshold` | 否 | float或int | 相对误差阈值，省略时根据`npu_pred`的dtype推断 |
| `err_ratio` | 否 | float或int | 允许未通过的数据比例，范围为`[0, 1]` |
| `abs_threshold` | 否 | float或int | 绝对误差阈值，默认等于`err_threshold` |
| `max_diff_hard` | 否 | float或int | 最大相对误差硬阈值，默认`0.1` |
| `display_lines` | 否 | int | 详情最大显示行数，默认`30`，最小为`2` |
| `verbose` | 否 | bool | 是否打印详情，默认`True` |

工具会将输入展平后比较，因此仅强制要求元素数量相同。为避免比较错位，建议调用前同时检查原始shape。PyTorch Tensor会分离计算图、移至CPU后转换为NumPy数组。

### 返回值说明

`precision_compare` 返回 `CompareResult` 命名元组。

| 字段 | 数据类型 | 说明 |
| --- | --- | --- |
| `passed` | bool | 是否通过 |
| `pass_rate` | float | 实际通过率，范围为`0`到`100.0` |
| `max_abs_err` | float | 最大绝对误差 |
| `max_rel_err` | float | 最大相对误差 |
| `mean_abs_err` | float | 平均绝对误差 |
| `mean_rel_err` | float | 平均相对误差 |
| `error_indices` | ndarray | 未通过点的展平索引 |
| `error_benchmarks` | ndarray | 未通过点的CPU标杆结果 |
| `error_predictions` | ndarray | 未通过点的NPU输出 |
| `error_abs_diffs` | ndarray | 未通过点的绝对误差 |
| `error_rel_diffs` | ndarray | 未通过点的相对误差 |
| `nan_count_pred` | int | NPU输出中的NaN数量 |
| `nan_count_bench` | int | CPU标杆结果中的NaN数量 |
| `inf_count_pred` | int | NPU输出中的Inf数量 |
| `inf_count_bench` | int | CPU标杆结果中的Inf数量 |
| `total_count` | int | 总数据点数 |
