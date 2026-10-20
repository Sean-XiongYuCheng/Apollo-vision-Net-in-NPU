# 性能对比工具

## 痛点分析

NPU算子优化需要稳定的性能数据作为依据。直接在函数前后计时，可能因NPU异步执行、首次初始化和Host调度等因素得到不准确的结果；如果优化前后的采集范围或计时口径不同，数据也无法直接比较。因此，需要使用统一的预热、同步、采样和对比流程。

## 工具能力概述

`ops_performance`是DrivingSDK提供的性能对比工具，用于采集算子或代码流程的性能，并比较优化前后或不同历史版本的性能变化。工具支持以下任务：

- 保存当前算子的Device耗时或Host端到端耗时；
- 对比优化前后的算子或代码流程；
- 将当前结果与一个或多个历史CSV比较；
- 根据退化阈值生成性能判定。

工具只负责性能采集与对比，不检查输出精度，也不会自动准备输入或恢复每轮运行状态。完成一次调用后，工具返回被测函数最后一次正式执行的结果，因此原测试可以继续验证输出。

## 后续内容概览

| 章节 | 内容说明 |
| --- | --- |
| [性能对比流程](#性能对比流程) | 说明Baseline和Performance两种工作模式、性能对比步骤及快速使用方法 |
| [性能测试准备](#性能测试准备) | 说明被测范围、Timer、采样次数和配置方式 |
| [基准性能采集（Baseline模式）](#基准性能采集baseline模式) | 说明如何采集并保存优化前的基准性能 |
| [优化结果对比（Performance模式）](#优化结果对比performance模式) | 说明如何采集优化后的性能并与Baseline结果对比 |
| [性能对比报告](#性能对比报告) | 说明CSV字段和性能判定结果 |
| [常见问题](#常见问题) | 说明采集范围和异常问题的排查方法 |
| [接口参数](#接口参数) | 集中列出调用接口和配置参数 |
| [注意事项](#注意事项) | 说明工具的使用边界 |

如果是第一次使用，建议按照[性能对比流程](#性能对比流程)、[性能测试准备](#性能测试准备)、[基准性能采集（Baseline模式）](#基准性能采集baseline模式)、[优化结果对比（Performance模式）](#优化结果对比performance模式)和[性能对比报告](#性能对比报告)的顺序阅读。已经完成基准采集的用户，可以直接进入[优化结果对比（Performance模式）](#优化结果对比performance模式)。

## 性能对比流程

性能对比工具提供两种工作模式：`baseline`用于保存优化前的基准性能，`performance`用于采集优化后的性能并生成对比结果。先了解两种模式的用途，有助于明确每次采集会生成基准数据还是对比报告。

性能快慢是相对结论。性能对比工具首先建立基准，再使用相同口径采集优化后的算子性能。只有两次采集采用相同输入、被测范围、计时指标和采样条件，对比结果才有意义。

一次完整的性能对比包含以下步骤：

1. 确定被测范围，提前准备不需要计入耗时的输入和环境。
2. 选择计时指标，确保优化前后的采集使用相同`timer`。
3. 使用`baseline`模式运行优化前的算子，生成基准CSV。
4. 使用`performance`模式在相同条件下运行优化后的算子。
5. 读取Performance CSV，判断性能提升或退化。

下图展示了数据文件之间的关系：算子优化前的实现通过`baseline`模式采集生成Baseline CSV，优化后的实现通过`performance`模式采集性能并读取该CSV文件进行性能比较，最终生成Performance CSV。图中的Baseline表示优化前的基准数据，Candidate表示优化后的待对比数据。

```text
优化前实现 → baseline模式采集Baseline性能 → 生成Baseline CSV
                                                    ↓
优化后实现 → performance模式采集Candidate性能 → 生成Performance CSV → 查看性能变化
```

### 快速使用

最常见的使用方式是先保存优化前的Baseline结果，再读取该结果对比优化后的算子。优化前后需要使用相同的`case_id`。

1. 使用Baseline模式采集优化前的算子，生成Baseline CSV。

    ```python
    from mx_driving.tools.ops_perf import ops_performance

    ops_performance(
        original_op,
        x_npu,
        _perf={
            "mode": "baseline",
            "case_id": "my_op_fp16",
        },
    )
    ```

2. 使用Performance模式采集优化后的算子，并通过`baseline_files`传入上一步生成的Baseline CSV路径。

    ```python
    output = ops_performance(
        optimized_op,
        x_npu,
        _perf={
            "mode": "performance",
            "case_id": "my_op_fp16",
            "baseline_files": ["path/to/baseline.csv"],
        },
    )
    ```

将`path/to/baseline.csv`替换为第一步生成的文件路径。运行后会生成Performance CSV，用于查看优化前后的耗时变化。需要调整计时和采样条件时，请继续阅读[性能测试准备](#性能测试准备)。

## 性能测试准备

要完成性能对比流程，首先需要准备性能测试。性能对比针对优化后的算子（下文称为Candidate）和优化前的算子（下文称为Baseline）。Candidate和Baseline只有在被测范围、计时指标和采样条件一致时才具备可比性。因此，在修改原调用之前，应依次确认要测量哪段代码、关注哪类耗时以及执行多少次采样，再通过`_perf`把这些条件写入调用；多个用例采用相同条件时，可以进一步使用`set_perf`复用配置。

### 被测范围

被测范围决定哪些操作会计入性能数据，常见范围包括单算子测试、多个算子组成的组合流程测试，以及包含前向和反向计算的端到端测试。其中，单算子测试通常只采集算子本身的耗时，因此输入准备和初始化应在`ops_performance`外完成；组合流程或端到端测试则需要将希望采集的完整流程封装到被测函数中。

确定范围时按以下步骤检查：

1. 明确目标是单算子、组合流程还是前反向过程。
2. 将需要重复测量的代码封装为一个callable。callable是可以被Python直接调用的对象，例如函数、方法或实现了`__call__`的类实例；具体封装代码请参见[基准性能采集（Baseline模式）](#基准性能采集baseline模式)中的前反向示例。
3. 将不属于被测范围的输入搬运和初始化移到调用外。
4. 确保每轮执行使用等价的输入和运行状态。

明确范围后，才能选择与优化目标一致的计时指标。

### 计时指标

Timer是性能对比工具的计时模式，用于指定正式采样时统计哪一段耗时。Kernel执行时间和接口端到端时间反映不同问题，错误选择`timer`可能导致采集数据正确但性能结论无效。Timer与被测范围不是一一对应关系：单算子、组合流程和端到端测试都可以选择`device`、`host`或`both`，区别在于统计设备侧Kernel耗时还是Host端到端耗时。

| `timer` | 统计范围 | 采集方式 | 推荐场景 |
| --- | --- | --- | --- |
| `device` | 每次调用产生的NPU Kernel时长之和 | NPU-only Level0 profiler | 分析Kernel或算子设备侧性能 |
| `host` | Python调用开始到NPU同步完成 | 同步配合`perf_counter_ns` | 分析单次接口端到端延迟 |
| `both` | 分别采集以上两种耗时 | 先完成Device采样，再执行Host采样 | 同时观察设备侧和端到端指标 |

`device`不包含Host下发和Kernel间空闲；`host`包含Python调用、任务下发、NPU执行和同步等待。

选择时可遵循以下原则：

- 优化Kernel实现时优先使用`device`。
- 优化Python调用、任务下发或单次接口延迟时使用`host`。
- 无法确定瓶颈位置或需要同时观察两项指标时使用 `both`。
- Baseline与Candidate始终使用相同的`timer`。

`timer="both"`会执行两组独立的正式采样，而不是在一次调用中同时计时。这样可以避免profiler干扰Host结果，但会增加执行次数。

### 采样次数

首次执行性能采集流程可能包含初始化和缓存建立，此时采集的性能不准确，因此需要先进行Warmup；正式采样次数`repeat`用于降低偶发抖动的影响。默认`warmup=5`、`repeat=5`，通常可以直接使用。

被测函数的调用次数如下：

| 计时方式 | 调用次数 |
| --- | --- |
| `device` 或 `host` | `warmup + repeat` |
| `both` | `warmup + 2 × repeat` |

对于耗时较短或波动较大的算子，可以适当增加`repeat`；对于会修改输入或累积梯度的函数，应先保证每轮状态一致。确定计时模式和采样次数后，需要通过`_perf`将这些配置传入单次性能采集。

### 单次配置

`_perf`是`ops_performance`的单次调用配置，用于控制当前用例的工作模式、计时模式和采样次数。单个用例应优先使用这种配置方式，便于从调用位置直接确认采样条件。下面以`my_op`表示被测函数、`x_npu`表示已准备好的输入，示例使用`baseline`模式采集Device耗时，在5次Warmup后执行10次正式采样：

```python
from mx_driving.tools.ops_perf import ops_performance

output = ops_performance(
    my_op,
    x_npu,
    _perf={
        "case_id": "my_op_fp16",
        "mode": "baseline",
        "timer": "device",
        "warmup": 5,
        "repeat": 10,
    },
)
```

示例中的配置含义如下：

| 配置 | 作用 | 示例效果 |
| --- | --- | --- |
| `case_id` | 标识当前测试用例 | CSV中使用`my_op_fp16`标识该行数据 |
| `mode` | 指定工作模式 | `baseline`表示保存当前基准性能 |
| `timer` | 指定计时模式 | `device`表示采集NPU Kernel耗时之和 |
| `warmup` | 指定预热次数 | 前5次调用不参与统计 |
| `repeat` | 指定正式采样次数 | 后10次调用用于计算平均耗时 |

这些参数只影响当前一次`ops_performance`调用，其他参数保持默认值。完成配置后，该调用会返回`my_op`最后一次正式执行的结果，并把平均耗时写入Baseline CSV。实际算子的完整改造方法请参见[基准性能采集（Baseline模式）](#基准性能采集baseline模式)。同一脚本存在多个配置相同的用例时，再使用公共配置减少重复代码。

### 公共配置

同一脚本包含多个性能用例时，可以使用`set_perf`提取重复的`mode`、`timer`、`warmup`和`repeat`配置。与`_perf`只影响单次调用不同，`set_perf`设置的配置会作为当前进程内后续调用的默认值。

两种配置方式的关系如下：

| 配置方式 | 生效范围 | 适用场景 | 优先级 |
| --- | --- | --- | --- |
| `_perf` | 当前一次调用 | 单个用例或用例特有配置 | 高 |
| `set_perf` | 当前进程内的后续调用 | 多个用例复用相同配置 | 低 |

下面将上一节的单次配置改为公共配置。两个用例复用相同的采样条件，只在各自的`_perf`中设置不同的`case_id`：

```python
from mx_driving.tools.ops_perf import ops_performance, set_perf

set_perf(
    mode="baseline",
    timer="device",
    warmup=5,
    repeat=10,
)

ops_performance(op_a, x_npu, _perf={"case_id": "case_a"})
ops_performance(op_b, y_npu, _perf={"case_id": "case_b"})

set_perf()
```

最终两个用例均使用`baseline`模式、Device计时、5次Warmup和10次正式采样。单次`_perf`中的同名配置优先级更高，例如`_perf={"case_id": "case_b", "repeat": 20}`会使`case_b`单独执行20次正式采样。示例末尾调用`set_perf()`清空公共配置，避免这些设置继续影响同一进程中的其他测试组。

以上两节说明了如何把采样条件传给工具。配置方式不会改变性能对比的先后顺序，首次使用时仍应先完成Baseline采集。

## 基准性能采集（Baseline模式）

Baseline是性能对比的参照值。首次使用时应先采集优化前的性能，这样既能确认测试环境和被测函数正常，也能生成后续比较所需的基准文件。本节只设置必需的用例标识，用于展示最小改造方式；需要调整采样条件时，可按照[单次配置](#单次配置)补充`_perf`。

假设原代码直接调用算子：

```python
src_npu = src.npu()
index_npu = index.npu()

output = mx_driving.scatter_add(src_npu, index_npu, dim=0)
```

改造时保留原函数和参数，只在外层增加`ops_performance`。输入准备在调用前完成，仅采集算子耗时，避免非算子开销影响采集结果。

按以下步骤采集Baseline：

1. 导入 `ops_performance`。
2. 将原函数作为第一个参数传入。
3. 将原位置参数和关键字参数依次传入。
4. 在 `_perf` 中设置能够区分测试场景的 `case_id`。
5. 运行脚本并查看生成的Baseline CSV。

以下代码为采集示例代码。`mode`的默认值为`baseline`，因此代码中可以省略该配置：

```python
from mx_driving.tools.ops_perf import ops_performance

src_npu = src.npu()
index_npu = index.npu()

output = ops_performance(
    mx_driving.scatter_add,
    src_npu,
    index_npu,
    dim=0,
    _perf={"case_id": "scatter_add_fp16"},
)
```

默认配置会执行5次Warmup和5次正式Device采样，并将平均耗时写入CSV。`output`是最后一次正式调用的结果，后续精度检查无需改变。完成本节后会得到Baseline CSV，下一步可将其用于对比优化后的实现。

如果需要采集多行逻辑，应先将完整流程封装为callable，确保工具重复执行的是同一测试单元。下面以前向和反向流程为例，每轮调用前使用`model.zero_grad(set_to_none=True)`清理梯度，避免梯度累积影响耗时和内存：

```python
def forward_backward(model, x):
    model.zero_grad(set_to_none=True)
    output = model(x)
    output.sum().backward()
    return output

output = ops_performance(
    forward_backward,
    model,
    x_npu,
    _perf={"case_id": "model_forward_backward"},
)
```

该示例把梯度清理纳入被测流程。如果不希望统计梯度清理耗时，可在被测函数中采用其他状态恢复方式，但必须保证Baseline和Candidate使用相同方式，并确保每轮调用前的梯度状态一致。

## 优化结果对比（Performance模式）

获得Baseline后，需要使用Performance模式采集优化后的Candidate性能。Performance模式包含两种配置：读取已保存的历史基准并与当前Candidate比较，或在同一脚本中依次采集Baseline和Candidate并直接比较。前者适合跨版本回归和CI，后者适合本地比较优化前后的两个算子实现。

| 对比条件 | 推荐方式 | 需要的配置 | 输出效果 |
| --- | --- | --- | --- |
| 已有稳定的Baseline CSV | 历史基准对比 | `mode="performance"`和`baseline_files` | 适合跨版本回归和CI |
| 没有基准文件，只需快速比较两个实现 | 同一脚本直接对比 | `mode="performance"`、`flag` 和相同 `case_id` | 适合本地优化验证 |

### 历史基准对比

历史文件对比适合持续集成和跨版本回归测试。它使用已保存的Baseline CSV，能够让当前Candidate与固定基准比较。

按以下步骤执行历史对比：

1. 使用Baseline模式保存优化前的结果。
2. 确保Candidate使用相同输入、Shape、dtype和运行环境。
3. 将一个或多个基线CSV路径传入`baseline_files`。
4. 设置 `details=True` 查看退化比例和判定结果。
5. 运行Candidate并查看Performance CSV。

以下示例中，`baseline_csv`是[基准性能采集（Baseline模式）](#基准性能采集baseline模式)生成的CSV路径。工具读取该文件，采集`optimized_op`并按5%的退化阈值生成详细报告：

```python
output = ops_performance(
    optimized_op,
    x_npu,
    _perf={
        "mode": "performance",
        "case_id": "my_op_fp16",
        "baseline_files": [baseline_csv],
        "threshold_pct": 5.0,
        "details": True,
    },
)
```

工具按`Case ID`和该Case的出现顺序查找历史记录。参数化测试多次使用相同`case_id`时，第N次调用会匹配基线中的第N条同名记录。多个基线文件均有匹配项时，每个文件生成一行比较结果。

### 同脚本性能对比

没有Baseline CSV时，可以在同一进程中快速比较两个实现。两次调用应使用相同的`case_id`和`timer`，并分别设置`baseline`与`candidate`标记。以下示例先登记优化前实现`old_op`，再登记优化后实现`new_op`；第二次调用完成后，工具会配对两侧数据并生成详细报告：

```python
ops_performance(
    old_op,
    x_npu,
    _perf={
        "mode": "performance",
        "flag": "baseline",
        "case_id": "my_op_fp16",
    },
)

ops_performance(
    new_op,
    x_npu,
    _perf={
        "mode": "performance",
        "flag": "candidate",
        "case_id": "my_op_fp16",
        "details": True,
    },
)
```

工具按调用脚本、`case_id`、`timer`和Rank配对，执行顺序不限。两侧完成后才写入报告；进程退出时仍缺少一侧，报告中会记录`INCOMPLETE_PAIR`。

## 性能对比报告

完成采样后，性能对比工具会将结果写入CSV。先定位报告，再查看耗时和判定字段，可以快速确认Candidate是否优于Baseline。

### 输出文件

未指定 `output_dir` 时，报告默认保存在调用脚本旁：

```text
{调用脚本目录}/performance_data/{调用脚本名称}/
```

例如，`tests/torch/test_scatter_add.py` 的默认目录为：

```text
tests/torch/performance_data/test_scatter_add/
```

文件名包含脚本名、模式和时间戳：

```text
{脚本名}_{baseline|performance}_{时间戳}[_rankN][_timer][_details].csv
```

其中，timer后缀只用于Performance文件，details后缀在开启详细输出时出现。

同一进程中，以下信息相同的结果会追加到同一个CSV，每个Case占一行：

- 调用脚本；
- 输出目录；
- 工作模式；
- Rank；
- 是否开启details；
- Performance模式使用的timer。

因此，一个测试脚本通常只生成少量报告，而不会为每个 `case_id` 单独创建文件。

### Baseline结果

Baseline报告保存当前性能，用于建立历史基准或单独分析耗时。表头固定包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `Case ID` | 被测用例标识 |
| `Device Time (us)` | 正式Device采样的平均耗时 |
| `Host + Device Time (us)` | 正式Host采样的平均端到端耗时 |

未选择的计时列留空。例如，默认Device计时的报告为：

```csv
Case ID,Device Time (us),Host + Device Time (us)
scatter_add_fp16,12.480,
```

设置 `details=True` 后额外输出 `Meta`，其中包含设备、版本、Git Commit、采样配置、统计信息以及用户提供的 `metadata`。这些信息有助于确认不同报告是否具备可比性。

### Performance结果

Performance报告记录Baseline、Candidate和性能变化。性能变化按照以下公式计算：

```text
Performance Comparison = (Baseline / Candidate - 1) × 100%
Regression = max(0, (Candidate / Baseline - 1) × 100%)
```

Performance Comparison为正表示Candidate加速，为负表示Candidate退化。两项指标公式不同，因此加速百分比和退化百分比不能直接互换。

Device对比的基本报告示例如下：

```csv
Case ID,Baseline Device Time (us),Candidate Device Time (us),Device Performance Comparison
scatter_add_fp16,12.480,10.000,+24.80%
```

设置`details=True`后，报告会增加Regression、Verdict、Baseline File和Meta。判定规则如下：

| 状态 | 含义 |
| --- | --- |
| `PASS` | 退化比例小于或等于 `threshold_pct` |
| `REGRESSION` | 退化比例大于 `threshold_pct` |
| `MISSING_BASELINE` | 历史文件中未找到匹配记录 |
| `INCOMPLETE_PAIR` | 进程内对比缺少一侧 |

默认`threshold_pct=5.0`。判定结果只写入详细报告，性能退化不会自动抛出异常；如果CI需要因此失败，应由上层流程读取报告并执行断言。

## 常见问题

本节用于处理两类常见问题：采集结果包含预期之外的非算子开销，以及采集过程出现异常或报告记录异常状态。其他输入状态和环境约束请参见[注意事项](#注意事项)。

### 非算子开销

如果输入准备位于被测函数内部，每次Warmup和正式采样都会重复执行相关操作，导致采集结果包含非算子开销。Python会先计算函数参数，再调用`ops_performance`，因此下面的采集方式会在正式采样前完成输入准备，仅测量单算子耗时：

```python
ops_performance(op, cpu_tensor.npu())
```

如果将`.npu()`放在被测函数内部，每次Warmup和正式采样都会执行输入准备，并将相关开销计入被测范围：

```python
def target(cpu_tensor):
    return op(cpu_tensor.npu())

ops_performance(target, cpu_tensor)
```

第二种采集方式适合测量包含输入准备的完整流程，但不应与仅测算子的Baseline直接比较。明确被测范围可以避免将优化收益或退化归因错误。

### 采集与报告异常

被测函数异常时，性能对比工具会先写入当前报告，再重新抛出原异常。设置`details=True`后，Meta中会记录异常类型、错误信息和Traceback；错误信息包含`out of memory`时，内部状态记为OOM。

以下情况由工具单独处理：

| 情况 | 处理方式 |
| --- | --- |
| `_perf` 配置无效 | 抛出 `OpsPerfUsageError` |
| `torch.npu`或profiler不可用 | 抛出`OpsPerfEnvironmentError` |
| 历史Case缺失 | 写入`MISSING_BASELINE` |
| 进程内配对不完整 | 进程退出时写入 `INCOMPLETE_PAIR` |
| 被测函数执行失败 | 保存报告后重新抛出原异常 |

异常报告用于辅助定位，不能替代对测试环境和被测函数的检查。

## 接口参数

本节集中列出接口和配置，便于已经了解使用流程的用户快速查阅。

### 调用接口

```python
result = ops_performance(function, *args, _perf=None, **kwargs)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `function` | 输入 | 被测函数或其他callable |
| `args`、`kwargs` | 输入 | 原样传给被测函数的参数 |
| `_perf` | 输入 | 性能对比工具配置字典，不会传给被测函数 |
| `result` | 输出 | 最后一次正式调用的函数返回值 |

`set_perf(_perf=None, **kwargs)` 用于设置进程级默认配置。单次调用配置的优先级高于全局配置。

### 配置参数

`_perf`参数配置可参考下表：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `case_id` | 被测函数名 | 非空用例标识 |
| `warmup` | `5` | Warmup次数，必须为非负整数 |
| `repeat` | `5` | 正式采样次数，必须为正整数 |
| `mode` | `baseline` | `baseline` 或 `performance` |
| `timer` | `device` | `device`、`host` 或 `both` |
| `details` | `False` | 是否输出判定、基线文件和Meta等详细字段 |
| `flag` | 空 | 进程内比较使用 `baseline` 或 `candidate` |
| `baseline_files` | 空 | 历史CSV路径或路径列表 |
| `threshold_pct` | `5.0` | 性能退化判定阈值 |
| `output_dir` | 自动 | 自定义输出目录 |
| `metadata` | 空 | 写入Meta的用户字典 |
| `quiet` | `False` | 是否关闭终端提示 |
| `synchronize_fn` | 自动 | 自定义同步函数，主要用于测试或特殊环境 |

`baseline_files`的优先级高于进程内`flag`配对逻辑。使用Performance模式时应明确选择一种对比方式，避免产生不符合预期的报告。

## 注意事项

性能采集前需要保证输入状态和运行环境一致，否则结果可能无法复现或比较。

- 性能对比工具不检查输出精度，性能采集后仍需单独验证计算结果；
- 工具不会自动复制或恢复输入，原地算子会使后续样本使用修改后的数据；
- 工具不会自动恢复随机状态，Baseline和Candidate应使用相同随机种子；
- 反向测试需要自行管理梯度，避免梯度累积影响耗时和内存；
- Baseline和Candidate应使用相同输入、Shape、dtype、硬件及软件环境；
- Baseline和Candidate应使用相同的timer、Warmup和repeat；
- 必要时设置`details=True`和`metadata`记录环境与用例信息；
- 对明显波动的结果应重复运行，避免依据单次数据得出结论；
- Host同步计时表示单次独立调用延迟，不代表异步流水场景下的吞吐率；
- Device时间是Kernel Duration之和，不包含Host下发开销和多Stream Kernel之间的空闲时间。
