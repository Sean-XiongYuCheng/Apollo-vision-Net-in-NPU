# 版本说明

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>Driving SDK</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers=><p>26.1.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>正式版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年7月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> 有关Driving SDK的版本维护，具体请参见[分支维护策略](../../../README.md#driving-sdk-分支维护策略)。

### 相关产品版本配套说明

**表 1**  Driving SDK软件版本配套表

|Driving SDK代码分支名称|CANN版本|TorchNPU版本|Python版本|PyTorch版本|
|--|--|--|--|--|
|master（在研版本）|在研版本|在研版本|3.9, 3.10, 3.11|2.7.1|
|branch_v26.1.0|9.1.0|26.1.0|3.9, 3.10, 3.11|2.7.1|
|branch_v26.0.0|9.0.0|26.0.0|3.9, 3.10, 3.11|2.7.1|

>[!NOTE]
>用户可根据需要选择Driving SDK代码分支下载源码并进行安装。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2**  Driving SDK与TorchNPU版本兼容

<table style="table-layout: fixed; width: 750px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Driving SDK</th>
      <th colspan="4">TorchNPU版本</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表 3**  Driving SDK与CANN版本兼容

<table style="table-layout: fixed; width: 750px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Driving SDK</th>
      <th colspan="4">CANN版本</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

- Driving SDK全量算子支持<term>Ascend 950PR&950DT系列产品</term>
- 自动驾驶模型BEVFormer、BEVFusion、Sparse4D、MapTRv2、UniAD、SparseDrive支持 <term>Ascend 950PR&950DT系列产品</term>
- 新增GR00T-N1.6模型
- 新增LingBot-VLA模型
- 新增StarVLA模型
- VLA模型GR00T-N1.6、Pi0.5支持<term>Ascend 950PR&950DT系列产品</term>

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[软件安装](../installation/installation.md)》|指导用户在NPU上完成Driving SDK的安装，内容涵盖硬件与操作系统兼容性说明、软件版本配套说明、驱动固件安装，以及三种软件安装的方式：镜像安装、在线安装、源码安装。帮助用户快速搭建Driving SDK开发环境。|-|
|《[快速入门](../get_started/quick_start_guide.md)》|以`scatter_max`算子和BEVFusion模型为例，指导初次接触Driving SDK的开发者完成NPU上的高性能算子调用和模型训练任务，帮助用户快速上手自动驾驶模型分布式训练。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00|2026-07-06|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101158|2026-07-06|无病毒，无恶意|

### 漏洞修补列表

无
