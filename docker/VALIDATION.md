# Apollo NPU FP32 Docker验证记录

验证日期：2026-10-20

## 镜像基础

- DrivingSDK：26.1.0
- CANN：9.1.0
- PyTorch：2.7.1
- torch_npu：2.7.1.post6
- Python：3.10
- 架构：aarch64
- 系统：openEuler 24.03

## 验证结果

- 三个预训练权重SHA256校验：通过
- torch_npu加载及NPU识别：通过
- NPU张量实算结果：通过
- MMCV动态库加载：通过
- MX Driving加载：通过
- Apollo配置读取：通过
- BEVFormer模型构建：通过
- 模型参数量：70,662,775
- 单卡真实数据训练一次迭代：通过

训练验证摘要：

- Epoch：[1][1/28130]
- batch size：1
- DataLoader workers：2
- NPU显存：10447 MB
- loss：46.6250
- grad_norm：34.1578
- 单步时间：69.184秒
- 数据时间：39.772秒

测试使用 runner.stop_iters=1 主动结束，因此 torchrun 最终返回退出码1并显示 STOP!!!，属于预期行为。
