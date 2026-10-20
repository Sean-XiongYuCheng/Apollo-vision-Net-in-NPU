# Apollo NPU FP32 组件版本清单

本仓库是 Apollo-Vision-Net 昇腾 NPU FP32 已验证环境的统一代码快照。

| 组件 | 来源分支 | 来源提交 | 说明 |
|---|---|---|---|
| Apollo-Vision-Net | release/fp32 | 540ad5e | Apollo FP32 发布基线 |
| DrivingSDK | master | ad53cd15 | 昇腾 DrivingSDK |
| mmcv | release/apollo-fp32 | ac5854e8 | BEVFormer MMCV 昇腾适配 |
| mmdetection_2.19 | release/apollo-fp32 | 6daf8d2c | 兼容 MMCV 1.7.2 |
| mmsegmentation_0.20 | release/apollo-fp32 | 1ee834ee | 兼容 MMCV 1.7.2 |
| mmdetection3d_0.18.1 | release/apollo-fp32 | 818b7260 | 修复无速度检测框评估 |

该版本只交付已验证的 FP32 路线，不包含后续 FP16 实验改动。
