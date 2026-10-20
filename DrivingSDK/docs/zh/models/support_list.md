# 模型清单

Driving SDK已适配自动驾驶感知、规控、端到端领域典型模型及部分主流VLA、世界模型等开源模型，模型清单分为以下两部分：

- **随版本演进的模型**：持续随CANN、TorchNPU及Driving SDK版本演进更新的模型，同步提供模型在不同产品上的最新性能数据。
- **不随版本演进的模型**：已日落不再随版本进行更新和维护的模型，但仍保留在仓库中供参考。

列表中Released为Y的模型表示已通过版本测试发布，为N的模型表示开发自验通过但未经过版本测试。表格仅展示各模型的部分典型性能数据供快速参考，如需了解模型详细介绍、使用指导及完整的训练性能与精度对比等信息，请点击表格中的模型名称跳转至对应README。

性能数据说明：

- **集群**：表示训练使用的硬件规模，格式为“节点数x卡数”，如1x8表示单机8卡，1x1表示单卡，2x8表示双机16卡。除特别标注外，性能数据均在单机8卡（1x8）环境下测得。
- **global batch size**：训练时每次迭代处理的总样本数。
- **精度格式**：训练使用的精度策略，分为FP32和Mixed（混合精度）两类。
- **性能**：默认单位为FPS（Frames Per Second，每秒处理帧数），数值越大性能越优；部分模型标注为迭代耗时（如s或ms），数值越小性能越优，已在性能值后标注单位。
- 竞品性能列中标注"竞品H"的表示该数据对应的竞品为竞品H，未标注的均为竞品A。

## 随版本演进的模型

<table>
    <tr>
        <td align="left">类别</td>
        <td align="left">模型</td>
        <td align="center" width="60">集群</td>
        <td align="center">global batch size</td>
        <td align="center" width="90">精度<br>格式</td>
        <td align="center">Atlas 800T A2性能<br><div align="center">（FPS）</div></td>
        <td align="center">Atlas 800T A3性能<br><div align="center">（FPS）</div></td>
        <td align="center">Atlas 950 SuperPOD性能<br><div align="center">（FPS）</div></td>
        <td align="center">竞品性能<br><div align="center">（FPS）</div></td>
        <td align="center">Released</td>
    </tr>
    <tr>
        <td rowspan="2">视觉感知</td>
        <td align="left"><a href="../../../model_examples/Sparse4D/README.md">Sparse4D</a></td>
        <td align="center">1x8</td>
        <td align="center">96</td>
        <td align="center">FP32</td>
        <td align="center">70.59</td>
        <td align="center">-</td>
        <td align="center">109.09</td>
        <td align="center">65.75</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DinoV3/README.md">DinoV3</a></td>
        <td align="center">1x8</td>
        <td align="center">512</td>
        <td align="center">Mixed</td>
        <td align="center">393.8</td>
        <td align="center">1144.1</td>
        <td align="center">-</td>
        <td align="center">616.8</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="1">点云感知</td>
        <td align="left"><a href="../../../model_examples/PointTransformerV3/README.md">PointTransformerV3</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">20.61</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">35.56</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="2">BEV感知</td>
        <td align="left"><a href="../../../model_examples/BEVFormer/README.md">BEVFormer</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">Mixed</td>
        <td align="center">3.33</td>
        <td align="center">-</td>
        <td align="center">11.64</td>
        <td align="center">3.75</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/BEVDet4D/README.md">BEVDet4D</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">7.04</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">5.59</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="1">融合感知</td>
        <td align="left"><a href="../../../model_examples/BEVFusion/README.md">BEVFusion</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">Mixed</td>
        <td align="center">32.78</td>
        <td align="center">-</td>
        <td align="center">75.76</td>
        <td align="center">26.59</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="2">Lane&Map</td>
        <td align="left"><a href="../../../model_examples/MapTR/README.md">MapTR</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">34.85</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">33.2</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/MapTRv2/README.md">MapTRv2</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">Mixed</td>
        <td align="center">23.03</td>
        <td align="center">56.25</td>
        <td align="center">54.1</td>
        <td align="center">21.91</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="3">规控</td>
        <td align="left"><a href="../../../model_examples/Diffusion-Planner/README.md">Diffusion-Planner</a></td>
        <td align="center">1x8</td>
        <td align="center">2048</td>
        <td align="center">FP32</td>
        <td align="center">5808.13</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">5304.32</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/HiVT/README.md">HiVT</a></td>
        <td align="center">1x8</td>
        <td align="center">256</td>
        <td align="center">FP32</td>
        <td align="center">645</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">652</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DriverAgent/README.md">DriverAgent</a></td>
        <td align="center">1x1</td>
        <td align="center">64</td>
        <td align="center">Mixed</td>
        <td align="center">140</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">149</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="7">端到端</td>
        <td rowspan="2" align="left"><a href="../../../model_examples/UniAD/README.md">UniAD</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td rowspan="2" align="center">8</td>
        <td rowspan="2" align="center">FP32</td>
        <td align="center">stage1: 1.002</td>
        <td align="center">stage1: 2.352</td>
        <td align="center">stage1: 2.500</td>
        <td align="center">stage1: 1.359</td>
        <td rowspan="2" align="center">Y</td>
    </tr>
    <tr>
        <td align="center">stage2: 1.554</td>
        <td align="center">stage2: 4.636</td>
        <td align="center">stage2: 4.545</td>
        <td align="center">stage2: 2</td>
    </tr>
    <tr>
        <td rowspan="2" align="left"><a href="../../../model_examples/SparseDrive/README.md">SparseDrive</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td align="center">stage1: 64</td>
        <td rowspan="2" align="center">Mixed</td>
        <td align="center">stage1: 46.3</td>
        <td align="center">stage1: 108.6</td>
        <td align="center">stage1: 101.3</td>
        <td align="center">stage1: 41.0</td>
        <td rowspan="2" align="center">Y</td>
    </tr>
    <tr>
        <td align="center">stage2: 48</td>
        <td align="center">stage2: 37.9</td>
        <td align="center">stage2: 88.4</td>
        <td align="center">stage2: 81.8</td>
        <td align="center">stage2: 35.2</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DiffusionDrive/README.md">DiffusionDrive</a></td>
        <td align="center">1x8</td>
        <td align="center">48</td>
        <td align="center">FP32</td>
        <td align="center">28.43</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">30.53</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="2" align="left"><a href="../../../model_examples/DiffusionDriveV2/README.md">DiffusionDriveV2</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td align="center">RL stage: 512</td>
        <td rowspan="2" align="center">Mixed</td>
        <td align="center">-</td>
        <td align="center">RL stage: 351.1</td>
        <td align="center">RL stage: 364.5</td>
        <td align="center">RL stage: 61.7</td>
        <td rowspan="2" align="center">N</td>
    </tr>
    <tr>
        <td align="center">SEL stage: 512/1024</td>
        <td align="center">-</td>
        <td align="center">SEL stage: 250.7</td>
        <td align="center">SEL stage: 283.2</td>
        <td align="center">SEL stage: 33.2</td>
    </tr>
    <tr>
        <td rowspan="8">VLA</td>
        <td align="left"><a href="../../../model_examples/OpenVLA/README.md">OpenVLA</a></td>
        <td align="center">1x8</td>
        <td align="center">256</td>
        <td align="center">FP32</td>
        <td align="center">56.14</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">73.12</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Pi-0.5/README.md">Pi-0.5</a></td>
        <td align="center">1x8</td>
        <td align="center">128</td>
        <td align="center">Mixed</td>
        <td align="center">-</td>
        <td align="center">155.1</td>
        <td align="center">165.0</td>
        <td align="center">70.8(竞品H)</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/GR00T-N1.5/README.md">GR00T-N1.5</a></td>
        <td align="center">1x8</td>
        <td align="center">512</td>
        <td align="center">FP32</td>
        <td align="center">337.35</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">276.38</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/GR00T-N1.6/README.md">GR00T-N1.6</a></td>
        <td align="center">1x8</td>
        <td align="center">640</td>
        <td align="center">FP32</td>
        <td align="center">449</td>
        <td align="center">-</td>
        <td align="center">870</td>
        <td align="center">457</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/GR00T-N1.7/README.md">GR00T-N1.7</a></td>
        <td align="center">1x8</td>
        <td align="center">640</td>
        <td align="center">FP32</td>
        <td align="center">-</td>
        <td align="center">836</td>
        <td align="center">885.2</td>
        <td align="center">444</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/StarVLA/README.md">StarVLA</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">-</td>
        <td align="center">123.08</td>
        <td align="center">-</td>
        <td align="center">49.61</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Lingbot-vla/README.md">Lingbot-vla</a></td>
        <td align="center">1x8</td>
        <td align="center">256</td>
        <td align="center">FP32</td>
        <td align="center">-</td>
        <td align="center">301.76</td>
        <td align="center">-</td>
        <td align="center">114.74(竞品H)</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/FastWAM/README.md">FastWAM</a></td>
        <td align="center">1x8</td>
        <td align="center">128</td>
        <td align="center">Mixed</td>
        <td align="center">-</td>
        <td align="center">123.63(SPS)</td>
        <td align="center">112.13(SPS)</td>
        <td align="center">60.53(SPS)(竞品A)</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="7">World Model</td>
        <td align="left"><a href="../../../model_examples/VGGT/README.md">VGGT</a></td>
        <td align="center">1x8</td>
        <td align="center">120</td>
        <td align="center">FP32</td>
        <td align="center">25.04</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">15.30</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Cosmos-Predict2/README.md">Cosmos-Predict2</a></td>
        <td align="center">2x8</td>
        <td align="center">16</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Cosmos-Transfer1/README.md">Cosmos-Transfer1</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">Mixed</td>
        <td align="center">10.34(s)</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">8.93(s)</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="2" align="left"><a href="../../../model_examples/Cosmos3/README.md">Cosmos3-nano</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td rowspan="2" align="center">-</td>
        <td rowspan="2" align="center">Mixed</td>
        <td align="center">-</td>
        <td align="center">reasoner: 25.15(s)</td>
        <td align="center">reasoner: 10.74(s)</td>
        <td align="center">reasoner: 26.69(s)</td>
        <td rowspan="2" align="center">Y</td>
    </tr>
    <tr>
        <td align="center">-</td>
        <td align="center">generator: 33.81(s)</td>
        <td align="center">generator: 20.71(s)</td>
        <td align="center">generator: 36.32(s)</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Cosmos-Drive-Dreams/README.md">Cosmos-Drive-Dreams</a></td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="2" align="left"><a href="../../../model_examples/Cosmos-Reason1/README.md">Cosmos-Reason1</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td align="center">SFT: 256</td>
        <td rowspan="2" align="center">-</td>
        <td align="center">SFT: 16.57(s)</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">SFT: 14.52(s)</td>
        <td rowspan="2" align="center">N</td>
    </tr>
    <tr>
        <td align="center">RL: 512</td>
        <td align="center">RL: 43.58(s)</td>
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">RL: 25.16(s)</td>
    </tr>
</table>

<br>

## 不随版本演进的模型

<table>
    <tr>
        <td align="left">类别</td>
        <td align="left">模型</td>
        <td align="center" width="60">集群</td>
        <td align="center">global batch size</td>
        <td align="center" width="90">精度<br>格式</td>
        <td align="center">Atlas 800T A2性能<br><div align="center">（FPS）</div></td>
        <td align="center">竞品性能<br><div align="center">（FPS）</div></td>
        <td align="center">Released</td>
    </tr>
    <tr>
        <td rowspan="9">视觉感知</td>
        <td align="left"><a href="../../../model_examples/CenterNet/README.md">CenterNet</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">Mixed</td>
        <td align="center">1257.44</td>
        <td align="center">542</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/FCOS/README.md">FCOS-resnet</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">196</td>
        <td align="center">196</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DETR/README.md">DETR</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">122</td>
        <td align="center">126</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Deformable-DETR/README.md">Deformable-DETR</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">63</td>
        <td align="center">65</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Mask2Former/README.md">Mask2Former</a></td>
        <td align="center">1x8</td>
        <td align="center">16</td>
        <td align="center">FP32</td>
        <td align="center">26.03</td>
        <td align="center">28.42</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/FCOS3D/README.md">FCOS3D</a></td>
        <td align="center">1x8</td>
        <td align="center">16</td>
        <td align="center">FP32</td>
        <td align="center">44.31</td>
        <td align="center">44.30</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DETR3D/README.md">DETR3D</a></td>
        <td align="center">1x8</td>
        <td align="center">24</td>
        <td align="center">FP32</td>
        <td align="center">14.35</td>
        <td align="center">14.28</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/StreamPETR/README.md">StreamPETR</a></td>
        <td align="center">1x8</td>
        <td align="center">16</td>
        <td align="center">FP32</td>
        <td align="center">26.016</td>
        <td align="center">25.397</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Yolov8/README.md">YOLOv8</a></td>
        <td align="center">1x8</td>
        <td align="center">-</td>
        <td align="center">FP32</td>
        <td align="center">214.64</td>
        <td align="center">479.73</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="5">点云感知</td>
        <td align="left"><a href="../../../model_examples/PointPillar/README.md">PointPillar</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">70.79</td>
        <td align="center">60.75</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/CenterPoint/README.md">CenterPoint(2D)</a></td>
        <td align="center">1x8</td>
        <td align="center">96</td>
        <td align="center">FP32</td>
        <td align="center">66.16</td>
        <td align="center">85.712</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/CenterPoint/README.md">CenterPoint(3D)</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">39.41</td>
        <td align="center">48.48</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/SalsaNext/README.md">SalsaNext</a></td>
        <td align="center">1x8</td>
        <td align="center">192</td>
        <td align="center">FP32</td>
        <td align="center">197.2</td>
        <td align="center">241.6</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Panoptic-PolarNet/README.md">Panoptic-PolarNet</a></td>
        <td align="center">1x1</td>
        <td align="center">2</td>
        <td align="center">FP32</td>
        <td align="center">1.28</td>
        <td align="center">1.69</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="5">BEV感知</td>
        <td align="left"><a href="../../../model_examples/BEVDet/README.md">BEVDet</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">73.81</td>
        <td align="center">37.16</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/BEVDepth/README.md">BEVDepth</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">Mixed</td>
        <td align="center">32.29</td>
        <td align="center">22.11</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/MatrixVT/README.md">MatrixVT</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">46.19</td>
        <td align="center">36.89</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="2" align="left"><a href="../../../model_examples/BEVNeXt/README.md">BEVNeXt</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td rowspan="2" align="center">64</td>
        <td rowspan="2" align="center">FP32</td>
        <td align="center">stage1: 16.568</td>
        <td align="center">stage1: 36.643</td>
        <td rowspan="2" align="center">N</td>
    </tr>
    <tr>
        <td align="center">stage2: 7.572</td>
        <td align="center">stage2: 11.651</td>
    </tr>
    <tr>
        <td rowspan="5">OCC感知</td>
        <td align="left"><a href="../../../model_examples/PanoOcc/README.md">PanoOCC</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">4.32</td>
        <td align="center">4.87</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/FlashOCC/README.md">FlashOCC</a></td>
        <td align="center">1x8</td>
        <td align="center">192</td>
        <td align="center">FP32</td>
        <td align="center">104.85</td>
        <td align="center">67.98</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/FBOCC/README.md">FB-OCC</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">20.80</td>
        <td align="center">33.61</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/TPVFormer/README.md">TPVFormer</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">6.69</td>
        <td align="center">10.32</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/SurroundOcc/README.md">SurroundOCC</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">7.59</td>
        <td align="center">7.78</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="2">Lane&Map</td>
        <td align="left"><a href="../../../model_examples/PivotNet/README.md">PivotNet</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">0.82(s)</td>
        <td align="center">0.58(s)</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/LaneSegNet/README.md">LaneSegNet</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">18</td>
        <td align="center">23.75</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="5">规控</td>
        <td align="left"><a href="../../../model_examples/QCNet/README.md">QCNet</a></td>
        <td align="center">1x8</td>
        <td align="center">32</td>
        <td align="center">FP32</td>
        <td align="center">75.29</td>
        <td align="center">94.11</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/MultiPath++/README.md">MultiPath++</a></td>
        <td align="center">1x1</td>
        <td align="center">128</td>
        <td align="center">FP32</td>
        <td align="center">856(ms)</td>
        <td align="center">646(ms)</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/GameFormer/README.md">GameFormer</a></td>
        <td align="center">1x8</td>
        <td align="center">4096</td>
        <td align="center">FP32</td>
        <td align="center">7501.8</td>
        <td align="center">6400</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/DenseTNT/README.md">DenseTNT</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">166</td>
        <td align="center">237</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/GameFormer-Planner/README.md">GameFormer-Planner</a></td>
        <td align="center">1x8</td>
        <td align="center">4096</td>
        <td align="center">FP32</td>
        <td align="center">5319</td>
        <td align="center">5185</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="1">端到端</td>
        <td align="left"><a href="../../../model_examples/VAD/README.md">VAD</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">4.121</td>
        <td align="center">7.048</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="3">LLM&VLM</td>
        <td align="left"><a href="../../../model_examples/LMDrive/README.md">LMDrive</a></td>
        <td align="center">1x8</td>
        <td align="center">16</td>
        <td align="center">FP32</td>
        <td align="center">8.02</td>
        <td align="center">13.85</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Senna/README.md">Senna</a></td>
        <td align="center">1x8</td>
        <td align="center">16</td>
        <td align="center">FP32</td>
        <td align="center">1.376</td>
        <td align="center">1.824</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/HPTR/README.md">HPTR</a></td>
        <td align="center">1x8</td>
        <td align="center">64</td>
        <td align="center">FP32</td>
        <td align="center">25.12</td>
        <td align="center">36.07</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="3">VLA</td>
        <td rowspan="2" align="left"><a href="../../../model_examples/Dexvla/README.md">Dexvla</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td rowspan="2" align="center">96</td>
        <td rowspan="2" align="center">Mixed</td>
        <td align="center">stage2: 16.72</td>
        <td align="center">stage2: 18.88</td>
        <td rowspan="2" align="center">Y</td>
    </tr>
    <tr>
        <td align="center">stage3: 15.85</td>
        <td align="center">stage3: 18.67</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/Pi-0/README.md">Pi-0</a></td>
        <td align="center">1x8</td>
        <td align="center">96</td>
        <td align="center">Mixed</td>
        <td align="center">116.36</td>
        <td align="center">136.17</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="3">World Model</td>
        <td align="left"><a href="../../../model_examples/OpenDWM/README.md">OpenDWM</a></td>
        <td align="center">1x8</td>
        <td align="center">2</td>
        <td align="center">FP32</td>
        <td align="center">1.82</td>
        <td align="center">1.82</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/MagicDrive-V2/README.md">MagicDrive-V2</a></td>
        <td align="center">1x8</td>
        <td align="center">4</td>
        <td align="center">FP32</td>
        <td align="center">stage1: 0.83</td>
        <td align="center">stage1: 1.5</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td align="left"><a href="../../../model_examples/NWM/README.md">Navigation World Models</a></td>
        <td align="center">1x8</td>
        <td align="center">96</td>
        <td align="center">FP32</td>
        <td align="center">363.39</td>
        <td align="center">383.06</td>
        <td align="center">N</td>
    </tr>
</table>

<br>
