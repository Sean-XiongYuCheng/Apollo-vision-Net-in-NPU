# Supported Models

Driving SDK has adapted typical open-source models in the autonomous driving perception, planning and control, and end-to-end domains, as well as some mainstream VLA and world models. The model list is divided into the following two parts:

- **Models that evolve with versions**: Models that are continuously updated as the CANN, TorchNPU, and Driving SDK versions evolve, with the latest performance data of the models on different products provided accordingly.
- **Models that do not evolve with versions**: Models that have been sunset and are no longer updated or maintained as versions evolve, but remain in the repository for reference.

In the `Released` column, `Y` means that the model has passed release testing and has been released, and `N` means that the model has passed internal validation but has not gone through release testing. The tables show only some typical performance data of each model for quick reference. For detailed model introductions, usage guides, and complete training performance and accuracy comparisons, click the model name in the table to go to the corresponding README.

Performance data notes:

- **Cluster**: Indicates the hardware scale used for training, in the format "number of nodes x number of devices". For example, 1x8 indicates a single machine with 8 devices, 1x1 indicates a single device, and 2x8 indicates two machines with 16 devices. Unless otherwise noted, performance data is measured in a single-machine 8-device (1x8) environment.
- **Global Batch Size**: The total number of samples processed in each iteration during training.
- **Precision Format**: The precision policy used for training, which is divided into two types: FP32 and Mixed (mixed precision).
- **Performance**: The default unit is Frames Per Second (FPS), and a larger value indicates better performance. Some models are marked with iteration time (for example, s or ms), and a smaller value indicates better performance. The unit is marked after the performance value.
- In the Competitor Performance column, data marked "competitor H" indicates that the corresponding competitor is competitor H, and unmarked data corresponds to competitor A.

## Models That Evolve with Versions

<table>
    <tr>
        <td align="left">Category</td>
        <td align="left">Model</td>
        <td align="center" width="60">Cluster</td>
        <td align="center">Global Batch Size</td>
        <td align="center" width="90">Precision</td>
        <td align="center">Atlas 800T A2 Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Atlas 800T A3 Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Atlas 950 SuperPOD Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Competitor Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Released</td>
    </tr>
    <tr>
        <td rowspan="2">Visual perception</td>
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
        <td align="center">-</td>
        <td align="center">-</td>
        <td align="center">616.8</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="1">Point cloud perception</td>
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
        <td rowspan="2">BEV perception</td>
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
        <td rowspan="1">Fusion perception</td>
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
        <td rowspan="2">Lane &amp; map</td>
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
        <td rowspan="3">Planning and control</td>
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
        <td rowspan="5">End-to-End</td>
        <td rowspan="2" align="left"><a href="../../../model_examples/UniAD/README.md">UniAD</a></td>
        <td rowspan="2" align="center">1x8</td>
        <td rowspan="2" align="center">8</td>
        <td rowspan="2" align="center">FP32</td>
        <td align="center">stage1: 1.002</td>
        <td align="center">-</td>
        <td align="center">stage1: 2.350</td>
        <td align="center">stage1: 1.359</td>
        <td rowspan="2" align="center">Y</td>
    </tr>
    <tr>
        <td align="center">stage2: 1.554</td>
        <td align="center">-</td>
        <td align="center">stage2: 3.333</td>
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
        <td rowspan="6">VLA</td>
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
        <td align="center">70.8(competitor H)</td>
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
        <td align="center">114.74(competitor H)</td>
        <td align="center">Y</td>
    </tr>
    <tr>
        <td rowspan="6">World model</td>
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

## Models That Do Not Evolve with Versions

<table>
    <tr>
        <td align="left">Category</td>
        <td align="left">Model</td>
        <td align="center" width="60">Cluster</td>
        <td align="center">Global Batch Size</td>
        <td align="center" width="90">Precision</td>
        <td align="center">Atlas 800T A2 Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Competitor Performance<br><div align="center">(FPS)</div></td>
        <td align="center">Released</td>
    </tr>
    <tr>
        <td rowspan="9">Visual perception</td>
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
        <td rowspan="5">Point cloud perception</td>
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
        <td rowspan="5">BEV perception</td>
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
        <td rowspan="5">OCC Perception</td>
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
        <td rowspan="2">Lane &amp; map</td>
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
        <td rowspan="5">Planning and control</td>
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
        <td rowspan="1">End-to-End</td>
        <td align="left"><a href="../../../model_examples/VAD/README.md">VAD</a></td>
        <td align="center">1x8</td>
        <td align="center">8</td>
        <td align="center">FP32</td>
        <td align="center">4.121</td>
        <td align="center">7.048</td>
        <td align="center">N</td>
    </tr>
    <tr>
        <td rowspan="3">LLM &amp; VLM</td>
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
        <td rowspan="3">World model</td>
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
