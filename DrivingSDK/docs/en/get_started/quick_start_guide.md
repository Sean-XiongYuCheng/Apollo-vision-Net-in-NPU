# Quick Start

Driving SDK provides two core capabilities: the [**High-Performance Operator Library**](#high-performance-operator-library) and the [**Preferred Model Library**](#preferred-model-library), which significantly reduce the workload of operator adaptation and model migration. This document provides examples based on these two core capabilities to help you get started quickly. Before you begin, click [Driving SDK Quick Installation](https://www.hiascend.com/en/developer/software/mindsdk/download) to set up the environment. For detailed installation instructions, see the [Driving SDK Software Installation](../installation/installation.md).

## High-Performance Operator Library

The High-Performance Operator Library provides multiple NPU-optimized operators that you can directly call to replace PyTorch native operators for better execution performance. For high-performance operators and their detailed descriptions, see the [API List](../api/README.md).

This section uses the [scatter_max](../api/context/scatter_max.md) operator as an example to demonstrate how to call the high-performance operators of Driving SDK and print the results to verify that the output is correct. Its main function is to aggregate the maximum values at the corresponding positions of multiple source tensors into a target tensor based on the indices and return which source tensor each maximum value comes from.

```python
import torch, torch_npu
from mx_driving import scatter_max

updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4],
                        [0, 2, 1, 3, 0, 3, 4, 2],
                        [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
print(out)
print(argmax)
```

If the code runs correctly, the output should be as follows:

```text
tensor([[2., 2., 3., 4., 4., 3., 2., 4.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 1., 3., 0., 3., 4., 2.]], device='npu:0')
tensor([[0, 2, 2, 2, 2, 2, 2, 0],
        [3, 3, 3, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1]], device='npu:0', dtype=torch.int32)
```

## Preferred Model Library

The Preferred Model Library provides NPU migration solutions for typical autonomous driving models such as BEVFusion, BEVFormer, and Sparse4D. [BEVFusion](../../../model_examples/BEVFusion/README.md) is an efficient and versatile multi-task multi-sensor fusion framework that unifies multimodal features in a shared bird's-eye view (BEV) representation space, which preserves geometric and semantic information well and therefore better supports 3D perception tasks. For more model examples and detailed usage guidance, refer to the [Model List](../models/support_list.md).

This section uses the BEVFusion model as an example to demonstrate the complete process from environment preparation to model training.

### Installing the Model Environment

1. Enter the root directory of the BEVFusion model code

    ```shell
    cd DrivingSDK/model_examples/BEVFusion
    ```

2. Install the dependencies using the one-click configuration script

    ```shell
    bash install_BEVFusion.sh
    ```

    > [!NOTE]
    >
    > The script automatically compiles and installs `mmcv` and `mmdetection3d` from source.

### Preparing the Training Dataset

1. Obtain the dataset

    Enter the `mmdetection3d` directory automatically created by the `install_BEVFusion.sh` script:

    ```shell
    cd mmdetection3d
    ```

    Download the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download) or create a soft link to the `nuscenes` folder, and then create the following directory structure in `mmdetection3d/`:

    ```shell
    data/
    ├── lyft
    ├── nuscenes
    │ ├── maps
    │ ├── samples
    │ ├── sweeps
    │ ├── v1.0-test
    │ ├── v1.0-trainval
    ├── s3dis
    ├── scannet
    ├── sunrgbd
    ```

2. Preprocess the data

    Preprocess the data in the `mmdetection3d` directory:

    ```shell
    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```

    After preprocessing, the `nuscenes` file structure is as follows:

    ```shell
    nuscenes/
    ├── maps
    ├── nuscenes_gt_database
    ├── nuscenes_infos_test.pkl
    ├── nuscenes_infos_train.pkl
    ├── nuscenes_infos_val.pkl
    ├── samples
    ├── sweeps
    ├── v1.0-test
    ├── v1.0-trainval
    ```

### Preparing Pretrained Weights

1. Create the `pretrained` directory to store weight files

    Create the `pretrained` folder in the `mmdetection3d` directory:

    ```bash
    mkdir pretrained
    cd pretrained
    ```

2. Obtain the model pretrained weights

    Manually download the pretrained weight files [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth) and [lidar-only pre-trained detector](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth), and place the weight files in the `pretrained` folder. The directory example is as follows:

    ```shell
    pretrained/
    ├── swint-nuimages-pretrained.pth
    ├── bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
     ```

### Starting Training

1. Return to the BEVFusion model root directory

    ```shell
    cd ../../
    ```

2. Run the training script

    ```shell
    # FP32 precision training (6 epochs by default)
    bash test/train_full_8p_base_fp32.sh --batch-size=4 --num-npu=8
    ```

    The script supports the following CLI parameters, including default values, keyword parameters, and positional parameters:

    **Table 1 parameter description**

    |Parameter|Description|
    |----|----|
    |--batch-size|Batch size per NPU device. The default value is 4.|
    |--num-npu|Number of NPU devices per node. The default value is 8.|
    |--epochs|Number of training epochs. The default value is 6.|

3. Check the training results

    The training log is saved in `test/output/0/train_0.log` by default. If you see output similar to the following, training has been started successfully:

    ```text
   02/03 23:41:05 - mmengine - INFO - Epoch(train) [1][   1/3862]  lr: 6.6667e-05  eta: 36 days, 9:13:56  time: 135.6712  data_time: 4.1619  memory: 18113  grad_norm: nan  loss: 18.0867  loss_heatmap: 6.6903  layer_-1_loss_cls: 0.5519  layer_-1_loss_bbox: 10.8446  matched_ious: 0.0018
   02/03 23:41:07 - mmengine - INFO - Epoch(train) [1][   2/3862]  lr: 6.6934e-05  eta: 18 days, 8:22:00  time: 68.4213  data_time: 2.0908  memory: 18231  grad_norm: nan  loss: 17.8768  loss_heatmap: 6.7121  layer_-1_loss_cls: 0.5522  layer_-1_loss_bbox: 10.6125  matched_ious: 0.0037
   ```

    After training completes, the training script validates the model automatically and outputs metrics similar to the following:

   ```text
   mAP: 0.6805
   mATE: 0.2762
   mASE: 0.2518
   mAOE: 0.3478
   mAVE: 0.2928
   mAAE: 0.1951
   NDS: 0.7039
   Eval time: 158.1s
   ```
