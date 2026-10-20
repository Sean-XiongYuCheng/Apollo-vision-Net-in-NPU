# Load Balancing

In autonomous driving scenarios, models must process multiple sensor data streams and diverse road traffic environments. Therefore, the compute workloads of an autonomous driving model can easily become unbalanced, creating a performance bottleneck.

Driving SDK provides a load balancing feature for autonomous driving models. It uses load balancing strategies to alleviate the performance bottleneck caused by unbalanced compute workloads and improve model performance.

## Background

Datasets for autonomous driving models usually contain multi-sensor data, including point clouds, radar data, and camera images.

Depending on the model task, data volume can vary greatly across road scenes. For example, the `QCNet` model is a neural network architecture for trajectory prediction that uses the `Argoverse2` dataset to learn trajectory features. The dataset uses `Agents` to represent traffic participants in road scenes, and the number of `Agents` can differ significantly from one road to another. In distributed training, this causes load imbalance across nodes. To wait for the most heavily loaded node, other nodes often enter a communication wait phase, which adds extra training overhead.

Models such as Deformable DETR and CenterPoint3D also have similar load imbalance issues. Therefore, dynamic load balancing can give each node in distributed training a similar load, avoid excessive communication wait overhead, and improve model performance.

## Driving SDK Load Balancing Overview

### `Dynamic Dataset`

The Driving SDK load balancing feature provides the `DynamicDataset` abstract class. This class includes two abstract methods, `sorting` and `bucketing`.

- `sorting` abstract method.
  Sort the elements that cause the main load imbalance bottleneck in the model.

- `bucketing` abstract method.
  Bucket the dataset samples according to the result of the `sorting` method.

You can inherit from this abstract class and implement its abstract methods. Driving SDK also provides two subclasses that inherit from `DynamicDataset`, and both implement the `bucketing` method.

- `UniformBucketingDynamicDataset`.
  This class takes the `num_bucket` parameter as the total number of buckets. The dataset samples are evenly distributed across all buckets according to the result of the `sorting` method.

- `CapacityBucketingDynamicDataset`.
  This class takes the `bucket_capacity` parameter as the bucket capacity. The dataset samples are assigned to each bucket in groups of `bucket_capacity` according to the result of the `sorting` method.

### `Dynamic Sampler`

The Driving SDK load balancing feature provides the `DynamicSampler` abstract class. This class includes the `bucket_arange` abstract method.

- `bucket_arange` abstract method.
  Randomize the result of the `bucketing` method in the `DynamicDataset` class.

You can inherit from this abstract class and implement its abstract methods. Driving SDK also provides two subclasses that inherit from `DynamicSampler`, and both implement the `bucket_arange` method.

- `DynamicDistributedSampler`.
  This class performs two rounds of randomization:
  1. Randomize the sample buckets.
  2. Randomize the data within each sample bucket.
  It applies to randomization of the bucketed result from `UniformBucketingDynamicDataset`.

- `ReplicasDistributedSampler`.
  This class performs three rounds of randomization:
  1. Randomize the sample buckets.
  2. Randomize the data within each sample bucket.
  3. Evenly distribute the data in each bucket across all nodes in distributed training, then randomize it again.
  It applies to randomization of the bucketed result from `CapacityBucketingDynamicDataset`.

### `Dynamic Transforms`

Functions for load-balancing preprocessing.

## Usage

This section uses the CenterPoint3D model as an example to describe how to use the Driving SDK load balancing feature.

### Identifying the Element That Introduces the Load Imbalance Bottleneck

The dataset for CenterPoint3D is point cloud data. During model training, the point cloud is first converted into a `voxel` tensor. The model is sensitive to the size of the `voxel` tensor. When the `voxel` tensor is larger, the model takes longer to compute. Conversely, it takes less time. Therefore, identify the `voxel` tensor size produced by point cloud conversion as the element that introduces the load imbalance bottleneck.

### Sorting by the `voxel` Size (`sorting`)

Generate the serialization files required for the dataset by following the "Prepare the Dataset" section in `model_examples/CenterPoint/README.md`. In the generation code, the following code retrieves the `voxel` tensor size of dataset samples.

```python
from mx_driving.dataset.utils import get_voxel_number_from_mean_vfe
voxel_num = get_voxel_number_from_mean_vfe(data_path, ref_sd_rec['filename'], sweeps, max_sweeps)
info["voxel_num"] = voxel_num
```

The `get_voxel_number_from_mean_vfe` function is located in `dataset/utils/dynamic_transform.py`. Therefore, users can obtain the `voxel` tensor size without making intrusive changes to the model.

Modify the model file `pcdet/datasets/nuscenes/nuscenes_dataset.py`. The original code is:

```python
class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        ...
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
```

After modification, the code is:

```python
from mx_driving.dataset import PointCloudDynamicDataset
class NuScenesDataset(PointCloudDynamicDataset, DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        DatasetTemplate.__init__(
            self, dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        ...
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        if training:
            self.sorting()
            self.buckets = self.bucketing(bucket_capacity=8)
        else:
            self.buckets = None
```

`NuScenesDataset` inherits `PointCloudDynamicDataset` in addition to the original parent class. Therefore, it can call the `sorting` and `bucketing` methods directly.

### Randomizing the `bucketing` Result

Modify the model file `pcdet/datasets/__init__.py`. The original code is:

```python
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```

After modification, the code is:

```python
from mx_driving.dataset import ReplicasDistributedSampler
sampler = ReplicasDistributedSampler(dataset)
```

After the modification, the model trains normally. Training performance with the load balancing feature improves by 18 percent compared with performance without it.
