import torch, torch_npu
import projects.mmdet3d_plugin              # 插件注册
from mmcv import Config
from mmcv.parallel import collate
from mmcv.device import scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

cfg = Config.fromfile('projects/configs/bevformer/bev_tiny_det_occ_apollo.py')
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
print("数据集大小:", len(dataset))

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
model = model.npu().eval()

data = collate([dataset[0]], samples_per_gpu=1)
data = scatter(data, [0])[0]                 # 张量搬到 0 号 NPU

with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data)
print("7.4 整图前向 OK, 输出类型:", type(result))