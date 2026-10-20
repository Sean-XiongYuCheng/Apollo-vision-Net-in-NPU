import os
from nuscenes.nuscenes import NuScenes
ns = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)
missing = [sd['filename'] for sd in ns.sample_data
           if not os.path.exists(os.path.join(ns.dataroot, sd['filename']))]
print('sample_data 总数:', len(ns.sample_data))
print('缺失文件数:', len(missing))
for p in missing[:10]: print('  缺:', p)
# 缺失=0 → 数据完整;缺的文件名会告诉你少下/少解了哪些 Part