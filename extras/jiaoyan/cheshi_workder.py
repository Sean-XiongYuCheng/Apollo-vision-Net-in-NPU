import pickle
from collections.abc import KeysView, ItemsView, ValuesView

import projects.mmdet3d_plugin
from mmcv import Config
from mmdet3d.datasets import build_dataset

VIEW_TYPES = (KeysView, ItemsView, ValuesView)
seen = set()
hits = []

def walk(x, path='dataset'):
    if id(x) in seen or len(hits) >= 50:
        return
    seen.add(id(x))

    if isinstance(x, VIEW_TYPES):
        hits.append((path, type(x).__name__, repr(x)[:160]))
        return

    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, f'{path}[{k!r}]')
    elif isinstance(x, (list, tuple, set)):
        for i, v in enumerate(x):
            walk(v, f'{path}[{i}]')
    elif hasattr(x, '__dict__'):
        for k, v in vars(x).items():
            walk(v, f'{path}.{k}')

cfg = Config.fromfile(
    'projects/configs/bevformer/bev_tiny_det_occ_apollo.py'
)
dataset = build_dataset(cfg.data.train)

print('Dataset:', type(dataset).__name__)
print('Samples:', len(dataset))

walk(dataset)

if hits:
    print('\nFOUND dict view objects:')
    for path, typ, val in hits:
        print(f'{typ}: {path} = {val}')
else:
    print('\nNo dict_keys / dict_items / dict_values found by scan.')

for name, obj in [
    ('dataset', dataset),
    ('dataset.pipeline', dataset.pipeline),
    ('dataset.data_infos', dataset.data_infos),
]:
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'PICKLE PASS: {name}')
    except Exception as e:
        print(f'PICKLE FAIL: {name}: {type(e).__name__}: {e}')