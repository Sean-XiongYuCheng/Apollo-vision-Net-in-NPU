"""NPU-friendly nuScenes tracking MOT accumulator (self-contained).

``patcher.replace_import("nuscenes.eval.tracking.mot", ...)`` imports this
module before any mx_driving patch is applied, so it must be importable on
its own: on Python 3.11 the ``motmetrics`` dependency needs the
``collections`` aliases restored (they would otherwise only be applied later
by ``CollectionsCompat`` during the patcher bootstrap phase).
"""

from __future__ import annotations

import collections as _collections
import collections.abc as _collections_abc
from collections import OrderedDict
from itertools import count

import numpy as np
import pandas as pd

for _name in (
    "Iterable",
    "Mapping",
    "MutableMapping",
    "Sequence",
    "MutableSequence",
    "Set",
    "MutableSet",
    "Callable",
    "Hashable",
    "Container",
):
    if not hasattr(_collections, _name) and hasattr(_collections_abc, _name):
        try:
            setattr(_collections, _name, getattr(_collections_abc, _name))
        except Exception:
            continue

# motmetrics 内部依赖 collections 的旧别名，必须在别名恢复后再导入
import motmetrics  # noqa: E402


class MOTAccumulatorCustom(motmetrics.mot.MOTAccumulator):
    """Drop-in replacement for nuScenes MOT accumulator."""

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        idx = pd.MultiIndex.from_tuples(indices, names=["FrameId", "Event"])
        return pd.DataFrame(events, index=idx, columns=["Type", "OId", "HId", "D"])

    @staticmethod
    def new_event_dataframe():
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["FrameId", "Event"])
        cats = pd.Categorical([], categories=["RAW", "FP", "MISS", "SWITCH", "MATCH"])
        return pd.DataFrame(
            OrderedDict(
                [
                    ("Type", pd.Series(cats)),
                    ("OId", pd.Series(dtype=object)),
                    ("HId", pd.Series(dtype=object)),
                    ("D", pd.Series(dtype=float)),
                ]
            ),
            index=idx,
        )

    @property
    def events(self):
        # dirty_events 由父类 MOTAccumulator.__init__ 定义，pylint 无法跨包推断
        if self.dirty_events:  # pylint: disable=access-member-before-definition
            self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @staticmethod
    def merge_event_dataframes(
        dfs,
        update_frame_indices=True,
        update_oids=True,
        update_hids=True,
        return_mappings=False,
    ):
        mapping_infos = []
        new_oid = count()
        new_hid = count()
        result = MOTAccumulatorCustom.new_event_dataframe()

        for df in dfs:
            if isinstance(df, MOTAccumulatorCustom):
                df = df.events
            copy_df = df.copy()
            infos = {}

            if update_frame_indices:
                if result.index.get_level_values(0).size > 0 and isinstance(result.index.get_level_values(0)[0], tuple):
                    index_temp = []
                    for item in result.index.get_level_values(0):
                        index_temp += list(item)
                    index_temp = np.array(index_temp)
                    next_frame_id = max(np.max(index_temp) + 1, np.unique(index_temp).shape[0])
                else:
                    next_frame_id = max(
                        result.index.get_level_values(0).max() + 1,
                        result.index.get_level_values(0).unique().shape[0],
                    )
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy_df.index = copy_df.index.map(lambda x, offset=next_frame_id: (x[0] + offset, x[1]))
                infos["frame_offset"] = next_frame_id

            if update_oids:
                oid_map = dict([oid, str(next(new_oid))] for oid in copy_df["OId"].dropna().unique())
                copy_df["OId"] = copy_df["OId"].map(lambda x, mapping=oid_map: mapping[x], na_action="ignore")
                infos["oid_map"] = oid_map

            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy_df["HId"].dropna().unique())
                copy_df["HId"] = copy_df["HId"].map(lambda x, mapping=hid_map: mapping[x], na_action="ignore")
                infos["hid_map"] = hid_map

            result = pd.concat((result, copy_df))
            mapping_infos.append(infos)

        if return_mappings:
            return result, mapping_infos
        return result
