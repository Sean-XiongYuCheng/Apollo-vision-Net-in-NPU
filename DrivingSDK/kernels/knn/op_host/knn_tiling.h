/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef KNN_TILING_H
#define KNN_TILING_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {
/****************TilingData definition*****************/
BEGIN_TILING_DATA_DEF(KnnTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, nPoint);
TILING_DATA_FIELD_DEF(uint32_t, nSource);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(bool, isFromKnn);
TILING_DATA_FIELD_DEF(int32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, targetGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, sourceTileSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, groupsPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, totalGroups);
// 保留历史字段顺序；新 tiling 字段只能追加，避免破坏 host/kernel 缓存兼容性。
TILING_DATA_FIELD_DEF(uint32_t, groupsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, groupsPerCoreBase);
TILING_DATA_FIELD_DEF(uint32_t, extraCoreGroups);
TILING_DATA_FIELD_DEF(bool, useTopK);
TILING_DATA_FIELD_DEF(uint32_t, sourceAligned);
TILING_DATA_FIELD_DEF(uint32_t, topkTmpSize);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topKTilingData);
TILING_DATA_FIELD_DEF(uint32_t, selectorMode);
TILING_DATA_FIELD_DEF(bool, balanceTargetGroups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Knn, KnnTilingData)
}

#endif // KNN_TILING_H
