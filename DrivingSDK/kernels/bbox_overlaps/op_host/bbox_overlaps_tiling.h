/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef BBOXOVERLAPS_TILING_H
#define BBOXOVERLAPS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BboxOverlapsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockCount)
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount)
TILING_DATA_FIELD_DEF(uint64_t, nSize)
TILING_DATA_FIELD_DEF(uint32_t, availableUBSize)
TILING_DATA_FIELD_DEF(uint32_t, modeFlag);
TILING_DATA_FIELD_DEF(float, margin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BboxOverlaps, BboxOverlapsTilingData)
}
#endif // BBOXOVERLAPS_TILING_H
