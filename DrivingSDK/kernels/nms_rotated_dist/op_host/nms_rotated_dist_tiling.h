/*

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef NMS_ROTATED_DIST_TILING_H
#define NMS_ROTATED_DIST_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NmsRotatedDistTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum) // used cores
TILING_DATA_FIELD_DEF(uint32_t, boxNum) // count of boxes
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum) // loop times
TILING_DATA_FIELD_DEF(uint32_t, boxNumPerHeadCore) // count of each core, = loop_time * 8
TILING_DATA_FIELD_DEF(uint32_t, boxNumPerTailCore) // count of tail core
TILING_DATA_FIELD_DEF(uint32_t, maskNum) // mask align 32bit
TILING_DATA_FIELD_DEF(float, threshold)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NmsRotatedDist, NmsRotatedDistTilingData)
} // namespace optiling

#endif // NMS_ROTATED_DIST_TILING_H
