/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef HUNGARIAN_TILING_H
#define HUNGARIAN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HungarianTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numRows);
TILING_DATA_FIELD_DEF(uint32_t, numCols);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(bool, transposed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Hungarian, HungarianTilingData)

}

#endif
