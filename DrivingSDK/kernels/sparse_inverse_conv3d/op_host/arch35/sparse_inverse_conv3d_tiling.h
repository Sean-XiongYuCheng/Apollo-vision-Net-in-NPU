/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef SPARSE_INVERSE_CONV3D_TILING_H
#define SPARSE_INVERSE_CONV3D_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseInverseConv3dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, inChannel);
TILING_DATA_FIELD_DEF(uint32_t, totalInputPoints);
TILING_DATA_FIELD_DEF(uint32_t, startOffset);
TILING_DATA_FIELD_DEF(uint32_t, totalPairs);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseInverseConv3d, SparseInverseConv3dTilingData)

} // namespace optiling
#endif // SPARSE_INVERSE_CONV3D_TILING_H
