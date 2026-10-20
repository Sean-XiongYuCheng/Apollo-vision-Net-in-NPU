/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ADAMW_PARAM_UPDATE_TILING_H
#define ADAMW_PARAM_UPDATE_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdamwParamUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, totalCoreTaskNum)
TILING_DATA_FIELD_DEF(float, lr)
TILING_DATA_FIELD_DEF(float, beta1)
TILING_DATA_FIELD_DEF(float, beta2)
TILING_DATA_FIELD_DEF(float, decay)
TILING_DATA_FIELD_DEF(float, eps)
TILING_DATA_FIELD_DEF(float, biasCorrection1)
TILING_DATA_FIELD_DEF(float, biasCorrection2)
TILING_DATA_FIELD_DEF(bool, amsgrad)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdamwParamUpdate, AdamwParamUpdateTilingData)
} // namespace optiling

#endif // ADAMW_PARAM_UPDATE_TILING_H
