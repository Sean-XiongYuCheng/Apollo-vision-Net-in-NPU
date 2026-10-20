/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "calc_poly_start_end_sl_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "csrc/utils.h"

namespace {
constexpr uint32_t DATA_BLOCK_SIZE = 32;
constexpr uint32_t ELEM_BYTE_SIZE = sizeof(float);
constexpr uint32_t FLOAT32_BYTES = 4;
constexpr uint32_t INT32_BYTES = 4;

constexpr uint32_t MIN_IDX = 0;
constexpr uint32_t POLY_LINE_IDX = 1;
constexpr uint32_t PTS_IDX = 2;
constexpr uint32_t S_CUM_IDX = 3;
constexpr uint32_t SL_OUTPUT_IDX = 0;
constexpr uint32_t POLY_START_OUTPUT_IDX = 1;
constexpr uint32_t POLY_END_OUTPUT_IDX = 2;
constexpr uint32_t BATCH_SIZE_IDX = 0;
constexpr uint32_t NUM_IDX = 1;
constexpr uint32_t NPoints_IDX = 1;
constexpr uint32_t SEQ_LENGTH_IDX = 2;
const uint64_t UB_RESERVE_BYTES = 16 * 1024;
const uint32_t BLOCK_BYTES = 32;
}

namespace optiling {
static ge::graphStatus TilingForCalcPolyStartEndSl(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    /* get some global information: aivNum */
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto coreNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    /* Compute tiling information */
    auto minIdxShapePtr = context->GetInputShape(MIN_IDX);
    auto polyLineShapePtr = context->GetInputShape(POLY_LINE_IDX);
    auto ptsShapePtr = context->GetInputShape(PTS_IDX);
    auto sCumShapePtr = context->GetInputShape(S_CUM_IDX);
    auto slShapePtr = context->GetOutputShape(SL_OUTPUT_IDX);
    auto polyStartShapePtr = context->GetOutputShape(POLY_START_OUTPUT_IDX);
    auto polyEndShapePtr = context->GetOutputShape(POLY_END_OUTPUT_IDX);
    if (minIdxShapePtr == nullptr || polyLineShapePtr == nullptr || ptsShapePtr == nullptr || sCumShapePtr == nullptr ||
        slShapePtr == nullptr || polyStartShapePtr == nullptr || polyEndShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto minIdxShape = minIdxShapePtr->GetStorageShape();
    uint32_t batchSize = minIdxShape.GetDim(BATCH_SIZE_IDX);
    uint32_t numIdx = minIdxShape.GetDim(NUM_IDX);

    auto polyLineShape = polyLineShapePtr->GetStorageShape();
    uint32_t npoints = polyLineShape.GetDim(NPoints_IDX);

    uint32_t dataSize = DATA_BLOCK_SIZE / ELEM_BYTE_SIZE;
    uint32_t numPointAligned = AlignUp(npoints, dataSize);
    uint32_t numIdxAligned = AlignUp(numIdx, dataSize);
    uint32_t numIdxRound64 = AlignUp(numIdx, (uint32_t)64);

    /* Compute tiling info */
    uint32_t totalTaskNum = batchSize;
    uint32_t numTaskPerCore = totalTaskNum / coreNum;
    uint32_t numTaskRemained = totalTaskNum % coreNum;

    uint64_t ubBytesTotal;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytesTotal);
    uint64_t ubBytes = ubBytesTotal - UB_RESERVE_BYTES;

    uint32_t ubPerBatchBytes = 0;
    ubPerBatchBytes += numPointAligned * 2 * FLOAT32_BYTES; // poly_line
    ubPerBatchBytes += numPointAligned * FLOAT32_BYTES; // s_cum
    ubPerBatchBytes += numIdxAligned * INT32_BYTES; // min_idx
    ubPerBatchBytes += numIdxRound64 * 2 * FLOAT32_BYTES; // points
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // poly_start
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // poly_end
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // sl
    ubPerBatchBytes += numIdxAligned * INT32_BYTES; // minIdxLocalscum
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // polyStartX
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // polyEndX
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // subPolyEndX
    ubPerBatchBytes += numIdxRound64 * 2 * FLOAT32_BYTES; // pointBufX
    ubPerBatchBytes += numIdxAligned * 2 * FLOAT32_BYTES; // dotValueBufX
    ubPerBatchBytes += numIdxAligned * INT32_BYTES; // batchIdxOffset
    ubPerBatchBytes += numIdxAligned * INT32_BYTES; // sCumBatchOffset
    ubPerBatchBytes += numIdxAligned * INT32_BYTES; // selectIdx

    ubPerBatchBytes += numIdxAligned * 2 * INT32_BYTES // offsetBufX
        + numIdxAligned * 2 * INT32_BYTES // offsetBufY
        + numIdxAligned * 2 * FLOAT32_BYTES; // offsetFloatBuf

    uint32_t compBatchNum = (ubBytes - BLOCK_BYTES) / ubPerBatchBytes;
    if (compBatchNum == 0) {
        return ge::GRAPH_FAILED;
    }

    CalcPolyStartEndSlTilingData tilingData;
    tilingData.set_batchSize(batchSize);
    tilingData.set_npoints(npoints);
    tilingData.set_numIdx(numIdx);
    tilingData.set_totalTaskNum(totalTaskNum);
    tilingData.set_numTaskPerCore(numTaskPerCore);
    tilingData.set_numTaskRemained(numTaskRemained);
    tilingData.set_compBatchNum(compBatchNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForCalcPolyStartEndSl(gert::InferShapeContext *context) {
    const gert::Shape *min_idx = context->GetInputShape(MIN_IDX);
    const gert::Shape *poly_line = context->GetInputShape(POLY_LINE_IDX);
    const gert::Shape *points = context->GetInputShape(PTS_IDX);
    const gert::Shape *s_cum = context->GetInputShape(S_CUM_IDX);
    gert::Shape *sl = context->GetOutputShape(SL_OUTPUT_IDX);
    gert::Shape *poly_start = context->GetOutputShape(POLY_START_OUTPUT_IDX);
    gert::Shape *poly_end = context->GetOutputShape(POLY_END_OUTPUT_IDX);
    if (min_idx == nullptr || poly_line == nullptr || points == nullptr || s_cum == nullptr || sl == nullptr ||
        poly_start == nullptr || poly_end == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batchSize = min_idx->GetDim(BATCH_SIZE_IDX);
    int64_t numIdx = min_idx->GetDim(NUM_IDX);
    int64_t dims = points->GetDim(2);
    *sl = {batchSize, numIdx, dims};
    *poly_start = {batchSize, numIdx, dims};
    *poly_end = {batchSize, numIdx, dims};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCalcPolyStartEndSl(gert::InferDataTypeContext *context) {
    const auto out_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(SL_OUTPUT_IDX, out_dtype);
    context->SetOutputDataType(POLY_START_OUTPUT_IDX, out_dtype);
    context->SetOutputDataType(POLY_END_OUTPUT_IDX, out_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class CalcPolyStartEndSl : public OpDef {
  public:
    explicit CalcPolyStartEndSl(const char *name) : OpDef(name) {
        this->Input("min_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("poly_line")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("s_cum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("poly_start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("poly_end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sl")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeForCalcPolyStartEndSl)
            .SetInferDataType(ge::InferDataTypeForCalcPolyStartEndSl);
        this->AICore().SetTiling(optiling::TilingForCalcPolyStartEndSl);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(CalcPolyStartEndSl);
}
