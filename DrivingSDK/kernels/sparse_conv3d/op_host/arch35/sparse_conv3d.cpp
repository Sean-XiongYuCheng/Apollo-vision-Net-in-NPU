/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "sparse_conv3d_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;


namespace {
const float AVALIABLE_UB_RATIO = 0.9;
const int32_t REVERSED_MEM = 88 * 1024;
const int32_t ALIGN_TASK_NUM = 32;
const int32_t USR_WORKSPACE_SIZE = 16 * 1024 * 1024;

const int32_t SHAPE_B_IDX = 0;
const int32_t SHAPE_D_IDX = 1;
const int32_t SHAPE_H_IDX = 2;
const int32_t SHAPE_W_IDX = 3;
const int32_t STRIDE_D_IDX = 0;
const int32_t STRIDE_H_IDX = 1;
const int32_t STRIDE_W_IDX = 2;
const int32_t PADDING_D_IDX = 0;
const int32_t PADDING_H_IDX = 1;
const int32_t PADDING_W_IDX = 2;
} // namespace


namespace optiling {
static ge::graphStatus TilingForSparseConv3d(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr || context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto indices_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t actualNum = indices_shape.GetDim(0);

    uint32_t coreTask = AlignUp(Ceil(actualNum, coreNum), 32);
    uint32_t usedCoreNum = Ceil(actualNum, coreTask);
    uint32_t lastCoreTask = 0;
    if (coreTask != 0) {
        lastCoreTask = actualNum % coreTask;
    }
    if (lastCoreTask == 0) {
        lastCoreTask = coreTask;
    }
    uint64_t availableUbSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbSize);
    availableUbSize = availableUbSize - REVERSED_MEM;
    context->SetLocalMemorySize(availableUbSize);
    availableUbSize *= AVALIABLE_UB_RATIO;

    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(0);
    auto kernelSizeData = reinterpret_cast<const int64_t*>(kernelSizePtr->GetData());

    uint32_t kernelD = kernelSizeData[0];
    uint32_t kernelH = kernelSizeData[1];
    uint32_t kernelW = kernelSizeData[2];
    uint32_t kernelSize = kernelD * kernelH * kernelW;

    uint32_t moveLen = availableUbSize / 4 / (kernelSize * 5 + 4);
    moveLen = moveLen / ALIGN_TASK_NUM * ALIGN_TASK_NUM;
    if (moveLen > coreTask) {
        moveLen = coreTask;
    }

    uint32_t repeatTimes = Ceil(coreTask, moveLen);
    uint32_t lastRepeatTimes = Ceil(lastCoreTask, moveLen);
    uint32_t moveTail = 0;
    uint32_t lastMoveTail = 0;
    if (moveLen != 0) {
        moveTail = coreTask % moveLen;
        lastMoveTail = lastCoreTask % moveLen;
    }
    if (moveTail == 0) {
        moveTail = moveLen;
    }
    if (lastMoveTail == 0) {
        lastMoveTail = moveLen;
    }

    auto outSpatialShapePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(1);
    auto stridePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(2);
    auto paddingPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(3);
    auto outSpatialShapeData = reinterpret_cast<const int64_t*>(outSpatialShapePtr->GetData());
    auto strideData = reinterpret_cast<const int64_t*>(stridePtr->GetData());
    auto paddingData = reinterpret_cast<const int64_t*>(paddingPtr->GetData());

    SparseConv3dTilingData tiling;
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_coreTask(coreTask);
    tiling.set_lastCoreTask(lastCoreTask);
    tiling.set_moveLen(moveLen);
    tiling.set_repeatTimes(repeatTimes);
    tiling.set_moveTail(moveTail);
    tiling.set_lastRepeatTimes(lastRepeatTimes);
    tiling.set_lastMoveTail(lastMoveTail);
    tiling.set_kernelD(kernelD);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_kernelSize(kernelSize);
    tiling.set_outfeatureB(outSpatialShapeData[SHAPE_B_IDX]);
    tiling.set_outputDepth(outSpatialShapeData[SHAPE_D_IDX]);
    tiling.set_outputHeight(outSpatialShapeData[SHAPE_H_IDX]);
    tiling.set_outputWidth(outSpatialShapeData[SHAPE_W_IDX]);
    tiling.set_strideDepth(strideData[STRIDE_D_IDX]);
    tiling.set_strideHeight(strideData[STRIDE_H_IDX]);
    tiling.set_strideWidth(strideData[STRIDE_W_IDX]);
    tiling.set_paddingDepth(paddingData[PADDING_D_IDX]);
    tiling.set_paddingHeight(paddingData[PADDING_H_IDX]);
    tiling.set_paddingWidth(paddingData[PADDING_W_IDX]);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = USR_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForSparseConv3d(gert::InferShapeContext* context)
{
    const gert::Shape* indicesShape = context->GetInputShape(0);

    if (indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* indicesOutShape = context->GetOutputShape(0);
    gert::Shape* indicesPairShape = context->GetOutputShape(1);
    if (indicesOutShape == nullptr || indicesPairShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t kernelSize = 27;
    uint64_t indicesSecondSize = indicesShape->GetDim(1);

    *indicesOutShape = {indicesSecondSize};
    *indicesPairShape = {kernelSize, indicesSecondSize};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseConv3d(gert::InferDataTypeContext* context)
{
    const ge::DataType indices_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, indices_dtype);
    context->SetOutputDataType(1, indices_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SparseConv3d : public OpDef {
public:
    explicit SparseConv3d(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("indices_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices_pair")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("kernel_size").ListInt();
        this->Attr("out_spatial_shape").ListInt();
        this->Attr("stride").ListInt();
        this->Attr("padding").ListInt();

        this->SetInferShape(ge::InferShapeForSparseConv3d).SetInferDataType(ge::InferDtypeForSparseConv3d);
        this->AICore().SetTiling(optiling::TilingForSparseConv3d);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SparseConv3d);
} // namespace ops
