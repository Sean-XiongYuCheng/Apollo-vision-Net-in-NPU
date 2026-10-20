/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "sparse_inverse_conv3d_tiling.h"

#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;

namespace {
const uint32_t FEATURE_IDX = 0;
const uint32_t ORIGIN_INDICES_IDX = 1;
const uint32_t OUTPUT_FEATURE_IDX = 0;
const uint32_t ATTR_KERNELS_IDX = 0;
const uint32_t ATTR_IN_CHANNELS_IDX = 1;
const uint32_t ATTR_START_OFFSET_IDX = 2;
const uint32_t ATTR_END_OFFSET_IDX = 3;
const uint32_t TOTAL_TASK_DIM_IDX = 0;
const uint32_t KERNEL_SIZE_IDX_0 = 0;
const uint32_t KERNEL_SIZE_IDX_1 = 1;
const uint32_t KERNEL_SIZE_IDX_2 = 2;
const uint32_t INT32_SIZE = 4;
} // namespace

namespace optiling {
static ge::graphStatus TilingForSparseInverseConv3d(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t aivNum = ascendPlatformInfo.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto featureShapePtr = context->GetInputShape(FEATURE_IDX);
    auto attrsPtr = context->GetAttrs();
    if (featureShapePtr == nullptr || attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto *inChannelsPtr = attrsPtr->GetInt(ATTR_IN_CHANNELS_IDX);
    const auto *startOffsetPtr = attrsPtr->GetInt(ATTR_START_OFFSET_IDX);
    const auto *endOffsetPtr = attrsPtr->GetInt(ATTR_END_OFFSET_IDX);
    if (inChannelsPtr == nullptr || startOffsetPtr == nullptr || endOffsetPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inChannel = static_cast<uint32_t>(*inChannelsPtr);
    uint32_t totalInputPoints = static_cast<uint32_t>(featureShapePtr->GetStorageShape().GetDim(TOTAL_TASK_DIM_IDX));
    uint32_t startOffset = static_cast<uint32_t>(*startOffsetPtr);
    uint32_t totalPairs = static_cast<uint32_t>(*endOffsetPtr) - startOffset;

    size_t systemWorkspaceSize = static_cast<size_t>(ascendPlatformInfo.GetLibApiWorkSpaceSize());
    size_t userWorkspaceSize = static_cast<size_t>(totalPairs) * INT32_SIZE;

    SparseInverseConv3dTilingData tilingData;
    context->SetBlockDim(aivNum);
    tilingData.set_inChannel(inChannel);
    tilingData.set_totalInputPoints(totalInputPoints);
    tilingData.set_startOffset(startOffset);
    tilingData.set_totalPairs(totalPairs);

    ADD_TILING_DATA(context, tilingData);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = systemWorkspaceSize + userWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShapeForSparseInverseConv3d(gert::InferShapeContext *context) {
    const gert::Shape *indicesShape = context->GetInputShape(ORIGIN_INDICES_IDX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto totalTaskCount = indicesShape->GetDim(TOTAL_TASK_DIM_IDX);
    auto kernelSizePtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNELS_IDX);
    auto kernelSizeData = reinterpret_cast<const int64_t *>(kernelSizePtr->GetData());
    uint32_t kernelSize = static_cast<uint32_t>(kernelSizeData[KERNEL_SIZE_IDX_0]) *
        static_cast<uint32_t>(kernelSizeData[KERNEL_SIZE_IDX_1]) *
        static_cast<uint32_t>(kernelSizeData[KERNEL_SIZE_IDX_2]);
    auto inChannelsPtr = attrsPtr->GetAttrPointer<int32_t>(ATTR_IN_CHANNELS_IDX);

    gert::Shape *outFeatureShape = context->GetOutputShape(OUTPUT_FEATURE_IDX);
    if (outFeatureShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outFeatureShape->SetDimNum(0);
    outFeatureShape->AppendDim(totalTaskCount);
    outFeatureShape->AppendDim(*inChannelsPtr * kernelSize);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForSparseInverseConv3d(gert::InferDataTypeContext *context) {
    const ge::DataType feature_dtype = context->GetInputDataType(FEATURE_IDX);
    context->SetOutputDataType(0, feature_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SparseInverseConv3d : public OpDef {
  public:
    explicit SparseInverseConv3d(const char *name) : OpDef(name) {
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("unique_indices_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sorted_idx_to_former_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("output_img2col")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size").AttrType(REQUIRED).ListInt();
        this->Attr("in_channels").AttrType(REQUIRED).Int();
        this->Attr("start_offset").AttrType(REQUIRED).Int();
        this->Attr("end_offset").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForSparseInverseConv3d)
            .SetInferDataType(ge::InferDtypeForSparseInverseConv3d);
        this->AICore().SetTiling(optiling::TilingForSparseInverseConv3d);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SparseInverseConv3d);
} // namespace ops
