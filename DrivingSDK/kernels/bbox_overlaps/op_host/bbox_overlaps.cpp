/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "common/op_host/common.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "bbox_overlaps_tiling.h"
using namespace ge;
using namespace matmul_tiling;

namespace {
constexpr float AVALIABLE_UB_RATIO = 0.8;

const uint32_t POS_INPUT_BOXES_A = 0;
const uint32_t POS_INPUT_BOXES_B = 1;

const uint32_t POS_OUTPUT_RES = 0;

const uint32_t POS_ATTR_MODE_FLAG = 0;
const uint32_t POS_ATTR_ALIGNED = 1;
const uint32_t POS_ATTR_MARGIN = 2;

const uint32_t BOXES_NUM_DIM = 1;
};

namespace optiling {
static ge::graphStatus TilingFunc4BboxOverlaps(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();

    context->SetBlockDim(aivNum);

    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize *= AVALIABLE_UB_RATIO;

    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto boxesATensorPtr = context->GetInputTensor(POS_INPUT_BOXES_A);
    auto boxesBTensorPtr = context->GetInputTensor(POS_INPUT_BOXES_B);
    auto attrs = context->GetAttrs();
    if (boxesATensorPtr == nullptr || boxesBTensorPtr == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto modeFlagPtr = attrs->GetAttrPointer<int>(POS_ATTR_MODE_FLAG);
    auto alignedPtr = attrs->GetAttrPointer<bool>(POS_ATTR_ALIGNED);
    auto marginPtr = attrs->GetAttrPointer<float>(POS_ATTR_MARGIN);
    if (modeFlagPtr == nullptr || alignedPtr == nullptr || marginPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto modeFlag = *modeFlagPtr;
    auto aligned = *alignedPtr;
    auto margin = *marginPtr;

    auto boxesAShape = boxesATensorPtr->GetStorageShape();
    auto boxesBShape = boxesBTensorPtr->GetStorageShape();

    auto nSizeA = boxesAShape.GetDim(BOXES_NUM_DIM);
    auto nSizeB = boxesBShape.GetDim(BOXES_NUM_DIM);
    auto nSize = nSizeA;

    if (aligned && nSizeA != nSizeB) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint64_t N_SIZE_LIMIT = 1ULL << 40;
    if (nSize == 0 || static_cast<uint64_t>(nSize) > N_SIZE_LIMIT) {
        return ge::GRAPH_FAILED;
    }

    BboxOverlapsTilingData tilingData;
    tilingData.set_modeFlag(modeFlag);
    tilingData.set_margin(margin);

    int32_t blockCount = nSize / aivNum;
    int32_t bigCoreCount = nSize % aivNum;

    tilingData.set_blockCount(blockCount);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_nSize(nSize);
    tilingData.set_availableUBSize(ubSize);

    ADD_TILING_DATA(context, tilingData);

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus Infershape4BboxOverlaps(gert::InferShapeContext *context) {
    auto boxesAShape = context->GetInputShape(POS_INPUT_BOXES_A);
    auto boxesBShape = context->GetInputShape(POS_INPUT_BOXES_B);
    auto areaOverlapShape = context->GetOutputShape(POS_OUTPUT_RES);
    if (boxesAShape == nullptr || boxesBShape == nullptr || areaOverlapShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto boxesANum = boxesAShape->GetDim(BOXES_NUM_DIM);
    auto boxesBNum = boxesBShape->GetDim(BOXES_NUM_DIM);

    areaOverlapShape->SetDimNum(0);
    areaOverlapShape->AppendDim(boxesANum);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BboxOverlaps(gert::InferDataTypeContext *context) {
    const ge::DataType box_dtype = context->GetInputDataType(POS_INPUT_BOXES_A);
    context->SetOutputDataType(POS_OUTPUT_RES, box_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class BboxOverlaps : public OpDef {
  public:
    explicit BboxOverlaps(const char *name) : OpDef(name) {
        this->Input("boxes_a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("boxes_b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("mode_flag").AttrType(OPTIONAL).Int(0);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(false);
        this->Attr("margin").AttrType(OPTIONAL).Float(1e-5);

        this->SetInferShape(ge::Infershape4BboxOverlaps).SetInferDataType(ge::InferDataType4BboxOverlaps);

        this->AICore().SetTiling(optiling::TilingFunc4BboxOverlaps);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(BboxOverlaps);
}
