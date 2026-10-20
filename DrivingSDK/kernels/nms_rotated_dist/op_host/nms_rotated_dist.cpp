/*

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "nms_rotated_dist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace optiling {
static ge::graphStatus NmsRotatedDistTilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    NmsRotatedDistTilingData tiling;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    if (context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetInputDesc(0) == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto boxShape = context->GetInputShape(0)->GetStorageShape();
    auto maskShape = context->GetOutputShape(0)->GetStorageShape();
    auto dtype = context->GetInputDesc(0)->GetDataType();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t boxNum = boxShape.GetDim(0);
    uint32_t maskNum = maskShape.GetDim(1);
    if (ge::DT_FLOAT != dtype) {
        return ge::GRAPH_FAILED;
    }

    uint32_t usedCoreNum = std::min(boxNum, coreNum);

    // Handle empty input case to avoid division by zero
    if (boxNum == 0) {
        return ge::GRAPH_SUCCESS;
    }

    uint32_t headCoreNum = boxNum % usedCoreNum; // 30

    uint32_t boxNumPerHeadCore = (boxNum + usedCoreNum - 1) / usedCoreNum; // 25
    uint32_t boxNumPerTailCore = boxNum / usedCoreNum;

    const float *thresholdPtr = attrs->GetAttrPointer<float>(0);
    if (thresholdPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float threshold = *thresholdPtr;
    context->SetBlockDim(usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_headCoreNum(headCoreNum);
    tiling.set_boxNumPerHeadCore(boxNumPerHeadCore);
    tiling.set_boxNumPerTailCore(boxNumPerTailCore);
    tiling.set_boxNum(boxNum);
    tiling.set_maskNum(maskNum);
    tiling.set_threshold(threshold);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus NmsRotatedDistInferShape(gert::InferShapeContext *context) { return GRAPH_SUCCESS; }

static ge::graphStatus NmsRotatedDistInferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_INT16);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class NmsRotatedDist : public OpDef {
  public:
    explicit NmsRotatedDist(const char *name) : OpDef(name) {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("threshold").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::NmsRotatedDistInferShape).SetInferDataType(ge::NmsRotatedDistInferDataType);

        this->AICore().SetTiling(optiling::NmsRotatedDistTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(NmsRotatedDist);
} // namespace ops
