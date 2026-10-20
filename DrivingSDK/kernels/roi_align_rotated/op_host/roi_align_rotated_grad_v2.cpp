/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "roi_align_rotated_grad_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
using namespace std;

namespace {
const uint32_t INPUT_INPUT = 0;
const uint32_t INPUT_ROIS = 1;
const uint32_t INPUT_GRAD_OUTPUT = 2;

const uint32_t INPUT_POOLED_HEIGHT = 0;
const uint32_t INPUT_POOLED_WIDTH = 1;
const uint32_t INPUT_SPATIAL_SCALE = 2;
const uint32_t INPUT_SAMPLING_RATIO = 3;
const uint32_t INPUT_ALIGNED = 4;
const uint32_t INPUT_CLOCKWISE = 5;

const uint32_t BOX_SIZE_DIM = 1;
const uint32_t BATCH_SIZE_DIM = 0;
const uint32_t CHANNEL_DIM = 1;
const uint32_t HEIGHT_DIM = 2;
const uint32_t WIDTH_DIM = 3;

const uint32_t OUTPUT_GRAD_INPUT = 0;

const uint32_t WORKSAPCE_16MBYTE_SIZE = 16 * 1024 * 1024;
const uint32_t LARGE_ROI_OUTER_ATOMIC_THRESHOLD = 1024;
const uint32_t PARALLEL_MODE_ROI = 0;
const uint32_t PARALLEL_MODE_POOLED_ROW = 1;
const uint32_t PARALLEL_MODE_LARGE_ROI = 2;
const uint32_t PARALLEL_MODE_POOLED_ROW_GROUP = 4;
} // namespace

namespace optiling {
static ge::graphStatus TilingFuncForRoiAlignRotatedGradV2(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    RoiAlignRotatedGradV2TilingData tiling;
    auto inputTensorPtr = context->GetInputTensor(INPUT_INPUT);
    auto RoiTensorPtr = context->GetInputTensor(INPUT_ROIS);
    auto gradOutputTensorPtr = context->GetInputTensor(INPUT_GRAD_OUTPUT);
    if (inputTensorPtr == nullptr || RoiTensorPtr == nullptr || gradOutputTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inputShape = inputTensorPtr->GetStorageShape();
    auto RoiShape = RoiTensorPtr->GetStorageShape();

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    uint32_t coreNum = ascendplatformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto pooledHeightAttr = attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_HEIGHT);
    auto pooledWidthAttr = attrs->GetAttrPointer<uint32_t>(INPUT_POOLED_WIDTH);
    auto spatialScaleAttr = attrs->GetAttrPointer<float>(INPUT_SPATIAL_SCALE);
    auto samplingRatioAttr = attrs->GetAttrPointer<uint32_t>(INPUT_SAMPLING_RATIO);
    auto alignedAttr = attrs->GetAttrPointer<bool>(INPUT_ALIGNED);
    auto clockwiseAttr = attrs->GetAttrPointer<bool>(INPUT_CLOCKWISE);
    if (pooledHeightAttr == nullptr || pooledWidthAttr == nullptr || spatialScaleAttr == nullptr ||
        samplingRatioAttr == nullptr || alignedAttr == nullptr || clockwiseAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t boxSize = RoiShape.GetDim(BOX_SIZE_DIM);
    int32_t pooledHeight = *pooledHeightAttr;
    int32_t pooledWidth = *pooledWidthAttr;
    if (pooledHeight <= 0 || pooledWidth <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = inputShape.GetDim(BATCH_SIZE_DIM);
    uint32_t channelNum = inputShape.GetDim(CHANNEL_DIM);
    uint32_t width = inputShape.GetDim(WIDTH_DIM);
    uint32_t height = inputShape.GetDim(HEIGHT_DIM);

    bool aligned = *alignedAttr;
    bool clockwise = *clockwiseAttr;
    int32_t samplingRatio = *samplingRatioAttr;
    float spatialScale = *spatialScaleAttr;

    uint32_t parallelMode = PARALLEL_MODE_ROI;
    uint64_t pooledArea = static_cast<uint64_t>(pooledHeight) * static_cast<uint64_t>(pooledWidth);
    bool enableRowSplit = samplingRatio >= 3 && pooledArea > 8;
    enableRowSplit = enableRowSplit || (samplingRatio == 2 && pooledArea > 25);
    enableRowSplit = enableRowSplit || (samplingRatio == 1 && pooledArea >= 50);
    bool enableAdaptiveRowSplit = samplingRatio == 0 && pooledArea >= 49 && boxSize > 1 && boxSize <= 16;
    bool enableSmallRoiRowSplit = samplingRatio == 2 && pooledArea >= 49 && boxSize > 1 && boxSize <= 64;
    if (boxSize >= LARGE_ROI_OUTER_ATOMIC_THRESHOLD) {
        parallelMode = PARALLEL_MODE_LARGE_ROI;
    } else if (samplingRatio > 0 && pooledArea == 1) {
        parallelMode = PARALLEL_MODE_POOLED_ROW_GROUP;
    } else if (enableAdaptiveRowSplit || enableSmallRoiRowSplit ||
        (boxSize > 1 && boxSize <= 16 && enableRowSplit && (boxSize < coreNum || pooledHeight < 8))) {
        parallelMode = PARALLEL_MODE_POOLED_ROW;
    }
    uint32_t rowGroupSize = 2;
    uint64_t totalTaskNum = boxSize;
    if (parallelMode == PARALLEL_MODE_POOLED_ROW) {
        totalTaskNum = static_cast<uint64_t>(boxSize) * static_cast<uint64_t>(pooledHeight);
    } else if (parallelMode == PARALLEL_MODE_POOLED_ROW_GROUP) {
        uint32_t rowGroupNum = static_cast<uint32_t>((pooledHeight + rowGroupSize - 1) / rowGroupSize);
        totalTaskNum = static_cast<uint64_t>(boxSize) * static_cast<uint64_t>(rowGroupNum);
    }
    if (totalTaskNum > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalTaskNumU32 = static_cast<uint32_t>(totalTaskNum);
    uint32_t usedCoreNum = totalTaskNumU32 == 0 ? 1 : (totalTaskNumU32 < coreNum ? totalTaskNumU32 : coreNum);
    context->SetBlockDim(usedCoreNum);
    uint32_t coreRoisNums = totalTaskNumU32 / usedCoreNum;
    uint32_t coreRoisTail = totalTaskNumU32 % usedCoreNum;

    tiling.set_coreRoisNums(coreRoisNums);
    tiling.set_coreRoisTail(coreRoisTail);
    tiling.set_boxSize(boxSize);
    tiling.set_pooledHeight(pooledHeight);
    tiling.set_pooledWidth(pooledWidth);
    tiling.set_batchSize(batchSize);
    tiling.set_channelNum(channelNum);
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_aligned(aligned);
    tiling.set_clockwise(clockwise);
    tiling.set_samplingRatio(samplingRatio);
    tiling.set_spatialScale(spatialScale);
    tiling.set_coreNum(usedCoreNum);
    tiling.set_parallelMode(parallelMode);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WORKSAPCE_16MBYTE_SIZE;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForRoiAlignRotatedGradV2(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(INPUT_INPUT);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *gradInputShape = context->GetOutputShape(OUTPUT_GRAD_INPUT);
    if (gradInputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gradInputShape->AppendDim(inputShape->GetDim(BATCH_SIZE_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(HEIGHT_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(WIDTH_DIM));
    gradInputShape->AppendDim(inputShape->GetDim(CHANNEL_DIM));
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeForRoiAlignRotatedGradV2(gert::InferDataTypeContext *context) {
    auto inputDtype = context->GetInputDataType(INPUT_INPUT);
    context->SetOutputDataType(OUTPUT_GRAD_INPUT, inputDtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoiAlignRotatedGradV2 : public OpDef {
  public:
    explicit RoiAlignRotatedGradV2(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("pooled_height").AttrType(REQUIRED).Int();
        this->Attr("pooled_width").AttrType(REQUIRED).Int();
        this->Attr("spatial_scale").AttrType(REQUIRED).Float();
        this->Attr("sampling_ratio").AttrType(REQUIRED).Int();
        this->Attr("aligned").AttrType(REQUIRED).Bool();
        this->Attr("clockwise").AttrType(REQUIRED).Bool();
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForRoiAlignRotatedGradV2)
            .SetInferDataType(ge::InferDataTypeForRoiAlignRotatedGradV2);
        this->AICore().SetTiling(optiling::TilingFuncForRoiAlignRotatedGradV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(RoiAlignRotatedGradV2);
}
