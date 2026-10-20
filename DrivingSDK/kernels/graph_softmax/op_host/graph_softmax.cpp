#include <algorithm>
#include <cstdint>
#include "ge/utils.h"
#include "graph_softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t AVAILABLE_UB_PERCENT = 80;
constexpr uint32_t PERCENT_BASE = 100;
constexpr uint32_t SRC_IDX = 0; // number of edge
constexpr uint32_t OUTPUT_IDX = 0;
constexpr uint32_t EDGE_IDX = 0;
constexpr uint32_t FEATURE_IDX = 1;
constexpr uint32_t N_IDX = 0;
constexpr uint32_t NUM_FEATURE = 8;
constexpr uint32_t MAX_EDGE_COUNT = 50000;
constexpr int32_t MAX_GROUP_COUNT = 1500000;
constexpr uint32_t MAX_REDUCE_ROWS = 255;
constexpr uint64_t DATA_BLOCK_BYTES = 32;
} // some const express

namespace optiling {
static ge::graphStatus TilingForGraphSoftmax(gert::TilingContext *context) {
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
    uint64_t ubSize = 0;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (aivNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *srcShape = context->GetInputShape(SRC_IDX); // [num_edge, num_feature]
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    if (srcShape == nullptr || attr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto NPtr = attr->GetAttrPointer<int>(N_IDX);
    if (NPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &storageShape = srcShape->GetStorageShape();
    if (storageShape.GetDimNum() != 2 || *NPtr <= 0 || *NPtr > MAX_GROUP_COUNT) {
        return ge::GRAPH_FAILED;
    }
    int64_t numEdgeValue = storageShape.GetDim(EDGE_IDX);
    int64_t numFeatureValue = storageShape.GetDim(FEATURE_IDX);
    if (numEdgeValue <= 0 || numEdgeValue > MAX_EDGE_COUNT || numFeatureValue != NUM_FEATURE) {
        return ge::GRAPH_FAILED;
    }

    uint32_t N = static_cast<uint32_t>(*NPtr);
    uint32_t numEdge = static_cast<uint32_t>(numEdgeValue);
    uint32_t numFeature = static_cast<uint32_t>(numFeatureValue);
    uint32_t totalTask = numEdge;
    uint32_t coreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t coreWorkspace = (N + aivNum - 1) / aivNum;
    uint32_t totalWorkspace = coreWorkspace * aivNum;
    uint32_t bigCoreCount = totalTask % aivNum == 0 ? aivNum : totalTask % aivNum;
    uint32_t singleLoopTaskCount = coreTask;

    ge::Shape reduceShape({MAX_REDUCE_ROWS, NUM_FEATURE});
    uint32_t maxReduceMaxTmpBytes = 0;
    uint32_t minReduceMaxTmpBytes = 0;
    uint32_t maxReduceSumTmpBytes = 0;
    uint32_t minReduceSumTmpBytes = 0;
    AscendC::GetReduceMaxMaxMinTmpSize(
        reduceShape, ge::DT_FLOAT, AscendC::ReducePattern::RA, true, false, maxReduceMaxTmpBytes, minReduceMaxTmpBytes);
    AscendC::GetReduceSumMaxMinTmpSize(
        reduceShape, ge::DT_FLOAT, AscendC::ReducePattern::RA, true, false, maxReduceSumTmpBytes, minReduceSumTmpBytes);
    uint32_t reduceTmpBytes = std::max(maxReduceMaxTmpBytes, maxReduceSumTmpBytes);
    if (reduceTmpBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t srcBytes = static_cast<uint64_t>(coreTask) * NUM_FEATURE * sizeof(float);
    uint64_t indexBytes = static_cast<uint64_t>(coreTask) * sizeof(uint32_t);
    uint64_t runDescriptorBytes = static_cast<uint64_t>(coreTask) * sizeof(uint32_t);
    uint64_t requiredUbBytes = 2 * AlignUp(srcBytes, DATA_BLOCK_BYTES) + AlignUp(indexBytes, DATA_BLOCK_BYTES) +
        3 * AlignUp(runDescriptorBytes, DATA_BLOCK_BYTES) +
        AlignUp(static_cast<uint64_t>(reduceTmpBytes), DATA_BLOCK_BYTES) + 2 * DATA_BLOCK_BYTES +
        static_cast<uint64_t>(aivNum) * DATA_BLOCK_BYTES;
    uint64_t availableUbBytes = ubSize * AVAILABLE_UB_PERCENT / PERCENT_BASE;
    if (requiredUbBytes > availableUbBytes) {
        return ge::GRAPH_FAILED;
    }

    GraphSoftmaxTilingData tilingData;

    tilingData.set_coreTask(coreTask);
    tilingData.set_coreWorkspace(coreWorkspace);
    tilingData.set_totalTask(totalTask);
    tilingData.set_totalWorkspace(totalWorkspace);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_singleLoopTaskCount(singleLoopTaskCount);
    tilingData.set_reduceTmpBytes(reduceTmpBytes);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t systemWorkspaceSize = ascendplatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize =
        static_cast<size_t>(2) * static_cast<size_t>(totalWorkspace) * numFeature * sizeof(float) +
        static_cast<size_t>(aivNum) * DATA_BLOCK_BYTES;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShapeForGraphSoftmax(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *src = context->GetInputShape(SRC_IDX);
    gert::Shape *softmaxResult = context->GetOutputShape(OUTPUT_IDX);
    if (src == nullptr || softmaxResult == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (src->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    uint64_t numEdge = src->GetDim(EDGE_IDX);
    uint64_t numFeature = src->GetDim(FEATURE_IDX);

    *softmaxResult = {numEdge, numFeature};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGraphSoftmax(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType num_valid_dtype = context->GetInputDataType(SRC_IDX);
    context->SetOutputDataType(OUTPUT_IDX, num_valid_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GraphSoftmax : public OpDef {
  public:
    explicit GraphSoftmax(const char *name) : OpDef(name) {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("softmax_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("N").Int();

        this->SetInferShape(ge::InferShapeForGraphSoftmax).SetInferDataType(ge::InferDataTypeForGraphSoftmax);
        this->AICore().SetTiling(optiling::TilingForGraphSoftmax);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(GraphSoftmax);
}
