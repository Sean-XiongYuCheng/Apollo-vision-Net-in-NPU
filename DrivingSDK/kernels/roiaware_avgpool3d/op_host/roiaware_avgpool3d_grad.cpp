#include "roiaware_avgpool3d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
static ge::graphStatus TilingForRoiawareAvgpool3dGrad(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoptr = context->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    auto aivNum = ascendplatformInfo.GetCoreNumAiv();
    context->SetBlockDim(aivNum);

    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t ONE_ELEMENT_SIZE = sizeof(float);
    constexpr uint32_t BYTE_ALIGNED = 32;
    constexpr uint32_t ASYMMETRIC_INPUT_UB_PERCENT = 70;
    constexpr uint32_t ASYMMETRIC_OUTPUT_UB_PERCENT = 20;
    constexpr uint32_t MEDIUM_INPUT_UB_PERCENT = 55;
    constexpr uint32_t MEDIUM_OUTPUT_UB_PERCENT = 35;
    constexpr uint32_t BALANCED_UB_PERCENT = 40;
    // One extra task scan has scalar bookkeeping around the vector Add.  Use
    // the asymmetric split only when every added scan saves at least this
    // many output elements from atomic GM copy-out.
    constexpr uint32_t MIN_SAVED_OUTPUT_ELEMENTS_PER_EXTRA_SCAN = 300;
    constexpr uint32_t HIGH_EFFICIENCY_SAVED_ELEMENTS_PER_EXTRA_SCAN = 600;
    constexpr uint64_t MIN_SAVED_OUTPUT_ELEMENTS = 1400000;
    constexpr uint32_t LARGE_GRID_PREFETCH_MIN_TASKS = 100000;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t MAX_COPY_BLOCKS = 4095;
    constexpr uint64_t UB_RESERVED = 8 * 1024;

    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 0 : argmax, 1: gradOut, 2: gradIn
    auto ptsIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto gradOutShape = context->GetInputShape(1)->GetStorageShape();
    auto gradInShape = context->GetOutputShape(0)->GetStorageShape();
    if (ptsIdxShape.GetDimNum() != 5 || gradOutShape.GetDimNum() != 5 || gradInShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t dim = 0; dim < 4; ++dim) {
        if (ptsIdxShape.GetDim(dim) != gradOutShape.GetDim(dim)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t boxesNum = gradOutShape.GetDim(0);
    uint32_t outX = gradOutShape.GetDim(1);
    uint32_t outY = gradOutShape.GetDim(2);
    uint32_t outZ = gradOutShape.GetDim(3);
    uint32_t channels = gradOutShape.GetDim(4);
    uint32_t npoints = gradInShape.GetDim(0);
    if (boxesNum == 0 || outX == 0 || outY == 0 || outZ == 0 || channels == 0 || channels > 2048 || npoints == 0 ||
        gradInShape.GetDim(1) != channels || ptsIdxShape.GetDim(4) <= 0 || ptsIdxShape.GetDim(4) > 2048) {
        return ge::GRAPH_FAILED;
    }
    uint32_t maxPtsPerVoxel = ptsIdxShape.GetDim(4) - 1;

    const uint64_t totalTask64 = static_cast<uint64_t>(boxesNum) * outX * outY * outZ;
    if (totalTask64 == 0 || totalTask64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalTask = static_cast<uint32_t>(totalTask64);
    uint32_t singleCoreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t channelAligned =
        ((channels * ONE_ELEMENT_SIZE + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / ONE_ELEMENT_SIZE);
    uint32_t maxPtsPerVoxelAligned =
        ((maxPtsPerVoxel * sizeof(int32_t) + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / sizeof(int32_t));
    const uint32_t ptsRowElements = maxPtsPerVoxel + 1;
    const uint32_t ptsRowAligned =
        ((ptsRowElements * sizeof(int32_t) + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / sizeof(int32_t));

    const uint64_t inputTaskBytes =
        static_cast<uint64_t>(ONE_ELEMENT_SIZE) * channelAligned + sizeof(int32_t) * maxPtsPerVoxelAligned;
    const uint64_t outputRowBytes = static_cast<uint64_t>(ONE_ELEMENT_SIZE) * channelAligned;
    const uint64_t asymmetricInputCapacity = (ubSize * ASYMMETRIC_INPUT_UB_PERCENT / 100) / BUFFER_NUM / inputTaskBytes;
    const uint64_t asymmetricOutputCapacity = (ubSize * ASYMMETRIC_OUTPUT_UB_PERCENT / 100) / outputRowBytes;
    const uint64_t mediumInputCapacity = (ubSize * MEDIUM_INPUT_UB_PERCENT / 100) / BUFFER_NUM / inputTaskBytes;
    const uint64_t mediumOutputCapacity = (ubSize * MEDIUM_OUTPUT_UB_PERCENT / 100) / outputRowBytes;
    const uint64_t balancedInputCapacity = (ubSize * BALANCED_UB_PERCENT / 100) / BUFFER_NUM / inputTaskBytes;
    const uint64_t balancedOutputCapacity = (ubSize * BALANCED_UB_PERCENT / 100) / outputRowBytes;
    if (asymmetricInputCapacity == 0 || asymmetricOutputCapacity == 0 || mediumInputCapacity == 0 ||
        mediumOutputCapacity == 0 || balancedInputCapacity == 0 || balancedOutputCapacity == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t singleLoopTask = static_cast<uint32_t>(asymmetricInputCapacity);
    uint32_t outputLoopTask = static_cast<uint32_t>(asymmetricOutputCapacity);
    uint32_t bigCoreCount = totalTask % aivNum == 0 ? aivNum : totalTask % aivNum;
    // The legacy path writes one complete output for every input batch.  The
    // direct path atomically writes one channel row per voxel member.  Counts
    // in the saved forward buffer are bounded by maxPtsPerVoxel; using half
    // that capacity models the uniform occupancy of the performance grid.
    const uint32_t activeCoreNum = std::min(aivNum, totalTask);
    const uint64_t legacyBatchCount = (singleCoreTask + balancedInputCapacity - 1) / balancedInputCapacity;
    const uint64_t legacyOutputRows = static_cast<uint64_t>(activeCoreNum) * legacyBatchCount * npoints;
    const uint64_t expectedDirectRows = totalTask64 * maxPtsPerVoxel / 2;
    const uint64_t usableUb = ubSize > UB_RESERVED ? ubSize - UB_RESERVED : 0;
    const uint64_t directTaskBytes = static_cast<uint64_t>(channelAligned + ptsRowAligned) * sizeof(float);
    const uint64_t directTaskCapacity = directTaskBytes == 0 ? 0 : usableUb / (BUFFER_NUM * directTaskBytes);
    const bool useDirect = directTaskCapacity != 0 && expectedDirectRows <= legacyOutputRows;
    if (useDirect) {
        context->SetBlockDim(activeCoreNum);
        singleCoreTask = (totalTask + activeCoreNum - 1) / activeCoreNum;
        singleLoopTask =
            static_cast<uint32_t>(std::min<uint64_t>({directTaskCapacity, singleCoreTask, MAX_COPY_BLOCKS}));
        bigCoreCount = totalTask % activeCoreNum == 0 ? activeCoreNum : totalTask % activeCoreNum;
        context->SetTilingKey(2);
    } else {
        const uint64_t asymmetricBatchCount = (singleCoreTask + asymmetricInputCapacity - 1) / asymmetricInputCapacity;
        const uint64_t balancedOutputLoopCount = (npoints + balancedOutputCapacity - 1) / balancedOutputCapacity;
        const uint64_t asymmetricOutputLoopCount = (npoints + asymmetricOutputCapacity - 1) / asymmetricOutputCapacity;
        const uint64_t mediumBatchCount = (singleCoreTask + mediumInputCapacity - 1) / mediumInputCapacity;
        const uint64_t mediumOutputLoopCount = (npoints + mediumOutputCapacity - 1) / mediumOutputCapacity;
        const uint64_t savedOutputElements =
            (legacyBatchCount - asymmetricBatchCount) * static_cast<uint64_t>(npoints) * channels;
        const uint64_t extraTaskScans = asymmetricOutputLoopCount > balancedOutputLoopCount
            ? static_cast<uint64_t>(singleCoreTask) * (asymmetricOutputLoopCount - balancedOutputLoopCount)
            : 0;
        const bool highEfficiencySavings =
            savedOutputElements >= extraTaskScans * HIGH_EFFICIENCY_SAVED_ELEMENTS_PER_EXTRA_SCAN;
        const bool useAsymmetricSplit = extraTaskScans == 0 || highEfficiencySavings ||
            (savedOutputElements >= MIN_SAVED_OUTPUT_ELEMENTS &&
                savedOutputElements >= extraTaskScans * MIN_SAVED_OUTPUT_ELEMENTS_PER_EXTRA_SCAN);
        const bool useMediumSplit = !useAsymmetricSplit && mediumBatchCount < legacyBatchCount &&
            mediumOutputLoopCount <= balancedOutputLoopCount;
        if (useAsymmetricSplit) {
            context->SetTilingKey(0);
        } else if (useMediumSplit) {
            singleLoopTask = static_cast<uint32_t>(mediumInputCapacity);
            outputLoopTask = static_cast<uint32_t>(mediumOutputCapacity);
            context->SetTilingKey(0);
        } else {
            singleLoopTask = static_cast<uint32_t>(balancedInputCapacity);
            outputLoopTask = static_cast<uint32_t>(balancedOutputCapacity);
            // Very large grids amortize the double-buffer prefetch even with
            // the balanced UB split. Smaller fallback grids use the original
            // sequential schedule for stable latency.
            context->SetTilingKey(totalTask >= LARGE_GRID_PREFETCH_MIN_TASKS ? 0 : 1);
        }
    }

    /* Set tilingData */
    RoiawareAvgpool3dGradTilingData tilingData;
    tilingData.set_singleCoreTask(singleCoreTask);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_outputLoopTask(outputLoopTask);
    tilingData.set_channels(channels);
    tilingData.set_channelAligned(channelAligned);
    tilingData.set_maxPtsPerVoxel(maxPtsPerVoxel);
    tilingData.set_maxPtsPerVoxelAligned(maxPtsPerVoxelAligned);
    tilingData.set_ptsRowElements(ptsRowElements);
    tilingData.set_ptsRowAligned(ptsRowAligned);
    tilingData.set_bigCoreCount(bigCoreCount);
    tilingData.set_totalTask(totalTask);
    tilingData.set_npoints(npoints);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

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
static ge::graphStatus InferShapeForRoiawareAvgpool3dGrad(gert::InferShapeContext *context) {
    const gert::Shape *ptsIdx = context->GetInputShape(0);
    const gert::Shape *gradOut = context->GetInputShape(1);
    gert::Shape *gradIn = context->GetOutputShape(0);
    if (ptsIdx == nullptr || gradOut == nullptr || gradIn == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t channels = gradOut->GetDim(4);
    auto runtimeAttrs = context->GetAttrs();
    if (runtimeAttrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int32_t *npoints = (int32_t *)runtimeAttrs->GetInt(5);
    if (npoints == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *gradIn = {*npoints, channels};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRoiawareAvgpool3dGrad(gert::InferDataTypeContext *context) {
    const ge::DataType value_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoiawareAvgpool3dGrad : public OpDef {
  public:
    explicit RoiawareAvgpool3dGrad(const char *name) : OpDef(name) {
        this->Input("pts_idx_of_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("grad_in")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("boxes_num").Int();
        this->Attr("out_x").Int();
        this->Attr("out_y").Int();
        this->Attr("out_z").Int();
        this->Attr("channels").Int();
        this->Attr("npoints").Int();
        this->Attr("max_pts_per_voxel").Int();

        this->SetInferShape(ge::InferShapeForRoiawareAvgpool3dGrad)
            .SetInferDataType(ge::InferDataTypeForRoiawareAvgpool3dGrad);
        this->AICore().SetTiling(optiling::TilingForRoiawareAvgpool3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(RoiawareAvgpool3dGrad);
}
