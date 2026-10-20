#include "roiaware_maxpool3d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus TilingForRoiawareMaxpool3dGrad(gert::TilingContext *context) {
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
    uint32_t ONE_ELEMENT_SIZE = sizeof(float);
    constexpr uint32_t BYTE_ALIGNED = 32;
    constexpr float INPUT_UB_RATIO = 0.8;
    constexpr float SMALL_SHAPE_INPUT_UB_RATIO = 0.6;
    constexpr uint32_t GRADOUT_BUFFER_NUMER = 3;
    constexpr uint32_t IDX_BUFFER_NUMBER = 2;
    constexpr uint32_t MAX_REPEAT_TIMES = 255;
    constexpr uint32_t COMPUTE_BYTE_SIZE = 256;
    constexpr uint32_t LEGACY_UB_RESERVED = 8 * 1024;
    // Fast path constants. One channel block is exactly one 32-byte
    // transfer for float32; keeping this granularity also makes odd channel
    // counts safe when the final block is written with DataCopyPad.
    constexpr uint32_t CHANNEL_BLOCK = 8;
    constexpr uint32_t CHANNEL_BLOCK_BYTES = CHANNEL_BLOCK * sizeof(float);
    constexpr uint32_t FAST_UB_RESERVED = 8 * 1024;
    constexpr uint32_t MAX_COPY_BLOCKS = 4095;
    // Benchmark-derived gate: the channel-block scatter path only amortizes
    // its extra UB setup on large dense cases. Smaller or fragmented shapes
    // stay on the legacy vector path to avoid regressions.
    constexpr uint32_t FAST_MIN_CHANNELS = 512;
    constexpr uint32_t FAST_MIN_NPOINTS = 640;
    constexpr uint32_t FAST2D_MIN_CHANNELS = 128;
    constexpr uint32_t FAST2D_MIN_NPOINTS = 640;
    constexpr uint32_t FAST2D_MIN_TOTAL_TASK = 128;
    constexpr uint32_t FAST2D_MAX_TASK_TILES = 8;
    constexpr uint32_t FAST2D_SMALL_CHANNEL_BLOCK = 32;
    constexpr uint32_t FAST2D_SMALL_CHANNEL_BLOCK_BYTES = FAST2D_SMALL_CHANNEL_BLOCK * sizeof(float);
    constexpr uint32_t FAST2D_LARGE_CHANNEL_BLOCK = 8;
    constexpr uint32_t FAST2D_LARGE_CHANNEL_BLOCK_BYTES = FAST2D_LARGE_CHANNEL_BLOCK * sizeof(float);
    constexpr uint32_t FAST2D_SMALL_CHANNELS_CUTOFF = 2049;
    constexpr uint32_t FAST2D_BROAD_MIN_NPOINTS = 640;
    constexpr uint32_t FAST2D_BROAD_MIN_CHANNEL_BLOCKS = 12;
    constexpr uint32_t FAST2D_BROAD_DENSE_MIN_NPOINTS = 900;
    constexpr uint32_t FAST2D_BROAD_MIN_TOTAL_TASK = 12000;
    constexpr uint32_t FAST2D_BROAD_MAX_TOTAL_TASK = 100000;
    constexpr uint32_t MEMBER_MIN_CHANNELS = 128;
    constexpr uint32_t MEMBER_BUFFER_NUM = 2;
    constexpr uint32_t MEMBER_INPUT_UB_PERCENT = 60;
    constexpr uint32_t MEMBER_SELECTED_UB_PERCENT = 30;
    constexpr uint32_t MEMBER_ROW_COST_NUMERATOR = 1;
    constexpr uint32_t MEMBER_ROW_COST_DENOMINATOR = 2;
    uint64_t ubSize;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 0: saved voxel members, 1: argmax, 2: gradOut, 0: gradIn.
    auto ptsIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto argmaxShape = context->GetInputShape(1)->GetStorageShape();
    auto gradOutShape = context->GetInputShape(2)->GetStorageShape();
    auto gradInShape = context->GetOutputShape(0)->GetStorageShape();

    if (ptsIdxShape.GetDimNum() != 5 || argmaxShape.GetDimNum() != 5 || gradOutShape.GetDimNum() != 5 ||
        gradInShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t dim = 0; dim < 5; ++dim) {
        if (argmaxShape.GetDim(dim) != gradOutShape.GetDim(dim)) {
            return ge::GRAPH_FAILED;
        }
        if (dim < 4 && ptsIdxShape.GetDim(dim) != gradOutShape.GetDim(dim)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t boxesNum = argmaxShape.GetDim(0);
    uint32_t outX = argmaxShape.GetDim(1);
    uint32_t outY = argmaxShape.GetDim(2);
    uint32_t outZ = argmaxShape.GetDim(3);
    uint32_t channels = argmaxShape.GetDim(4);
    uint32_t npoints = gradInShape.GetDim(0);
    if (boxesNum == 0 || outX == 0 || outY == 0 || outZ == 0 || channels == 0 || channels > 2048 || npoints == 0 ||
        gradInShape.GetDim(1) != channels || ptsIdxShape.GetDim(4) <= 1 || ptsIdxShape.GetDim(4) > 2048) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t ptsRowElements = ptsIdxShape.GetDim(4);
    const uint32_t maxPtsPerVoxel = ptsRowElements - 1;
    const uint32_t ptsRowAligned =
        ((ptsRowElements * sizeof(int32_t) + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / sizeof(int32_t));

    const uint64_t totalTask64 = static_cast<uint64_t>(boxesNum) * outX * outY * outZ;
    if (totalTask64 == 0 || totalTask64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalTask = static_cast<uint32_t>(totalTask64);
    uint32_t coreTask = (totalTask + aivNum - 1) / aivNum;
    uint32_t firstSmallCoreIdx = totalTask % aivNum == 0 ? aivNum : totalTask % aivNum;
    uint32_t channelAligned =
        ((channels * ONE_ELEMENT_SIZE + BYTE_ALIGNED - 1) / BYTE_ALIGNED) * (BYTE_ALIGNED / ONE_ELEMENT_SIZE);

    const float inputUbRatio = channels >= 128 && channels < 192 ? SMALL_SHAPE_INPUT_UB_RATIO : INPUT_UB_RATIO;
    uint32_t singleLoopTask = (ubSize * inputUbRatio) /
        (GRADOUT_BUFFER_NUMER * ONE_ELEMENT_SIZE + IDX_BUFFER_NUMBER * sizeof(int32_t)) / channelAligned;
    uint32_t singleLoopOutput = 1;

    uint32_t singleLoopTask_add = singleLoopTask <= MAX_REPEAT_TIMES ? singleLoopTask : MAX_REPEAT_TIMES; // for Add api
    uint32_t singleLoopTask_select =
        (COMPUTE_BYTE_SIZE * MAX_REPEAT_TIMES) / (channelAligned * ONE_ELEMENT_SIZE); // for Select Api
    singleLoopTask = singleLoopTask_add < singleLoopTask_select ? singleLoopTask_add : singleLoopTask_select;
    singleLoopTask = singleLoopTask == 0 ? 1 : singleLoopTask;

    const uint64_t legacyInputElements = static_cast<uint64_t>(singleLoopTask) * channelAligned;
    const uint64_t legacyArgmaxBytes =
        ((legacyInputElements * sizeof(int32_t) + COMPUTE_BYTE_SIZE - 1) / COMPUTE_BYTE_SIZE) * COMPUTE_BYTE_SIZE;
    const uint64_t legacyMaskBytes = BYTE_ALIGNED * (((legacyInputElements + 7) / 8 + BYTE_ALIGNED - 1) / BYTE_ALIGNED);
    const uint64_t legacyInputBytes =
        2 * legacyArgmaxBytes + (GRADOUT_BUFFER_NUMER * legacyInputElements * ONE_ELEMENT_SIZE) + legacyMaskBytes;
    if (ubSize > legacyInputBytes + LEGACY_UB_RESERVED) {
        singleLoopOutput = static_cast<uint32_t>(
            (ubSize - legacyInputBytes - LEGACY_UB_RESERVED) / (channelAligned * ONE_ELEMENT_SIZE));
        singleLoopOutput = singleLoopOutput == 0 ? 1 : singleLoopOutput;
        singleLoopOutput = singleLoopOutput > npoints ? npoints : singleLoopOutput;
    }

    // Fast channel-block scatter path. For small tensors the legacy vector
    // compare path has lower scalar/launch overhead, so retain it as a safe
    // fallback. The fast path needs one [npoints, 8] output tile and one
    // argmax/grad-out tile in UB; if that does not fit, use the legacy key.
    const uint32_t channelBlockNum = (channels + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    const uint64_t fastOutputBytes = static_cast<uint64_t>(npoints) * CHANNEL_BLOCK_BYTES;
    const uint64_t fastAvailableUb = ubSize > FAST_UB_RESERVED ? ubSize - FAST_UB_RESERVED : 0;
    const uint64_t fastMinInputBytes = 2 * CHANNEL_BLOCK_BYTES;
    const bool fastUbFits = fastAvailableUb > fastOutputBytes + fastMinInputBytes;
    const uint32_t fast2DSmallChannelBlockNum =
        (channels + FAST2D_SMALL_CHANNEL_BLOCK - 1) / FAST2D_SMALL_CHANNEL_BLOCK;
    const uint64_t fast2DSmallOutputBytes = static_cast<uint64_t>(npoints) * FAST2D_SMALL_CHANNEL_BLOCK_BYTES;
    const uint64_t fast2DSmallMinInputBytes = 2 * FAST2D_SMALL_CHANNEL_BLOCK_BYTES;
    const bool fast2DSmallUbFits = fastAvailableUb > fast2DSmallOutputBytes + fast2DSmallMinInputBytes;
    // On a non-trivial task grid, splitting one channel block across three
    // or more cores duplicates too much of the [npoints, channel-block]
    // atomic copy-out. Keep those shapes on the vectorized fallback; tiny
    // grids still need the extra task-axis parallelism to fill the AIVs.
    const bool fast2DSmallHasEnoughChannelParallelism =
        totalTask <= 512 || static_cast<uint64_t>(fast2DSmallChannelBlockNum) * 2 >= aivNum;
    const uint32_t fast2DLargeChannelBlockNum =
        (channels + FAST2D_LARGE_CHANNEL_BLOCK - 1) / FAST2D_LARGE_CHANNEL_BLOCK;
    const uint64_t fast2DLargeOutputBytes = static_cast<uint64_t>(npoints) * FAST2D_LARGE_CHANNEL_BLOCK_BYTES;
    const uint64_t fast2DLargeMinInputBytes = 2 * FAST2D_LARGE_CHANNEL_BLOCK_BYTES;
    const bool fast2DLargeUbFits = fastAvailableUb > fast2DLargeOutputBytes + fast2DLargeMinInputBytes;
    const bool scatterWorkFits = static_cast<uint64_t>(totalTask) <= static_cast<uint64_t>(npoints) * 64;
    const bool lowDensityGrid = static_cast<uint64_t>(totalTask) <= static_cast<uint64_t>(npoints) * 2;
    const bool useFastPath = channels >= FAST_MIN_CHANNELS && npoints >= FAST_MIN_NPOINTS && scatterWorkFits &&
        totalTask > 4096 && npoints <= MAX_COPY_BLOCKS && totalTask >= CHANNEL_BLOCK && channelBlockNum != 0 &&
        fastUbFits;
    const bool useLowDensityFast2DSmallPath = totalTask >= FAST2D_MIN_TOTAL_TASK && totalTask <= 4096 &&
        lowDensityGrid && fast2DSmallHasEnoughChannelParallelism;
    // The 32-channel path also wins on moderately sized task grids when
    // enough point rows amortize the local scatter and output-tile setup.
    // Cap totalTask because the scalar UB scatter eventually loses to the
    // vectorized compare/select path as the grid grows.
    const bool useBroadFast2DSmallPath = npoints >= FAST2D_BROAD_MIN_NPOINTS &&
        fast2DSmallChannelBlockNum >= FAST2D_BROAD_MIN_CHANNEL_BLOCKS && totalTask <= FAST2D_BROAD_MAX_TOTAL_TASK &&
        (totalTask < FAST2D_MIN_TOTAL_TASK || npoints >= FAST2D_BROAD_DENSE_MIN_NPOINTS ||
            totalTask >= FAST2D_BROAD_MIN_TOTAL_TASK);
    const bool useFast2DSmallPath = channels >= FAST2D_MIN_CHANNELS && channels < FAST2D_SMALL_CHANNELS_CUTOFF &&
        npoints >= FAST2D_MIN_NPOINTS && npoints <= MAX_COPY_BLOCKS && fast2DSmallChannelBlockNum != 0 &&
        fast2DSmallChannelBlockNum < aivNum && fast2DSmallUbFits &&
        (useLowDensityFast2DSmallPath || useBroadFast2DSmallPath);
    const bool useFast2DLargePath = channels >= FAST2D_MIN_CHANNELS && npoints >= FAST2D_MIN_NPOINTS &&
        scatterWorkFits && npoints <= MAX_COPY_BLOCKS && totalTask >= FAST2D_MIN_TOTAL_TASK &&
        fast2DLargeChannelBlockNum != 0 && fast2DLargeChannelBlockNum < aivNum && fast2DLargeUbFits;
    // Forward max pooling guarantees every non-negative argmax is one of the
    // saved voxel members.  Scanning those members with vector Compare/Select
    // is cheaper than scanning all npoints when the expected member-row count
    // is well below the legacy output traffic.
    const uint32_t activeCoreNum = totalTask < aivNum ? totalTask : aivNum;
    const uint64_t legacyBatchCount = (coreTask + singleLoopTask - 1) / singleLoopTask;
    const uint64_t legacyOutputRows = static_cast<uint64_t>(activeCoreNum) * legacyBatchCount * npoints;
    const uint64_t expectedMemberRows = totalTask64 * maxPtsPerVoxel / 2;
    const uint64_t memberInputTaskBytes = static_cast<uint64_t>(2 * channelAligned + ptsRowAligned) * ONE_ELEMENT_SIZE;
    const uint64_t memberInputCapacity =
        (fastAvailableUb * MEMBER_INPUT_UB_PERCENT / 100) / MEMBER_BUFFER_NUM / memberInputTaskBytes;
    const uint64_t memberSelectedCapacity =
        (fastAvailableUb * MEMBER_SELECTED_UB_PERCENT / 100) / (channelAligned * ONE_ELEMENT_SIZE);
    const bool useMemberPath = channels >= MEMBER_MIN_CHANNELS && memberInputCapacity != 0 &&
        memberSelectedCapacity != 0 &&
        expectedMemberRows * MEMBER_ROW_COST_DENOMINATOR <= legacyOutputRows * MEMBER_ROW_COST_NUMERATOR;

    if (useFastPath) {
        const uint32_t usedCoreNum = channelBlockNum < aivNum ? channelBlockNum : aivNum;
        context->SetBlockDim(usedCoreNum);

        uint64_t fastTaskBytes = fastAvailableUb - fastOutputBytes;
        uint32_t fastTaskBatch = static_cast<uint32_t>(fastTaskBytes / (2 * CHANNEL_BLOCK_BYTES));
        fastTaskBatch = fastTaskBatch == 0 ? 1 : fastTaskBatch;
        fastTaskBatch = fastTaskBatch > MAX_COPY_BLOCKS ? MAX_COPY_BLOCKS : fastTaskBatch;
        fastTaskBatch = fastTaskBatch > totalTask ? totalTask : fastTaskBatch;
        singleLoopTask = fastTaskBatch;
        singleLoopOutput = npoints;
        context->SetTilingKey(1);
    } else if (useFast2DSmallPath) {
        uint32_t taskTileNum = (aivNum + fast2DSmallChannelBlockNum - 1) / fast2DSmallChannelBlockNum;
        taskTileNum = taskTileNum < 2 ? 2 : taskTileNum;
        taskTileNum = taskTileNum > FAST2D_MAX_TASK_TILES ? FAST2D_MAX_TASK_TILES : taskTileNum;
        taskTileNum = taskTileNum > totalTask ? totalTask : taskTileNum;
        const uint32_t workItemNum = fast2DSmallChannelBlockNum * taskTileNum;
        const uint32_t usedCoreNum = workItemNum < aivNum ? workItemNum : aivNum;
        context->SetBlockDim(usedCoreNum);

        uint64_t fastTaskBytes = fastAvailableUb - fast2DSmallOutputBytes;
        uint32_t fastTaskBatch = static_cast<uint32_t>(fastTaskBytes / (2 * FAST2D_SMALL_CHANNEL_BLOCK_BYTES));
        fastTaskBatch = fastTaskBatch == 0 ? 1 : fastTaskBatch;
        fastTaskBatch = fastTaskBatch > MAX_COPY_BLOCKS ? MAX_COPY_BLOCKS : fastTaskBatch;
        singleLoopTask = fastTaskBatch;
        singleLoopOutput = npoints;
        coreTask = (totalTask + taskTileNum - 1) / taskTileNum;
        firstSmallCoreIdx = taskTileNum;
        context->SetTilingKey(3);
    } else if (useFast2DLargePath) {
        uint32_t taskTileNum = (aivNum + fast2DLargeChannelBlockNum - 1) / fast2DLargeChannelBlockNum;
        taskTileNum = taskTileNum < 2 ? 2 : taskTileNum;
        taskTileNum = taskTileNum > FAST2D_MAX_TASK_TILES ? FAST2D_MAX_TASK_TILES : taskTileNum;
        taskTileNum = taskTileNum > totalTask ? totalTask : taskTileNum;
        const uint32_t workItemNum = fast2DLargeChannelBlockNum * taskTileNum;
        const uint32_t usedCoreNum = workItemNum < aivNum ? workItemNum : aivNum;
        context->SetBlockDim(usedCoreNum);

        uint64_t fastTaskBytes = fastAvailableUb - fast2DLargeOutputBytes;
        uint32_t fastTaskBatch = static_cast<uint32_t>(fastTaskBytes / (2 * FAST2D_LARGE_CHANNEL_BLOCK_BYTES));
        fastTaskBatch = fastTaskBatch == 0 ? 1 : fastTaskBatch;
        fastTaskBatch = fastTaskBatch > MAX_COPY_BLOCKS ? MAX_COPY_BLOCKS : fastTaskBatch;
        singleLoopTask = fastTaskBatch;
        singleLoopOutput = npoints;
        coreTask = (totalTask + taskTileNum - 1) / taskTileNum;
        firstSmallCoreIdx = taskTileNum;
        context->SetTilingKey(2);
    } else if (useMemberPath) {
        context->SetBlockDim(activeCoreNum);
        coreTask = (totalTask + activeCoreNum - 1) / activeCoreNum;
        firstSmallCoreIdx = totalTask % activeCoreNum == 0 ? activeCoreNum : totalTask % activeCoreNum;
        singleLoopTask = static_cast<uint32_t>(memberInputCapacity);
        singleLoopTask = singleLoopTask > coreTask ? coreTask : singleLoopTask;
        singleLoopTask = singleLoopTask > MAX_COPY_BLOCKS ? MAX_COPY_BLOCKS : singleLoopTask;
        singleLoopOutput = static_cast<uint32_t>(memberSelectedCapacity);
        singleLoopOutput = singleLoopOutput > maxPtsPerVoxel ? maxPtsPerVoxel : singleLoopOutput;
        context->SetTilingKey(4);
    } else {
        context->SetTilingKey(0);
    }
    /* Set tilingData */
    RoiawareMaxpool3dGradTilingData tilingData;

    tilingData.set_totalTask(totalTask);
    tilingData.set_coreTask(coreTask);
    tilingData.set_firstSmallCoreIdx(firstSmallCoreIdx);
    tilingData.set_singleLoopTask(singleLoopTask);
    tilingData.set_singleLoopOutput(singleLoopOutput);
    tilingData.set_channelAligned(channelAligned);
    tilingData.set_channels(channels);
    tilingData.set_npoints(npoints);
    tilingData.set_ptsRowElements(ptsRowElements);
    tilingData.set_ptsRowAligned(ptsRowAligned);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

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
static ge::graphStatus InferShapeForRoiawareMaxpool3dGrad(gert::InferShapeContext *context) {
    const gert::Shape *argmax = context->GetInputShape(1);
    const gert::Shape *gradOut = context->GetInputShape(2);
    gert::Shape *gradIn = context->GetOutputShape(0);
    if (argmax == nullptr || gradOut == nullptr || gradIn == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t channels = argmax->GetDim(4);
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

static ge::graphStatus InferDataTypeForRoiawareMaxpool3dGrad(gert::InferDataTypeContext *context) {
    const ge::DataType value_dtype = context->GetInputDataType(2);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class RoiawareMaxpool3dGrad : public OpDef {
  public:
    explicit RoiawareMaxpool3dGrad(const char *name) : OpDef(name) {
        this->Input("pts_idx_of_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("argmax")
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

        this->SetInferShape(ge::InferShapeForRoiawareMaxpool3dGrad)
            .SetInferDataType(ge::InferDataTypeForRoiawareMaxpool3dGrad);
        this->AICore().SetTiling(optiling::TilingForRoiawareMaxpool3dGrad);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(RoiawareMaxpool3dGrad);
}
