/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t NUM_FEATURE = 8;
constexpr uint32_t FEATURE_BYTES = NUM_FEATURE * sizeof(float);
constexpr uint32_t INT_BYTES = sizeof(uint32_t);
constexpr uint32_t VALUES_PER_BLOCK = 8;
constexpr uint32_t MAX_REDUCE_ROWS = 255;
constexpr uint32_t MAX_BROADCAST_ROWS = 64;
constexpr uint32_t RUNWISE_THRESHOLD = 3;
constexpr uint32_t RUN_SAMPLE_ROWS = 64;
constexpr uint32_t STRICTLY_INCREASING_MIN_ROWS = 5 * 1024;
constexpr float SAVE_VALUE = 1e-16F;
constexpr float NEG_INF = -3.402823466e+38F;
} // namespace

class KernelGraphSoftmax {
  public:
    __aicore__ inline KernelGraphSoftmax() {}

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult, GM_ADDR userWorkspace,
        const GraphSoftmaxTilingData *tiling) {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitUB();
        InitGM(src, index, softmaxResult, userWorkspace);
        InitEvents();
    }

    __aicore__ inline void Process() {
        if (totalTask_ >= STRICTLY_INCREASING_MIN_ROWS) {
            if (coreTask_ > 0) {
                CopyIndexIn();
                if (coreTask_ > 1) {
                    CopyShiftedIndexIn();
                }
                SyncMte2ToScalar();
                SyncMte2ToVector();
            }
            PublishIndexOrder();
            // Every AIV's order flag is now globally visible.
            SyncAll();
            CopyIndexOrderFlagsIn();
            SyncMte2ToScalar();

            if (AllIndexesStrictlyIncreasing()) {
                if (coreTask_ > 0) {
                    Duplicate(srcLocal_, 1.0F, coreTask_ * NUM_FEATURE);
                    PipeBarrier<PIPE_V>();
                    CopyOut();
                }
                return;
            }
        }

        // The unique-index fast path does not consume scatter workspace. Initialize
        // it only after that path misses so large sparse workspaces are not cleared
        // unnecessarily.
        InitWorkspace();
        SyncAll();
        if (totalTask_ < STRICTLY_INCREASING_MIN_ROWS) {
            if (coreTask_ > 0) {
                CopyIndexIn();
                SyncMte2ToScalar();
            }
        }

        if (coreTask_ > 0) {
            CopySrcIn();
            runsBuilt_ = ShouldBuildRuns();
            if (runsBuilt_) {
                BuildRuns();
            } else {
                useRunwisePath_ = false;
                useOrderedRunSum_ = false;
            }
            SyncMte2ToVector();
            SyncMte2ToMte3();
            ScatterMax();
        }
        SyncAll();

        if (coreTask_ > 0) {
            ComputeExp();
            ScatterSum();
        }
        SyncAll();

        if (coreTask_ > 0) {
            ComputeOutput();
            CopyOut();
        }
    }

  private:
    __aicore__ inline void InitTiling(const GraphSoftmaxTilingData *tiling) {
        maxCoreTask_ = tiling->coreTask;
        coreTask_ = maxCoreTask_;
        coreWorkspace_ = tiling->coreWorkspace;
        totalTask_ = tiling->totalTask;
        totalWorkspace_ = tiling->totalWorkspace;
        reduceTmpBytes_ = tiling->reduceTmpBytes;

        if (blockIdx_ < tiling->bigCoreCount) {
            taskOffset_ = blockIdx_ * maxCoreTask_;
        } else {
            taskOffset_ = tiling->bigCoreCount * maxCoreTask_ + (blockIdx_ - tiling->bigCoreCount) * (maxCoreTask_ - 1);
            coreTask_ = maxCoreTask_ - 1;
        }
    }

    __aicore__ inline void InitUB() {
        pipe_->InitBuffer(srcBuf_, maxCoreTask_ * FEATURE_BYTES);
        pipe_->InitBuffer(statBuf_, maxCoreTask_ * FEATURE_BYTES);
        pipe_->InitBuffer(indexBuf_, maxCoreTask_ * INT_BYTES);
        pipe_->InitBuffer(runStartBuf_, maxCoreTask_ * INT_BYTES);
        pipe_->InitBuffer(runLengthBuf_, maxCoreTask_ * INT_BYTES);
        pipe_->InitBuffer(runGroupBuf_, maxCoreTask_ * INT_BYTES);
        pipe_->InitBuffer(reduceTmpBuf_, reduceTmpBytes_);
        pipe_->InitBuffer(reducePartialBuf_, FEATURE_BYTES);
        pipe_->InitBuffer(indexOrderFlagBuf_, GetBlockNum() * VALUES_PER_BLOCK * INT_BYTES);

        srcLocal_ = srcBuf_.Get<float>();
        statLocal_ = statBuf_.Get<float>();
        indexLocal_ = indexBuf_.Get<uint32_t>();
        runStartLocal_ = runStartBuf_.Get<uint32_t>();
        runLengthLocal_ = runLengthBuf_.Get<uint32_t>();
        runGroupLocal_ = runGroupBuf_.Get<uint32_t>();
        reduceTmpLocal_ = reduceTmpBuf_.Get<uint8_t>();
        reducePartialLocal_ = reducePartialBuf_.Get<float>();
        indexOrderFlagLocal_ = indexOrderFlagBuf_.Get<uint32_t>();
    }

    __aicore__ inline void InitGM(GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult, GM_ADDR userWorkspace) {
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), totalTask_ * NUM_FEATURE);
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(index), totalTask_);
        softmaxResultGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmaxResult), totalTask_ * NUM_FEATURE);

        scatterMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace), totalWorkspace_ * NUM_FEATURE);
        scatterSumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace) + static_cast<uint64_t>(totalWorkspace_) * NUM_FEATURE,
            totalWorkspace_ * NUM_FEATURE);
        uint64_t flagOffset = static_cast<uint64_t>(totalWorkspace_) * NUM_FEATURE * 2;
        indexOrderFlagGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t *>(userWorkspace) + flagOffset, GetBlockNum() * VALUES_PER_BLOCK);
    }

    __aicore__ inline void InitWorkspace() {
        uint64_t workspaceOffset = static_cast<uint64_t>(coreWorkspace_) * blockIdx_ * NUM_FEATURE;
        GlobalTensor<float> scatterMaxInitGm = scatterMaxGm_[workspaceOffset];
        InitGlobalMemory(scatterMaxInitGm, coreWorkspace_ * NUM_FEATURE, NEG_INF);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> scatterSumInitGm = scatterSumGm_[workspaceOffset];
        InitGlobalMemory(scatterSumInitGm, coreWorkspace_ * NUM_FEATURE, 0.0F);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitEvents() {
        eventMte2S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        eventMte2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMte3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMte2Mte3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        eventMte3Mte2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        eventSMte3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        eventMte3S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        eventVS_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    }

    __aicore__ inline void CopyIndexIn() {
        DataCopyExtParams indexCopyParams{1, coreTask_ * INT_BYTES, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
        DataCopyPad(indexLocal_, indexGm_[taskOffset_], indexCopyParams, indexPadParams);
    }

    __aicore__ inline void CopyShiftedIndexIn() {
        DataCopyExtParams shiftedCopyParams{1, (coreTask_ - 1) * INT_BYTES, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> shiftedPadParams{true, 0, 0, 0};
        DataCopyPad(runStartLocal_, indexGm_[taskOffset_ + 1], shiftedCopyParams, shiftedPadParams);
    }

    __aicore__ inline void CopySrcIn() {
        DataCopy(srcLocal_, srcGm_[static_cast<uint64_t>(taskOffset_) * NUM_FEATURE], coreTask_ * NUM_FEATURE);
    }

    __aicore__ inline void CopyOut() {
        SyncVectorToMte3();
        DataCopy(
            softmaxResultGm_[static_cast<uint64_t>(taskOffset_) * NUM_FEATURE], srcLocal_, coreTask_ * NUM_FEATURE);
    }

    __aicore__ inline bool LocalIndexesStrictlyIncreasing() {
        if (coreTask_ == 0) {
            return true;
        }

        uint32_t first = indexLocal_.GetValue(0);
        if (taskOffset_ > 0 && first <= indexGm_.GetValue(taskOffset_ - 1)) {
            return false;
        }

        constexpr uint32_t COMPARE_ELEMENTS = 64;
        uint32_t adjacentCount = coreTask_ - 1;
        uint32_t vectorCount = adjacentCount / COMPARE_ELEMENTS * COMPARE_ELEMENTS;
        if (vectorCount > 0) {
            LocalTensor<int32_t> indexIntLocal = indexLocal_.ReinterpretCast<int32_t>();
            LocalTensor<int32_t> shiftedIndexLocal = runStartLocal_.ReinterpretCast<int32_t>();
            LocalTensor<uint8_t> orderMaskLocal = runLengthLocal_.ReinterpretCast<uint8_t>();
            Compare(orderMaskLocal, shiftedIndexLocal, indexIntLocal, CMPMODE::LE, vectorCount);
            SyncVectorToScalar();

            LocalTensor<uint64_t> orderMaskWords = orderMaskLocal.ReinterpretCast<uint64_t>();
            for (uint32_t word = 0; word < vectorCount / COMPARE_ELEMENTS; ++word) {
                if (orderMaskWords.GetValue(word) != 0UL) {
                    return false;
                }
            }
        }

        uint32_t tailStart = vectorCount + 1;
        uint32_t previous = indexLocal_.GetValue(tailStart - 1);
        for (uint32_t row = tailStart; row < coreTask_; ++row) {
            uint32_t current = indexLocal_.GetValue(row);
            if (current <= previous) {
                return false;
            }
            previous = current;
        }
        return true;
    }

    __aicore__ inline void PublishIndexOrder() {
        indexOrderFlagLocal_.SetValue(0, LocalIndexesStrictlyIncreasing() ? 0U : 1U);
        SyncScalarToMte3();
        DataCopy(indexOrderFlagGm_[blockIdx_ * VALUES_PER_BLOCK], indexOrderFlagLocal_, VALUES_PER_BLOCK);
        SyncMte3ToScalar();
    }

    __aicore__ inline void CopyIndexOrderFlagsIn() {
        DataCopy(indexOrderFlagLocal_, indexOrderFlagGm_, GetBlockNum() * VALUES_PER_BLOCK);
    }

    __aicore__ inline bool AllIndexesStrictlyIncreasing() {
        for (uint32_t block = 0; block < GetBlockNum(); ++block) {
            if (indexOrderFlagLocal_.GetValue(block * VALUES_PER_BLOCK) != 0U) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool ShouldBuildRuns() {
        if (coreTask_ < RUNWISE_THRESHOLD) {
            return false;
        }

        uint32_t sampleRows = coreTask_ > RUN_SAMPLE_ROWS ? RUN_SAMPLE_ROWS : coreTask_;
        uint32_t sampleRunCount = 1;
        uint32_t group = indexLocal_.GetValue(0);
        for (uint32_t row = 1; row < sampleRows; ++row) {
            uint32_t nextGroup = indexLocal_.GetValue(row);
            if (nextGroup != group) {
                ++sampleRunCount;
                group = nextGroup;
            }
        }
        return sampleRunCount * RUNWISE_THRESHOLD <= sampleRows;
    }

    __aicore__ inline void BuildRuns() {
        runCount_ = 0;
        uint32_t runStart = 0;
        uint32_t group = indexLocal_.GetValue(0);
        for (uint32_t row = 1; row < coreTask_; ++row) {
            uint32_t nextGroup = indexLocal_.GetValue(row);
            if (nextGroup != group) {
                SaveRun(runStart, row - runStart, group);
                runStart = row;
                group = nextGroup;
            }
        }
        SaveRun(runStart, coreTask_ - runStart, group);
        useRunwisePath_ = runCount_ * RUNWISE_THRESHOLD <= coreTask_;
        useOrderedRunSum_ = useRunwisePath_ && RunGroupsStrictlyIncreasing();
    }

    __aicore__ inline void SaveRun(uint32_t start, uint32_t length, uint32_t group) {
        runStartLocal_.SetValue(runCount_, start);
        runLengthLocal_.SetValue(runCount_, length);
        runGroupLocal_.SetValue(runCount_, group);
        ++runCount_;
    }

    __aicore__ inline bool RunGroupsStrictlyIncreasing() {
        for (uint32_t run = 1; run < runCount_; ++run) {
            if (runGroupLocal_.GetValue(run) <= runGroupLocal_.GetValue(run - 1)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void ReduceMaxRun(uint32_t start, uint32_t length) {
        uint32_t processed = 0;
        bool firstChunk = true;
        while (processed < length) {
            uint32_t chunkRows = length - processed;
            chunkRows = chunkRows > MAX_REDUCE_ROWS ? MAX_REDUCE_ROWS : chunkRows;
            uint32_t shape[2] = {chunkRows, NUM_FEATURE};
            if (firstChunk) {
                ReduceMax<float, Pattern::Reduce::RA, false>(statLocal_[start * NUM_FEATURE],
                    srcLocal_[(start + processed) * NUM_FEATURE], reduceTmpLocal_, shape, true);
                firstChunk = false;
            } else {
                ReduceMax<float, Pattern::Reduce::RA, false>(
                    reducePartialLocal_, srcLocal_[(start + processed) * NUM_FEATURE], reduceTmpLocal_, shape, true);
                Max(statLocal_[start * NUM_FEATURE], statLocal_[start * NUM_FEATURE], reducePartialLocal_, NUM_FEATURE);
                PipeBarrier<PIPE_V>();
            }
            processed += chunkRows;
        }
    }

    __aicore__ inline void ReduceSumRun(uint32_t start, uint32_t length) {
        uint32_t processed = 0;
        bool firstChunk = true;
        while (processed < length) {
            uint32_t chunkRows = length - processed;
            chunkRows = chunkRows > MAX_REDUCE_ROWS ? MAX_REDUCE_ROWS : chunkRows;
            uint32_t shape[2] = {chunkRows, NUM_FEATURE};
            if (firstChunk) {
                ReduceSum<float, Pattern::Reduce::RA, false>(statLocal_[start * NUM_FEATURE],
                    srcLocal_[(start + processed) * NUM_FEATURE], reduceTmpLocal_, shape, true);
                firstChunk = false;
            } else {
                ReduceSum<float, Pattern::Reduce::RA, false>(
                    reducePartialLocal_, srcLocal_[(start + processed) * NUM_FEATURE], reduceTmpLocal_, shape, true);
                Add(statLocal_[start * NUM_FEATURE], statLocal_[start * NUM_FEATURE], reducePartialLocal_, NUM_FEATURE);
                PipeBarrier<PIPE_V>();
            }
            processed += chunkRows;
        }
    }

    __aicore__ inline void ScatterMax() {
        if (!runsBuilt_) {
            SetAtomicMax<float>();
            for (uint32_t row = 0; row < coreTask_; ++row) {
                uint32_t group = indexLocal_.GetValue(row);
                DataCopy(scatterMaxGm_[static_cast<uint64_t>(group) * NUM_FEATURE], srcLocal_[row * NUM_FEATURE],
                    NUM_FEATURE);
            }
            SetAtomicNone();
            SyncMte3ToMte2();
            return;
        }

        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t length = runLengthLocal_.GetValue(run);
            if (length > 1) {
                ReduceMaxRun(runStartLocal_.GetValue(run), length);
            }
        }
        PipeBarrier<PIPE_V>();
        SyncVectorToMte3();

        SetAtomicMax<float>();
        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t start = runStartLocal_.GetValue(run);
            uint32_t group = runGroupLocal_.GetValue(run);
            if (runLengthLocal_.GetValue(run) == 1) {
                DataCopy(scatterMaxGm_[static_cast<uint64_t>(group) * NUM_FEATURE], srcLocal_[start * NUM_FEATURE],
                    NUM_FEATURE);
            } else {
                DataCopy(scatterMaxGm_[static_cast<uint64_t>(group) * NUM_FEATURE], statLocal_[start * NUM_FEATURE],
                    NUM_FEATURE);
            }
        }
        SetAtomicNone();
        SyncMte3ToMte2();
    }

    __aicore__ inline void ScatterSum() {
        if (useOrderedRunSum_) {
            for (uint32_t run = 0; run < runCount_; ++run) {
                uint32_t length = runLengthLocal_.GetValue(run);
                if (length > 1) {
                    ReduceSumRun(runStartLocal_.GetValue(run), length);
                }
            }
            PipeBarrier<PIPE_V>();
            SyncVectorToMte3();

            SetAtomicAdd<float>();
            for (uint32_t run = 0; run < runCount_; ++run) {
                uint32_t start = runStartLocal_.GetValue(run);
                uint32_t group = runGroupLocal_.GetValue(run);
                if (runLengthLocal_.GetValue(run) == 1) {
                    DataCopy(scatterSumGm_[static_cast<uint64_t>(group) * NUM_FEATURE], srcLocal_[start * NUM_FEATURE],
                        NUM_FEATURE);
                } else {
                    DataCopy(scatterSumGm_[static_cast<uint64_t>(group) * NUM_FEATURE], statLocal_[start * NUM_FEATURE],
                        NUM_FEATURE);
                }
            }
        } else {
            SyncVectorToMte3();
            SetAtomicAdd<float>();
            for (uint32_t row = 0; row < coreTask_; ++row) {
                uint32_t group = indexLocal_.GetValue(row);
                DataCopy(scatterSumGm_[static_cast<uint64_t>(group) * NUM_FEATURE], srcLocal_[row * NUM_FEATURE],
                    NUM_FEATURE);
            }
        }
        SetAtomicNone();
        SyncMte3ToMte2();
    }

    __aicore__ inline void GatherByEdge(const GlobalTensor<float> &statGm) {
        for (uint32_t row = 0; row < coreTask_; ++row) {
            uint32_t group = indexLocal_.GetValue(row);
            DataCopy(statLocal_[row * NUM_FEATURE], statGm[static_cast<uint64_t>(group) * NUM_FEATURE], NUM_FEATURE);
        }
        SyncMte2ToVector();
    }

    __aicore__ inline void GatherByRun(const GlobalTensor<float> &statGm) {
        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t start = runStartLocal_.GetValue(run);
            uint32_t group = runGroupLocal_.GetValue(run);
            DataCopy(statLocal_[start * NUM_FEATURE], statGm[static_cast<uint64_t>(group) * NUM_FEATURE], NUM_FEATURE);
        }
        SyncMte2ToVector();
    }

    __aicore__ inline void ApplyRunwiseSub() {
        BinaryRepeatParams params{1, 1, 1, 1, 1, 0};
        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t start = runStartLocal_.GetValue(run);
            uint32_t remaining = runLengthLocal_.GetValue(run);
            uint32_t processed = 0;
            while (remaining > 0) {
                uint8_t repeat = static_cast<uint8_t>(remaining > MAX_BROADCAST_ROWS ? MAX_BROADCAST_ROWS : remaining);
                Sub(srcLocal_[(start + processed) * NUM_FEATURE], srcLocal_[(start + processed) * NUM_FEATURE],
                    statLocal_[start * NUM_FEATURE], NUM_FEATURE, repeat, params);
                processed += repeat;
                remaining -= repeat;
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyRunwiseDiv() {
        BinaryRepeatParams params{1, 1, 1, 1, 1, 0};
        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t start = runStartLocal_.GetValue(run);
            Adds(statLocal_[start * NUM_FEATURE], statLocal_[start * NUM_FEATURE], SAVE_VALUE, NUM_FEATURE);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t run = 0; run < runCount_; ++run) {
            uint32_t start = runStartLocal_.GetValue(run);
            uint32_t remaining = runLengthLocal_.GetValue(run);
            uint32_t processed = 0;
            while (remaining > 0) {
                uint8_t repeat = static_cast<uint8_t>(remaining > MAX_BROADCAST_ROWS ? MAX_BROADCAST_ROWS : remaining);
                Div(srcLocal_[(start + processed) * NUM_FEATURE], srcLocal_[(start + processed) * NUM_FEATURE],
                    statLocal_[start * NUM_FEATURE], NUM_FEATURE, repeat, params);
                processed += repeat;
                remaining -= repeat;
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeExp() {
        if (useRunwisePath_) {
            GatherByRun(scatterMaxGm_);
            ApplyRunwiseSub();
        } else {
            GatherByEdge(scatterMaxGm_);
            Sub(srcLocal_, srcLocal_, statLocal_, coreTask_ * NUM_FEATURE);
            PipeBarrier<PIPE_V>();
        }
        Exp(srcLocal_, srcLocal_, coreTask_ * NUM_FEATURE);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeOutput() {
        if (useRunwisePath_) {
            GatherByRun(scatterSumGm_);
            ApplyRunwiseDiv();
        } else {
            GatherByEdge(scatterSumGm_);
            Adds(statLocal_, statLocal_, SAVE_VALUE, coreTask_ * NUM_FEATURE);
            PipeBarrier<PIPE_V>();
            Div(srcLocal_, srcLocal_, statLocal_, coreTask_ * NUM_FEATURE);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SyncMte2ToScalar() {
        SetFlag<HardEvent::MTE2_S>(eventMte2S_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S_);
    }

    __aicore__ inline void SyncMte2ToVector() {
        SetFlag<HardEvent::MTE2_V>(eventMte2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V_);
    }

    __aicore__ inline void SyncVectorToMte3() {
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
    }

    __aicore__ inline void SyncMte2ToMte3() {
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2Mte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2Mte3_);
    }

    __aicore__ inline void SyncMte3ToMte2() {
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
    }

    __aicore__ inline void SyncScalarToMte3() {
        SetFlag<HardEvent::S_MTE3>(eventSMte3_);
        WaitFlag<HardEvent::S_MTE3>(eventSMte3_);
    }

    __aicore__ inline void SyncMte3ToScalar() {
        SetFlag<HardEvent::MTE3_S>(eventMte3S_);
        WaitFlag<HardEvent::MTE3_S>(eventMte3S_);
    }

    __aicore__ inline void SyncVectorToScalar() {
        SetFlag<HardEvent::V_S>(eventVS_);
        WaitFlag<HardEvent::V_S>(eventVS_);
    }

  private:
    TPipe *pipe_;
    int32_t eventMte2S_;
    int32_t eventMte2V_;
    int32_t eventVMte3_;
    int32_t eventMte2Mte3_;
    int32_t eventMte3Mte2_;
    int32_t eventSMte3_;
    int32_t eventMte3S_;
    int32_t eventVS_;

    uint32_t blockIdx_;
    uint32_t maxCoreTask_;
    uint32_t coreTask_;
    uint32_t coreWorkspace_;
    uint32_t totalTask_;
    uint32_t totalWorkspace_;
    uint32_t taskOffset_;
    uint32_t reduceTmpBytes_;
    uint32_t runCount_;
    bool runsBuilt_;
    bool useRunwisePath_;
    bool useOrderedRunSum_;

    GlobalTensor<float> srcGm_;
    GlobalTensor<float> softmaxResultGm_;
    GlobalTensor<float> scatterMaxGm_;
    GlobalTensor<float> scatterSumGm_;
    GlobalTensor<uint32_t> indexGm_;
    GlobalTensor<uint32_t> indexOrderFlagGm_;

    TBuf<TPosition::VECCALC> srcBuf_;
    TBuf<TPosition::VECCALC> statBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> runStartBuf_;
    TBuf<TPosition::VECCALC> runLengthBuf_;
    TBuf<TPosition::VECCALC> runGroupBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> reducePartialBuf_;
    TBuf<TPosition::VECCALC> indexOrderFlagBuf_;

    LocalTensor<float> srcLocal_;
    LocalTensor<float> statLocal_;
    LocalTensor<uint32_t> indexLocal_;
    LocalTensor<uint32_t> runStartLocal_;
    LocalTensor<uint32_t> runLengthLocal_;
    LocalTensor<uint32_t> runGroupLocal_;
    LocalTensor<uint8_t> reduceTmpLocal_;
    LocalTensor<float> reducePartialLocal_;
    LocalTensor<uint32_t> indexOrderFlagLocal_;
};

extern "C" __global__ __aicore__ void graph_softmax(
    GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult, GM_ADDR workspace, GM_ADDR tilingData) {
#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GET_TILING_DATA(tiling, tilingData);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    KernelGraphSoftmax op;
    op.Init(&pipe, src, index, softmaxResult, userWorkspace, &tiling);
    op.Process();
}
