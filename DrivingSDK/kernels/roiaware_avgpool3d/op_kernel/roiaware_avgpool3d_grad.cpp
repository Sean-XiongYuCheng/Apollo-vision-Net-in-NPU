#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_ALIGN_BYTE = 32;
constexpr uint32_t kDirectBufferNum = 2;
constexpr uint32_t kMaxConsecutiveSimdChannels = 768;

template <typename dataType> class RoiAwareAvgpool3dGrad {
  public:
    __aicore__ inline RoiAwareAvgpool3dGrad() {}

    __aicore__ inline void Init(TPipe *pipe, const GM_ADDR pts_idx_of_voxels, const GM_ADDR grad_out,
        const GM_ADDR grad_in, const RoiawareAvgpool3dGradTilingData *tiling) {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitGM(pts_idx_of_voxels, grad_out, grad_in);
        InitUB();
    }

    __aicore__ inline void Process() {
        if (singleCoreTask_ == 0) {
            return;
        }

        uint32_t processedTask = 0;
        uint32_t currentTaskCount = min(singleCoreTask_, singleLoopTask_);
        CopyInNTask(currentTaskCount, globalTaskIdx_);
        while (processedTask < singleCoreTask_) {
            const uint32_t nextTaskStart = processedTask + currentTaskCount;
            const uint32_t nextTaskCount =
                nextTaskStart < singleCoreTask_ ? min(singleLoopTask_, singleCoreTask_ - nextTaskStart) : 0;
            if (nextTaskCount != 0) {
                CopyInNTask(nextTaskCount, globalTaskIdx_ + nextTaskStart);
            }
            ComputeNTask(currentTaskCount, globalTaskIdx_ + processedTask);
            processedTask = nextTaskStart;
            currentTaskCount = nextTaskCount;
        }
    }

    __aicore__ inline void ProcessSequential() {
        for (uint32_t localTaskIdx = 0; localTaskIdx < singleCoreTask_; localTaskIdx += singleLoopTask_) {
            const uint32_t currentTaskCount = min(singleCoreTask_ - localTaskIdx, singleLoopTask_);
            const uint32_t currentGlobalTaskIdx = globalTaskIdx_ + localTaskIdx;
            CopyInNTask(currentTaskCount, currentGlobalTaskIdx);
            ComputeNTask(currentTaskCount, currentGlobalTaskIdx);
        }
    }

  protected:
    uint64_t blkIdx_;
    TPipe *pipe_;
    GlobalTensor<dataType> goutGm_, ginGm_;
    GlobalTensor<int32_t> ptsIdxOfsVoxelsGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> goutQue_, ptsIdxQue_;
    TBuf<TPosition::VECCALC> ginBuf_, iterIdxBuf_;
    LocalTensor<int32_t> iterIdxLocal_;

    // tiling
    uint32_t singleCoreTask_, singleLoopTask_, outputLoopTask_;
    uint32_t channels_, channelAligned_, maxPtsPerVoxel_, maxPtsPerVoxelAligned_;
    uint32_t globalTaskIdx_;
    uint32_t totalTask_, npoints_;

    // copy params
    DataCopyExtParams gradOutCopyInParams, ptsIdxCopyInParams, ginCopyOutParams;
    DataCopyPadExtParams<dataType> gradOutCopyInPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> ptsIdxCopyInPadParams{false, 0, 0, 0};

  private:
    __aicore__ inline void InitTiling(const RoiawareAvgpool3dGradTilingData *tiling) {
        singleCoreTask_ = tiling->singleCoreTask;
        singleLoopTask_ = tiling->singleLoopTask;
        outputLoopTask_ = tiling->outputLoopTask;
        channels_ = tiling->channels;
        channelAligned_ = tiling->channelAligned;
        maxPtsPerVoxel_ = tiling->maxPtsPerVoxel;
        maxPtsPerVoxelAligned_ = tiling->maxPtsPerVoxelAligned;
        totalTask_ = tiling->totalTask;
        npoints_ = tiling->npoints;

        if (blkIdx_ < tiling->bigCoreCount) {
            // big core
            globalTaskIdx_ = singleCoreTask_ * blkIdx_;
        } else {
            // small core
            globalTaskIdx_ =
                (singleCoreTask_ * tiling->bigCoreCount) + (singleCoreTask_ - 1) * (blkIdx_ - tiling->bigCoreCount);
            singleCoreTask_ = singleCoreTask_ - 1;
        }
    }

    __aicore__ inline void InitGM(const GM_ADDR pts_idx_of_voxels, const GM_ADDR grad_out, const GM_ADDR grad_in) {
        ptsIdxOfsVoxelsGm_.SetGlobalBuffer((__gm__ int32_t *)pts_idx_of_voxels,
            static_cast<uint64_t>(totalTask_) * maxPtsPerVoxelAligned_ * sizeof(int32_t));
        goutGm_.SetGlobalBuffer(
            (__gm__ dataType *)grad_out, static_cast<uint64_t>(totalTask_) * channelAligned_ * sizeof(dataType));
        ginGm_.SetGlobalBuffer(
            (__gm__ dataType *)grad_in, static_cast<uint64_t>(npoints_) * channelAligned_ * sizeof(dataType));
    }

    __aicore__ inline void InitUB() {
        pipe_->InitBuffer(goutQue_, BUFFER_NUM, singleLoopTask_ * channelAligned_ * sizeof(dataType));
        pipe_->InitBuffer(ptsIdxQue_, BUFFER_NUM, singleLoopTask_ * maxPtsPerVoxelAligned_ * sizeof(int32_t));
        pipe_->InitBuffer(ginBuf_, outputLoopTask_ * channelAligned_ * sizeof(dataType));
        pipe_->InitBuffer(iterIdxBuf_, min(singleLoopTask_, totalTask_) * sizeof(int32_t));
        iterIdxLocal_ = iterIdxBuf_.AllocTensor<int32_t>();
    }

    __aicore__ inline void CopyInNTask(const uint32_t &taskCount, const uint64_t &globalTaskIdx) {
        LocalTensor<dataType> goutLocal = goutQue_.AllocTensor<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.AllocTensor<int32_t>();

        uint64_t globalGradOutIdx = globalTaskIdx * channels_;
        uint64_t globalPtsIdxOutIdx = globalTaskIdx * (maxPtsPerVoxel_ + 1);

        gradOutCopyInParams = {static_cast<uint16_t>(taskCount), channels_ * sizeof(dataType), 0, 0, 0};
        DataCopyPad(goutLocal, goutGm_[globalGradOutIdx], gradOutCopyInParams, gradOutCopyInPadParams);

        ptsIdxCopyInParams = {static_cast<uint16_t>(taskCount), maxPtsPerVoxel_ * sizeof(int32_t),
            static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0};
        DataCopyPad(ptsIdxLocal, ptsIdxOfsVoxelsGm_[globalPtsIdxOutIdx + 1], ptsIdxCopyInParams, ptsIdxCopyInPadParams);

        goutQue_.EnQue<dataType>(goutLocal);
        ptsIdxQue_.EnQue<int32_t>(ptsIdxLocal);
    }

    __aicore__ inline void ComputeNTask(const uint32_t &taskCount, const uint64_t &globalTaskIdx) {
        LocalTensor<dataType> goutLocal = goutQue_.DeQue<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.DeQue<int32_t>();
        LocalTensor<dataType> ginLocal = ginBuf_.AllocTensor<dataType>();

        Duplicate<int32_t>(iterIdxLocal_, static_cast<int32_t>(0), min(totalTask_, singleLoopTask_));

        int32_t gradInChannelsIdx = 0;
        for (; gradInChannelsIdx < npoints_; gradInChannelsIdx += outputLoopTask_) {
            int32_t eventID1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventID1);
            WaitFlag<HardEvent::MTE3_V>(eventID1);

            Duplicate<dataType>(ginLocal, static_cast<dataType>(0), outputLoopTask_ * channelAligned_);
            uint32_t curLoopOutputTaskCount = min(outputLoopTask_, npoints_ - gradInChannelsIdx);

            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < taskCount; i++) {
                int32_t curPtsIdxCount = ptsIdxOfsVoxelsGm_.GetValue((globalTaskIdx + i) * (maxPtsPerVoxel_ + 1));
                if (gradInChannelsIdx == 0) {
                    Muls(goutLocal[i * channelAligned_], goutLocal[i * channelAligned_],
                        static_cast<dataType>(1.0f / max(curPtsIdxCount, 1)), channelAligned_);
                }
                int32_t curPtsIdx = iterIdxLocal_.GetValue(i);
                ComputeOneTask(i, curPtsIdxCount, gradInChannelsIdx, curPtsIdx, curLoopOutputTaskCount, ptsIdxLocal,
                    goutLocal, ginLocal);
                iterIdxLocal_.SetValue(i, curPtsIdx);
            }

            int32_t eventID2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID2);
            WaitFlag<HardEvent::V_MTE3>(eventID2);

            ginCopyOutParams = {static_cast<uint16_t>(curLoopOutputTaskCount),
                static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
            SetAtomicAdd<dataType>();
            DataCopyPad(ginGm_[static_cast<uint64_t>(gradInChannelsIdx) * channels_], ginLocal, ginCopyOutParams);
            SetAtomicNone();
        }

        goutQue_.FreeTensor(goutLocal);
        ptsIdxQue_.FreeTensor(ptsIdxLocal);
    }

    __aicore__ inline void ComputeOneTask(const int32_t &goutLocalTaskIdx, const int32_t &curPtsIdxCount,
        const int32_t &gradInChannelsIdx, int32_t &curPtsIdx, const uint32_t &curLoopOutputTaskCount,
        const LocalTensor<int32_t> &ptsIdxLocal, const LocalTensor<dataType> &goutLocal,
        LocalTensor<dataType> &ginLocal) {
        if (channelAligned_ > kMaxConsecutiveSimdChannels) {
            ComputeOneTaskScalar(goutLocalTaskIdx, curPtsIdxCount, gradInChannelsIdx, curPtsIdx, curLoopOutputTaskCount,
                ptsIdxLocal, goutLocal, ginLocal);
            return;
        }

        uint32_t offset = goutLocalTaskIdx * maxPtsPerVoxelAligned_;
        while (curPtsIdx < curPtsIdxCount) {
            uint32_t outputIdx = ptsIdxLocal.GetValue(offset + curPtsIdx);
            if (outputIdx >= gradInChannelsIdx + curLoopOutputTaskCount) {
                break;
            }
            if (outputIdx < static_cast<uint32_t>(gradInChannelsIdx)) {
                curPtsIdx++;
                continue;
            }

            // Forward appends point indices while scanning points in ascending
            // order.  When the members in this output tile form a consecutive
            // run, add the same grad_out row to all destination rows with one
            // strided SIMD operation per channel chunk instead of issuing one
            // vector operation from Scalar for every member.
            const uint32_t remainingMembers = static_cast<uint32_t>(curPtsIdxCount - curPtsIdx);
            const uint32_t remainingOutputRows = gradInChannelsIdx + curLoopOutputTaskCount - outputIdx;
            const uint32_t consecutiveCount = min(remainingMembers, remainingOutputRows);
            if (consecutiveCount > 1) {
                const uint32_t lastOutputIdx = ptsIdxLocal.GetValue(offset + curPtsIdx + consecutiveCount - 1);
                if (lastOutputIdx == outputIdx + consecutiveCount - 1) {
                    AccumulateConsecutiveRows(
                        outputIdx - gradInChannelsIdx, consecutiveCount, goutLocalTaskIdx, goutLocal, ginLocal);
                    curPtsIdx += consecutiveCount;
                    continue;
                }
            }

            outputIdx -= gradInChannelsIdx;
            Add(ginLocal[outputIdx * channelAligned_], goutLocal[goutLocalTaskIdx * channelAligned_],
                ginLocal[outputIdx * channelAligned_], channelAligned_);
            curPtsIdx++;
        }
    }

    __aicore__ inline void ComputeOneTaskScalar(const int32_t &goutLocalTaskIdx, const int32_t &curPtsIdxCount,
        const int32_t &gradInChannelsIdx, int32_t &curPtsIdx, const uint32_t &curLoopOutputTaskCount,
        const LocalTensor<int32_t> &ptsIdxLocal, const LocalTensor<dataType> &goutLocal,
        LocalTensor<dataType> &ginLocal) {
        const uint32_t offset = goutLocalTaskIdx * maxPtsPerVoxelAligned_;
        while (curPtsIdx < curPtsIdxCount) {
            uint32_t outputIdx = ptsIdxLocal.GetValue(offset + curPtsIdx);
            if (outputIdx >= gradInChannelsIdx + curLoopOutputTaskCount) {
                break;
            }
            if (outputIdx >= static_cast<uint32_t>(gradInChannelsIdx)) {
                outputIdx -= gradInChannelsIdx;
                Add(ginLocal[outputIdx * channelAligned_], goutLocal[goutLocalTaskIdx * channelAligned_],
                    ginLocal[outputIdx * channelAligned_], channelAligned_);
            }
            curPtsIdx++;
        }
    }

    __aicore__ inline void AccumulateConsecutiveRows(uint32_t outputRow, uint32_t rowCount, uint32_t goutLocalTaskIdx,
        const LocalTensor<dataType> &goutLocal, LocalTensor<dataType> &ginLocal) {
        constexpr uint32_t VECTOR_ELEMENTS = 64;
        constexpr uint32_t MAX_REPEAT_TIMES = 255;
        const uint8_t rowStride = static_cast<uint8_t>(channelAligned_ / (BUFFER_ALIGN_BYTE / sizeof(dataType)));
        const BinaryRepeatParams repeatParams{1, 1, 1, rowStride, rowStride, 0};

        for (uint32_t rowOffset = 0; rowOffset < rowCount; rowOffset += MAX_REPEAT_TIMES) {
            const uint8_t repeatTimes = static_cast<uint8_t>(min(MAX_REPEAT_TIMES, rowCount - rowOffset));
            for (uint32_t channelOffset = 0; channelOffset < channels_; channelOffset += VECTOR_ELEMENTS) {
                const uint64_t mask = min(VECTOR_ELEMENTS, channels_ - channelOffset);
                const uint32_t ginOffset = (outputRow + rowOffset) * channelAligned_ + channelOffset;
                const uint32_t goutOffset = goutLocalTaskIdx * channelAligned_ + channelOffset;
                Add(ginLocal[ginOffset], ginLocal[ginOffset], goutLocal[goutOffset], mask, repeatTimes, repeatParams);
            }
        }
    }
};

template <bool fullOutput> class RoiawareAvgpool3dGradKernel {
  public:
    __aicore__ inline RoiawareAvgpool3dGradKernel() {}

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR ptsIdxOfVoxels, GM_ADDR gradOut, GM_ADDR gradIn,
        const RoiawareAvgpool3dGradTilingData *tiling) {
        pipe_ = pipe;
        InitTiling(tiling);
        ptsIdxGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(ptsIdxOfVoxels), static_cast<uint64_t>(totalTask_) * ptsRowElements_);
        gradOutGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(gradOut), static_cast<uint64_t>(totalTask_) * channels_);
        gradInGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(gradIn), static_cast<uint64_t>(npoints_) * channels_);

        pipe_->InitBuffer(gradOutQueue_, kDirectBufferNum, singleLoopTask_ * channelAligned_ * sizeof(float));
        pipe_->InitBuffer(ptsIdxQueue_, kDirectBufferNum, singleLoopTask_ * ptsRowAligned_ * sizeof(int32_t));
        if (fullOutput) {
            pipe_->InitBuffer(gradInBuffer_, npoints_ * channelAligned_ * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (fullOutput) {
            ProcessFullOutput();
        } else {
            ProcessDirectScatter();
        }
    }

  private:
    __aicore__ inline void ProcessDirectScatter() {
        if (coreTask_ == 0) {
            return;
        }

        uint32_t processedTask = 0;
        uint32_t currentTaskCount = min(singleLoopTask_, coreTask_);
        CopyIn(coreTaskStart_, currentTaskCount);
        while (processedTask < coreTask_) {
            const uint32_t nextTaskStart = processedTask + currentTaskCount;
            const uint32_t nextTaskCount =
                nextTaskStart < coreTask_ ? min(singleLoopTask_, coreTask_ - nextTaskStart) : 0;
            ProcessDirectBatch(currentTaskCount, coreTaskStart_ + nextTaskStart, nextTaskCount);
            processedTask = nextTaskStart;
            currentTaskCount = nextTaskCount;
        }
    }

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, kDirectBufferNum> gradOutQueue_;
    TQue<QuePosition::VECIN, kDirectBufferNum> ptsIdxQueue_;
    TBuf<TPosition::VECCALC> gradInBuffer_;
    GlobalTensor<float> gradOutGm_;
    GlobalTensor<float> gradInGm_;
    GlobalTensor<int32_t> ptsIdxGm_;

    uint32_t coreTask_ = 0;
    uint32_t coreTaskStart_ = 0;
    uint32_t singleLoopTask_ = 0;
    uint32_t channels_ = 0;
    uint32_t channelAligned_ = 0;
    uint32_t ptsRowElements_ = 0;
    uint32_t ptsRowAligned_ = 0;
    uint32_t totalTask_ = 0;
    uint32_t npoints_ = 0;

    DataCopyPadExtParams<float> gradOutPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> ptsIdxPadParams_{false, 0, 0, 0};

    __aicore__ inline void InitTiling(const RoiawareAvgpool3dGradTilingData *tiling) {
        coreTask_ = tiling->singleCoreTask;
        singleLoopTask_ = tiling->singleLoopTask;
        channels_ = tiling->channels;
        channelAligned_ = tiling->channelAligned;
        ptsRowElements_ = tiling->ptsRowElements;
        ptsRowAligned_ = tiling->ptsRowAligned;
        totalTask_ = tiling->totalTask;
        npoints_ = tiling->npoints;

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx < tiling->bigCoreCount) {
            coreTaskStart_ = blockIdx * coreTask_;
        } else {
            coreTaskStart_ = tiling->bigCoreCount * coreTask_ + (blockIdx - tiling->bigCoreCount) * (coreTask_ - 1);
            --coreTask_;
        }
    }

    __aicore__ inline void CopyIn(uint32_t globalTaskStart, uint32_t taskCount) {
        LocalTensor<float> gradOutLocal = gradOutQueue_.AllocTensor<float>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQueue_.AllocTensor<int32_t>();

        const DataCopyExtParams gradOutCopyParams{
            static_cast<uint16_t>(taskCount), static_cast<uint32_t>(channels_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(gradOutLocal, gradOutGm_[static_cast<uint64_t>(globalTaskStart) * channels_], gradOutCopyParams,
            gradOutPadParams_);

        const DataCopyExtParams ptsIdxCopyParams{
            static_cast<uint16_t>(taskCount), static_cast<uint32_t>(ptsRowElements_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(ptsIdxLocal, ptsIdxGm_[static_cast<uint64_t>(globalTaskStart) * ptsRowElements_], ptsIdxCopyParams,
            ptsIdxPadParams_);

        gradOutQueue_.EnQue<float>(gradOutLocal);
        ptsIdxQueue_.EnQue<int32_t>(ptsIdxLocal);
    }

    __aicore__ inline int32_t PointCount(const LocalTensor<int32_t> &ptsIdxLocal, uint32_t taskIdx) const {
        const int32_t count = ptsIdxLocal.GetValue(taskIdx * ptsRowAligned_);
        if (count <= 0) {
            return 0;
        }
        return min(count, static_cast<int32_t>(ptsRowElements_ - 1));
    }

    __aicore__ inline void ProcessDirectBatch(uint32_t taskCount, uint32_t nextTaskStart, uint32_t nextTaskCount) {
        LocalTensor<float> gradOutLocal = gradOutQueue_.DeQue<float>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQueue_.DeQue<int32_t>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (nextTaskCount != 0) {
            CopyIn(nextTaskStart, nextTaskCount);
        }

        bool hasContribution = false;
        for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
            const int32_t pointCount = PointCount(ptsIdxLocal, taskIdx);
            if (pointCount == 0) {
                continue;
            }
            Muls(gradOutLocal[taskIdx * channelAligned_], gradOutLocal[taskIdx * channelAligned_],
                1.0f / static_cast<float>(pointCount), channels_);
            hasContribution = true;
        }

        if (hasContribution) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetAtomicAdd<float>();
            for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
                const int32_t pointCount = PointCount(ptsIdxLocal, taskIdx);
                const uint32_t ptsOffset = taskIdx * ptsRowAligned_ + 1;
                for (int32_t pointOffset = 0; pointOffset < pointCount; ++pointOffset) {
                    const int32_t pointIdx = ptsIdxLocal.GetValue(ptsOffset + pointOffset);
                    if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(npoints_)) {
                        continue;
                    }
                    const DataCopyExtParams gradInCopyParams{
                        1, static_cast<uint32_t>(channels_ * sizeof(float)), 0, 0, 0};
                    DataCopyPad(gradInGm_[static_cast<uint64_t>(pointIdx) * channels_],
                        gradOutLocal[taskIdx * channelAligned_], gradInCopyParams);
                }
            }
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        gradOutQueue_.FreeTensor(gradOutLocal);
        ptsIdxQueue_.FreeTensor(ptsIdxLocal);
    }

    __aicore__ inline void ProcessFullOutput() {
        if (coreTask_ == 0) {
            return;
        }

        LocalTensor<float> gradInLocal = gradInBuffer_.Get<float>();
        Duplicate(gradInLocal, 0.0f, npoints_ * channelAligned_);

        bool hasContribution = false;
        uint32_t processedTask = 0;
        uint32_t currentTaskCount = min(singleLoopTask_, coreTask_);
        CopyIn(coreTaskStart_, currentTaskCount);
        while (processedTask < coreTask_) {
            const uint32_t nextTaskStart = processedTask + currentTaskCount;
            const uint32_t nextTaskCount =
                nextTaskStart < coreTask_ ? min(singleLoopTask_, coreTask_ - nextTaskStart) : 0;
            hasContribution =
                ProcessFullBatch(gradInLocal, currentTaskCount, coreTaskStart_ + nextTaskStart, nextTaskCount) ||
                hasContribution;
            processedTask = nextTaskStart;
            currentTaskCount = nextTaskCount;
        }

        if (!hasContribution) {
            return;
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const bool needsAtomic = GetBlockNum() != 1;
        if (needsAtomic) {
            SetAtomicAdd<float>();
        }
        constexpr uint32_t kMaxCopyBlocks = 4095;
        for (uint32_t pointStart = 0; pointStart < npoints_; pointStart += kMaxCopyBlocks) {
            const uint32_t pointCount = min(kMaxCopyBlocks, npoints_ - pointStart);
            const DataCopyExtParams outputCopyParams{
                static_cast<uint16_t>(pointCount), static_cast<uint32_t>(channels_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(gradInGm_[static_cast<uint64_t>(pointStart) * channels_],
                gradInLocal[pointStart * channelAligned_], outputCopyParams);
        }
        if (needsAtomic) {
            SetAtomicNone();
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline bool ProcessFullBatch(
        LocalTensor<float> &gradInLocal, uint32_t taskCount, uint32_t nextTaskStart, uint32_t nextTaskCount) {
        LocalTensor<float> gradOutLocal = gradOutQueue_.DeQue<float>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQueue_.DeQue<int32_t>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (nextTaskCount != 0) {
            CopyIn(nextTaskStart, nextTaskCount);
        }

        bool hasContribution = false;
        for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
            const int32_t pointCount = PointCount(ptsIdxLocal, taskIdx);
            if (pointCount == 0) {
                continue;
            }
            const uint32_t gradOutOffset = taskIdx * channelAligned_;
            Muls(gradOutLocal[gradOutOffset], gradOutLocal[gradOutOffset], 1.0f / static_cast<float>(pointCount),
                channels_);
            const uint32_t ptsOffset = taskIdx * ptsRowAligned_ + 1;
            for (int32_t pointOffset = 0; pointOffset < pointCount; ++pointOffset) {
                const int32_t pointIdx = ptsIdxLocal.GetValue(ptsOffset + pointOffset);
                if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(npoints_)) {
                    continue;
                }
                const uint32_t gradInOffset = static_cast<uint32_t>(pointIdx) * channelAligned_;
                Add(gradInLocal[gradInOffset], gradOutLocal[gradOutOffset], gradInLocal[gradInOffset], channels_);
                hasContribution = true;
            }
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        gradOutQueue_.FreeTensor(gradOutLocal);
        ptsIdxQueue_.FreeTensor(ptsIdxLocal);
        return hasContribution;
    }
};

extern "C" __global__ __aicore__ void roiaware_avgpool3d_grad(
    GM_ADDR pts_idx_of_voxels, GM_ADDR grad_out, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling_data) {
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);

    TPipe pipe;
    if (TILING_KEY_IS(2)) {
        RoiawareAvgpool3dGradKernel<false> op;
        op.Init(&pipe, pts_idx_of_voxels, grad_out, grad_in, &tiling);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        RoiAwareAvgpool3dGrad<float> op;
        op.Init(&pipe, pts_idx_of_voxels, grad_out, grad_in, &tiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        RoiAwareAvgpool3dGrad<float> op;
        op.Init(&pipe, pts_idx_of_voxels, grad_out, grad_in, &tiling);
        op.ProcessSequential();
    }
}
