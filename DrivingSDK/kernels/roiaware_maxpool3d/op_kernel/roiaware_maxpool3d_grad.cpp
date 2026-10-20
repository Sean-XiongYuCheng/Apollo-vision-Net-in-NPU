#include "kernel_operator.h"
using namespace AscendC;
#define BUFFER_NUM 2
#define BUFFER_ALIGN_BYTE 32
#define COMPARE_ALGIN_BYTE 256

template <typename dataType, bool aicpu> class KernelRoiawareMaxpool3dGrad {
  public:
    __aicore__ inline KernelRoiawareMaxpool3dGrad() {}
    __aicore__ void Init(const GM_ADDR argmax, const GM_ADDR gradOut, GM_ADDR gradIn, TPipe *pipe,
        const RoiawareMaxpool3dGradTilingData *tiling_data)

    {
        this->pipe_ = pipe;
        this->blkIdx_ = GetBlockIdx();
        this->InitTiling(tiling_data);
        this->InitGlobalBuffer(argmax, gradOut, gradIn, tiling_data);
        this->InitUBBuffer();
    }

    __aicore__ void Process() {
        if (coreTask_ == 0) {
            return;
        }

        uint32_t globalIdx = goutCoreStartIdx_;
        uint32_t curLoopTaskCount = min(singleLoopTask_, coreTask_);
        CopyInNTask(globalIdx, curLoopTaskCount);

        for (uint32_t curLoopTaskIdx = singleLoopTask_; curLoopTaskIdx < coreTask_; curLoopTaskIdx += singleLoopTask_) {
            globalIdx += singleLoopTask_;
            uint32_t nextLoopTaskCount = min(singleLoopTask_, coreTask_ - curLoopTaskIdx);
            CopyInNTask(globalIdx, nextLoopTaskCount);
            ProcessNTask(curLoopTaskCount);
            curLoopTaskCount = nextLoopTaskCount;
        }
        ProcessNTask(curLoopTaskCount);
    }

  protected:
    TPipe *pipe_;
    uint32_t blkIdx_;
    uint32_t channels_, nPoints_, channelAligned_, alignedArgmaxByteLength_;
    uint32_t coreTask_, singleLoopTask_, singleLoopOutput_;
    uint32_t argmaxCoreStartIdx_, goutCoreStartIdx_;

    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_, goutQue_;
    TBuf<TPosition::VECCALC> tmpMaskBuf_;
    TBuf<TPosition::VECCALC> tmpSelectedBuf_;
    TBuf<TPosition::VECCALC> ginBuf_;

    GlobalTensor<int32_t> argmaxGm_;
    GlobalTensor<dataType> goutGm_, ginGm_;

    DataCopyExtParams goutCopyParams_;
    DataCopyExtParams argmaxCopyParams_;
    DataCopyPadExtParams<dataType> goutPadParams_{true, 0, 0, 0};
    DataCopyPadExtParams<int32_t> argmaxPadParams_{true, 0, 0, 0};

  private:
    __aicore__ inline void InitTiling(const RoiawareMaxpool3dGradTilingData *tiling_data) {
        this->channels_ = tiling_data->channels;
        this->nPoints_ = tiling_data->npoints;
        this->singleLoopTask_ = tiling_data->singleLoopTask;
        this->singleLoopOutput_ = tiling_data->singleLoopOutput;
        this->channelAligned_ = tiling_data->channelAligned;

        if (blkIdx_ >= tiling_data->firstSmallCoreIdx) {
            this->coreTask_ = tiling_data->coreTask - 1;
            argmaxCoreStartIdx_ = tiling_data->coreTask * blkIdx_ - (blkIdx_ - tiling_data->firstSmallCoreIdx);
            goutCoreStartIdx_ = argmaxCoreStartIdx_;
        } else {
            this->coreTask_ = tiling_data->coreTask;
            argmaxCoreStartIdx_ = tiling_data->coreTask * blkIdx_;
            goutCoreStartIdx_ = argmaxCoreStartIdx_;
        }
        this->alignedArgmaxByteLength_ =
            Ceil(singleLoopTask_ * channelAligned_ * sizeof(int32_t), COMPARE_ALGIN_BYTE) * COMPARE_ALGIN_BYTE;
    }

    __aicore__ inline void InitGlobalBuffer(const GM_ADDR argmax, const GM_ADDR gradOut, const GM_ADDR gradIn,
        const RoiawareMaxpool3dGradTilingData *tiling_data) {
        uint64_t argmaxLength = static_cast<uint64_t>(tiling_data->totalTask) * channels_ * sizeof(int32_t);
        uint64_t gradOutLength = static_cast<uint64_t>(tiling_data->totalTask) * channels_ * sizeof(dataType);
        uint64_t gradInLength = static_cast<uint64_t>(nPoints_) * channels_ * sizeof(dataType);

        this->argmaxGm_.SetGlobalBuffer((__gm__ int32_t *)argmax, argmaxLength);
        this->goutGm_.SetGlobalBuffer((__gm__ dataType *)gradOut, gradOutLength);
        this->ginGm_.SetGlobalBuffer((__gm__ dataType *)gradIn, gradInLength);
    }

    __aicore__ inline void InitUBBuffer() {
        this->pipe_->InitBuffer(argmaxQue_, BUFFER_NUM, alignedArgmaxByteLength_);

        this->pipe_->InitBuffer(goutQue_, BUFFER_NUM, singleLoopTask_ * channelAligned_ * sizeof(dataType));
        this->pipe_->InitBuffer(ginBuf_, singleLoopOutput_ * channelAligned_ * sizeof(dataType));

        uint32_t tmpMaskBufferByteLength = BUFFER_ALIGN_BYTE *
            Ceil(Ceil(channelAligned_ * singleLoopTask_, AscendCUtils::GetBitSize(sizeof(uint8_t))), BUFFER_ALIGN_BYTE);
        this->pipe_->InitBuffer(this->tmpMaskBuf_, tmpMaskBufferByteLength);
        this->pipe_->InitBuffer(this->tmpSelectedBuf_, singleLoopTask_ * channelAligned_ * sizeof(dataType));
    }

    __aicore__ inline void CopyInNTask(int64_t inputIdx, int32_t curLoopTaskCount) {
        LocalTensor<dataType> goutLocal = goutQue_.AllocTensor<dataType>();
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.AllocTensor<int32_t>();

        int64_t goutLocalOffset = inputIdx * channels_;
        goutCopyParams_ = {
            static_cast<uint16_t>(curLoopTaskCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
        DataCopyPad(goutLocal, goutGm_[goutLocalOffset], goutCopyParams_, goutPadParams_);

        int64_t argmaxLocalOffset = inputIdx * channels_;
        argmaxCopyParams_ = {
            static_cast<uint16_t>(curLoopTaskCount), static_cast<uint32_t>(channels_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(argmaxLocal, argmaxGm_[argmaxLocalOffset], argmaxCopyParams_, argmaxPadParams_);

        goutQue_.EnQue<dataType>(goutLocal);
        argmaxQue_.EnQue<int32_t>(argmaxLocal);
    }

    __aicore__ inline void ProcessNTask(const uint32_t &taskCount) {
        LocalTensor<dataType> goutLocal = goutQue_.DeQue<dataType>();
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.DeQue<int32_t>();

        for (int32_t outputIdx = 0; outputIdx < nPoints_; outputIdx += singleLoopOutput_) {
            uint32_t curOutputTask = min(singleLoopOutput_, nPoints_ - outputIdx);
            if (outputIdx + singleLoopOutput_ >= nPoints_) {
                Compute<true>(taskCount, curOutputTask, outputIdx, goutLocal, argmaxLocal);
            } else {
                Compute<false>(taskCount, curOutputTask, outputIdx, goutLocal, argmaxLocal);
            }
        }
    }

    __aicore__ inline void GinLocalReduceSum(const uint32_t &oi, const uint32_t &curLoopTaskCount,
        LocalTensor<dataType> &ginLocal, LocalTensor<dataType> &selectLocal) {
        uint32_t selectRepStride = (channelAligned_ * sizeof(dataType)) / (BUFFER_ALIGN_BYTE);
        uint64_t maxElementsCount = 256 / sizeof(dataType);
        uint32_t baseIdx = oi * channelAligned_;

        if (selectRepStride >= 64) {
            for (uint32_t taskIdx = 0; taskIdx < curLoopTaskCount; taskIdx++) {
                // Both input rows are zero-padded to channelAligned_. Using
                // the aligned length also makes the vector tail after the
                // first 512 elements participate in the accumulation.
                Add(ginLocal[oi * channelAligned_], selectLocal[taskIdx * channelAligned_],
                    ginLocal[oi * channelAligned_], channelAligned_);
            }
            return;
        }

        for (int32_t offsetIdx = 0; offsetIdx < channelAligned_; offsetIdx += maxElementsCount) {
            uint64_t mask =
                maxElementsCount < channelAligned_ - offsetIdx ? maxElementsCount : channelAligned_ - offsetIdx;
            Add(ginLocal[offsetIdx + baseIdx], selectLocal[offsetIdx], ginLocal[offsetIdx + baseIdx], mask,
                curLoopTaskCount, {1, 1, 1, 0, static_cast<uint8_t>(selectRepStride), 0});
        }
    }

    template <bool isTail>
    __aicore__ inline void Compute(const uint32_t &curLoopTaskCount, const uint32_t &curLoopOutputCount,
        const uint64_t &outputIdx, LocalTensor<dataType> &goutLocal, LocalTensor<int32_t> &argmaxLocal) {
        LocalTensor<dataType> ginLocal = ginBuf_.AllocTensor<dataType>();
        LocalTensor<uint8_t> maskLocal = tmpMaskBuf_.AllocTensor<uint8_t>();
        LocalTensor<dataType> selectLocal = tmpSelectedBuf_.AllocTensor<dataType>();

        int32_t eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId1);
        WaitFlag<HardEvent::MTE3_V>(eventId1);
        Duplicate<dataType>(ginLocal, 0, singleLoopOutput_ * channelAligned_);
        PipeBarrier<PIPE_ALL>();

        for (int32_t oi = 0; oi < curLoopOutputCount; oi++) {
            int32_t curOutIdx = outputIdx + oi;
            CompareScalar(maskLocal, argmaxLocal, static_cast<int32_t>(curOutIdx), CMPMODE::EQ,
                alignedArgmaxByteLength_ / sizeof(int32_t));
            PipeBarrier<PIPE_V>();
            Select(selectLocal, maskLocal, goutLocal, static_cast<dataType>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                curLoopTaskCount * channelAligned_);
            PipeBarrier<PIPE_V>();
            GinLocalReduceSum(oi, curLoopTaskCount, ginLocal, selectLocal);
        }

        int32_t eventId2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId2);
        WaitFlag<HardEvent::V_MTE3>(eventId2);

        if (isTail) {
            goutQue_.FreeTensor(goutLocal);
            argmaxQue_.FreeTensor(argmaxLocal);
        }

        SetAtomicAdd<dataType>();
        DataCopyPad(ginGm_[outputIdx * channels_], ginLocal,
            {static_cast<uint16_t>(curLoopOutputCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0});
        SetAtomicNone();
    }
};

// Channel-block scatter implementation used for the large, usual RoIAware
// shapes.  The legacy implementation above scans the complete task batch for
// every point index (O(tasks * channels * npoints)).  Here one core owns an
// 8-channel/32-byte block, keeps the complete [npoints, 8] destination tile
// in UB, and scatters each input element into that tile (O(tasks * channels)).
// The host only selects this key when the destination tile and one input batch
// fit in UB; all other shapes use the legacy kernel.
template <typename dataType> class KernelRoiawareMaxpool3dGradFast {
  public:
    __aicore__ inline KernelRoiawareMaxpool3dGradFast() {}

    __aicore__ inline void Init(const GM_ADDR argmax, const GM_ADDR gradOut, GM_ADDR gradIn, TPipe *pipe,
        const RoiawareMaxpool3dGradTilingData *tiling_data) {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        totalTask_ = tiling_data->totalTask;
        singleLoopTask_ = tiling_data->singleLoopTask == 0 ? 1 : tiling_data->singleLoopTask;
        channels_ = tiling_data->channels;
        nPoints_ = tiling_data->npoints;
        channelBlockNum_ = (channels_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;

        argmaxGm_.SetGlobalBuffer((__gm__ int32_t *)argmax, static_cast<uint64_t>(totalTask_) * channels_);
        goutGm_.SetGlobalBuffer((__gm__ dataType *)gradOut, static_cast<uint64_t>(totalTask_) * channels_);
        ginGm_.SetGlobalBuffer((__gm__ dataType *)gradIn, static_cast<uint64_t>(nPoints_) * channels_);

        // The host fast-path check guarantees these allocations fit.  Each
        // task row is padded to one 32-byte block for both input tensors.
        pipe_->InitBuffer(argmaxBuf_, singleLoopTask_ * CHANNEL_BLOCK * sizeof(int32_t));
        pipe_->InitBuffer(goutBuf_, singleLoopTask_ * CHANNEL_BLOCK * sizeof(dataType));
        pipe_->InitBuffer(ginBuf_, nPoints_ * CHANNEL_BLOCK * sizeof(dataType));
    }

    __aicore__ inline void Process() {
        ASSERT(blockNum_ != 0 && "block dim can not be zero!");
        for (uint32_t channelBlockIdx = blockIdx_; channelBlockIdx < channelBlockNum_; channelBlockIdx += blockNum_) {
            ProcessChannelBlock(channelBlockIdx);
        }
    }

  private:
    static constexpr uint32_t CHANNEL_BLOCK = 8;

    __aicore__ inline void ProcessChannelBlock(const uint32_t channelBlockIdx) {
        const uint32_t channelStart = channelBlockIdx * CHANNEL_BLOCK;
        const uint32_t channelCount = min(CHANNEL_BLOCK, channels_ - channelStart);

        LocalTensor<int32_t> argmaxLocal = argmaxBuf_.Get<int32_t>();
        LocalTensor<dataType> goutLocal = goutBuf_.Get<dataType>();
        LocalTensor<dataType> ginLocal = ginBuf_.Get<dataType>();

        // Start from zero because the public wrapper allocates grad_in as a
        // zero tensor and this kernel writes each output channel exactly once.
        Duplicate<dataType>(ginLocal, static_cast<dataType>(0), nPoints_ * CHANNEL_BLOCK);
        PipeBarrier<PIPE_ALL>();

        for (uint64_t taskStart = 0; taskStart < totalTask_; taskStart += singleLoopTask_) {
            const uint64_t remainingTask = static_cast<uint64_t>(totalTask_) - taskStart;
            const uint32_t taskCount =
                remainingTask < singleLoopTask_ ? static_cast<uint32_t>(remainingTask) : singleLoopTask_;
            const uint64_t inputOffset = taskStart * channels_ + channelStart;
            const uint32_t sourceStride = (channels_ - channelCount) * sizeof(int32_t);
            const uint32_t valueStride = (channels_ - channelCount) * sizeof(dataType);

            DataCopyExtParams argmaxCopyParams{static_cast<uint16_t>(taskCount),
                static_cast<uint32_t>(channelCount * sizeof(int32_t)), sourceStride, 0, 0};
            DataCopyExtParams goutCopyParams{static_cast<uint16_t>(taskCount),
                static_cast<uint32_t>(channelCount * sizeof(dataType)), valueStride, 0, 0};
            DataCopyPad(argmaxLocal, argmaxGm_[inputOffset], argmaxCopyParams, argmaxPadParams_);
            DataCopyPad(goutLocal, goutGm_[inputOffset], goutCopyParams, goutPadParams_);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
                const uint32_t localBase = taskIdx * CHANNEL_BLOCK;
                for (uint32_t lane = 0; lane < channelCount; ++lane) {
                    const int32_t pointIdx = argmaxLocal.GetValue(localBase + lane);
                    // Forward max-pooling uses -1 for an empty voxel.  The
                    // range check also keeps malformed user-provided argmax
                    // values from indexing outside the local destination.
                    if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(nPoints_)) {
                        continue;
                    }
                    const uint32_t outputOffset = static_cast<uint32_t>(pointIdx) * CHANNEL_BLOCK + lane;
                    ginLocal.SetValue(
                        outputOffset, ginLocal.GetValue(outputOffset) + goutLocal.GetValue(localBase + lane));
                }
            }
            PipeBarrier<PIPE_ALL>();
        }

        // A float output is initialized to zero by the C++ wrapper. Aligned
        // channel rows are disjoint 32-byte transfers and can be written
        // directly. Atomic add keeps the odd-channel case safe when adjacent
        // rows are not 32-byte aligned.
        const uint32_t outputStride = (channels_ - channelCount) * sizeof(dataType);
        DataCopyExtParams outputCopyParams{static_cast<uint16_t>(nPoints_),
            static_cast<uint32_t>(channelCount * sizeof(dataType)), 0, outputStride, 0};
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        if (channels_ % CHANNEL_BLOCK == 0) {
            DataCopyPad(ginGm_[channelStart], ginLocal, outputCopyParams);
        } else {
            SetAtomicAdd<dataType>();
            DataCopyPad(ginGm_[channelStart], ginLocal, outputCopyParams);
            SetAtomicNone();
        }
        int32_t copyDoneEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(copyDoneEventId);
        WaitFlag<HardEvent::MTE3_V>(copyDoneEventId);
    }

  private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t blockNum_;
    uint32_t totalTask_;
    uint32_t singleLoopTask_;
    uint32_t channels_;
    uint32_t nPoints_;
    uint32_t channelBlockNum_;

    TBuf<TPosition::VECCALC> argmaxBuf_;
    TBuf<TPosition::VECCALC> goutBuf_;
    TBuf<TPosition::VECCALC> ginBuf_;
    GlobalTensor<int32_t> argmaxGm_;
    GlobalTensor<dataType> goutGm_;
    GlobalTensor<dataType> ginGm_;
    DataCopyPadExtParams<int32_t> argmaxPadParams_{true, 0, 0, 0};
    DataCopyPadExtParams<dataType> goutPadParams_{true, 0, 0, 0};
};

// 2D fast path for shapes where channel blocks alone do not expose enough
// parallelism.  A work item owns one task range and one 8-channel block.  Task
// ranges for the same channel block overlap in output, so partial [npoints, 8]
// tiles are merged with atomic add at copy-out.
template <typename dataType, uint32_t CHANNEL_BLOCK> class KernelRoiawareMaxpool3dGradFast2D {
  public:
    __aicore__ inline KernelRoiawareMaxpool3dGradFast2D() {}

    __aicore__ inline void Init(const GM_ADDR argmax, const GM_ADDR gradOut, GM_ADDR gradIn, TPipe *pipe,
        const RoiawareMaxpool3dGradTilingData *tiling_data) {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        totalTask_ = tiling_data->totalTask;
        taskTileTask_ = tiling_data->coreTask == 0 ? 1 : tiling_data->coreTask;
        taskTileNum_ = tiling_data->firstSmallCoreIdx == 0 ? 1 : tiling_data->firstSmallCoreIdx;
        singleLoopTask_ = tiling_data->singleLoopTask == 0 ? 1 : tiling_data->singleLoopTask;
        channels_ = tiling_data->channels;
        nPoints_ = tiling_data->npoints;
        channelBlockNum_ = (channels_ + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;

        argmaxGm_.SetGlobalBuffer((__gm__ int32_t *)argmax, static_cast<uint64_t>(totalTask_) * channels_);
        goutGm_.SetGlobalBuffer((__gm__ dataType *)gradOut, static_cast<uint64_t>(totalTask_) * channels_);
        ginGm_.SetGlobalBuffer((__gm__ dataType *)gradIn, static_cast<uint64_t>(nPoints_) * channels_);

        pipe_->InitBuffer(argmaxBuf_, singleLoopTask_ * CHANNEL_BLOCK * sizeof(int32_t));
        pipe_->InitBuffer(goutBuf_, singleLoopTask_ * CHANNEL_BLOCK * sizeof(dataType));
        pipe_->InitBuffer(ginBuf_, nPoints_ * CHANNEL_BLOCK * sizeof(dataType));
    }

    __aicore__ inline void Process() {
        ASSERT(blockNum_ != 0 && "block dim can not be zero!");
        const uint32_t totalWorkItem = taskTileNum_ * channelBlockNum_;
        for (uint32_t workItemIdx = blockIdx_; workItemIdx < totalWorkItem; workItemIdx += blockNum_) {
            const uint32_t channelBlockIdx = workItemIdx % channelBlockNum_;
            const uint32_t taskTileIdx = workItemIdx / channelBlockNum_;
            const uint64_t taskStart = static_cast<uint64_t>(taskTileIdx) * taskTileTask_;
            if (taskStart >= totalTask_) {
                continue;
            }
            const uint64_t remainingTask = static_cast<uint64_t>(totalTask_) - taskStart;
            const uint32_t taskCount =
                remainingTask < taskTileTask_ ? static_cast<uint32_t>(remainingTask) : taskTileTask_;
            ProcessTaskChannelBlock(channelBlockIdx, taskStart, taskCount);
        }
    }

  private:
    __aicore__ inline void ProcessTaskChannelBlock(
        const uint32_t channelBlockIdx, const uint64_t taskStart, const uint32_t taskCount) {
        const uint32_t channelStart = channelBlockIdx * CHANNEL_BLOCK;
        const uint32_t channelCount = min(CHANNEL_BLOCK, channels_ - channelStart);

        LocalTensor<int32_t> argmaxLocal = argmaxBuf_.Get<int32_t>();
        LocalTensor<dataType> goutLocal = goutBuf_.Get<dataType>();
        LocalTensor<dataType> ginLocal = ginBuf_.Get<dataType>();

        Duplicate<dataType>(ginLocal, static_cast<dataType>(0), nPoints_ * CHANNEL_BLOCK);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t taskOffset = 0; taskOffset < taskCount; taskOffset += singleLoopTask_) {
            const uint32_t remainingTask = taskCount - taskOffset;
            const uint32_t curTaskCount = remainingTask < singleLoopTask_ ? remainingTask : singleLoopTask_;
            const uint64_t inputOffset = (taskStart + taskOffset) * channels_ + channelStart;
            const uint32_t sourceStride = (channels_ - channelCount) * sizeof(int32_t);
            const uint32_t valueStride = (channels_ - channelCount) * sizeof(dataType);
            const uint32_t argmaxBlockBytes =
                ((channelCount * sizeof(int32_t) + BUFFER_ALIGN_BYTE - 1) / BUFFER_ALIGN_BYTE) * BUFFER_ALIGN_BYTE;
            const uint32_t valueBlockBytes =
                ((channelCount * sizeof(dataType) + BUFFER_ALIGN_BYTE - 1) / BUFFER_ALIGN_BYTE) * BUFFER_ALIGN_BYTE;
            const uint32_t argmaxLocalStride = (CHANNEL_BLOCK * sizeof(int32_t) - argmaxBlockBytes) / BUFFER_ALIGN_BYTE;
            const uint32_t valueLocalStride = (CHANNEL_BLOCK * sizeof(dataType) - valueBlockBytes) / BUFFER_ALIGN_BYTE;

            DataCopyExtParams argmaxCopyParams{static_cast<uint16_t>(curTaskCount),
                static_cast<uint32_t>(channelCount * sizeof(int32_t)), sourceStride, argmaxLocalStride, 0};
            DataCopyExtParams goutCopyParams{static_cast<uint16_t>(curTaskCount),
                static_cast<uint32_t>(channelCount * sizeof(dataType)), valueStride, valueLocalStride, 0};
            DataCopyPad(argmaxLocal, argmaxGm_[inputOffset], argmaxCopyParams, argmaxPadParams_);
            DataCopyPad(goutLocal, goutGm_[inputOffset], goutCopyParams, goutPadParams_);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t taskIdx = 0; taskIdx < curTaskCount; ++taskIdx) {
                const uint32_t localBase = taskIdx * CHANNEL_BLOCK;
                for (uint32_t lane = 0; lane < channelCount; ++lane) {
                    const int32_t pointIdx = argmaxLocal.GetValue(localBase + lane);
                    if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(nPoints_)) {
                        continue;
                    }
                    const uint32_t outputOffset = static_cast<uint32_t>(pointIdx) * CHANNEL_BLOCK + lane;
                    ginLocal.SetValue(
                        outputOffset, ginLocal.GetValue(outputOffset) + goutLocal.GetValue(localBase + lane));
                }
            }
            PipeBarrier<PIPE_ALL>();
        }

        const uint32_t outputStride = (channels_ - channelCount) * sizeof(dataType);
        // UB2GM srcStride is measured in 32-byte blocks and starts after the
        // aligned transfer width. Skip the remaining part of the padded UB
        // row for a partial channel block.
        const uint32_t sourceBlockBytes =
            ((channelCount * sizeof(dataType) + BUFFER_ALIGN_BYTE - 1) / BUFFER_ALIGN_BYTE) * BUFFER_ALIGN_BYTE;
        const uint32_t sourceStride = (CHANNEL_BLOCK * sizeof(dataType) - sourceBlockBytes) / BUFFER_ALIGN_BYTE;
        DataCopyExtParams outputCopyParams{static_cast<uint16_t>(nPoints_),
            static_cast<uint32_t>(channelCount * sizeof(dataType)), sourceStride, outputStride, 0};
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        SetAtomicAdd<dataType>();
        DataCopyPad(ginGm_[channelStart], ginLocal, outputCopyParams);
        SetAtomicNone();
        int32_t copyDoneEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(copyDoneEventId);
        WaitFlag<HardEvent::MTE3_V>(copyDoneEventId);
    }

  private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t blockNum_;
    uint32_t totalTask_;
    uint32_t taskTileTask_;
    uint32_t taskTileNum_;
    uint32_t singleLoopTask_;
    uint32_t channels_;
    uint32_t nPoints_;
    uint32_t channelBlockNum_;

    TBuf<TPosition::VECCALC> argmaxBuf_;
    TBuf<TPosition::VECCALC> goutBuf_;
    TBuf<TPosition::VECCALC> ginBuf_;
    GlobalTensor<int32_t> argmaxGm_;
    GlobalTensor<dataType> goutGm_;
    GlobalTensor<dataType> ginGm_;
    DataCopyPadExtParams<int32_t> argmaxPadParams_{true, 0, 0, 0};
    DataCopyPadExtParams<dataType> goutPadParams_{true, 0, 0, 0};
};

// Vector member-scan path. A forward-consistent argmax must be one of the
// saved members of its voxel, so sparse voxels can scan that short member list
// instead of all npoints. CompareScalar and Select still process a complete
// channel row at a time; selected rows are batched in UB before atomic copy-out.
template <typename dataType> class KernelRoiawareMaxpool3dGradMember {
  public:
    __aicore__ inline KernelRoiawareMaxpool3dGradMember() {}

    __aicore__ inline void Init(const GM_ADDR ptsIdx, const GM_ADDR argmax, const GM_ADDR gradOut, GM_ADDR gradIn,
        TPipe *pipe, const RoiawareMaxpool3dGradTilingData *tiling) {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        channels_ = tiling->channels;
        channelAligned_ = tiling->channelAligned;
        nPoints_ = tiling->npoints;
        totalTask_ = tiling->totalTask;
        singleLoopTask_ = tiling->singleLoopTask == 0 ? 1 : tiling->singleLoopTask;
        memberBatch_ = tiling->singleLoopOutput == 0 ? 1 : tiling->singleLoopOutput;
        ptsRowElements_ = tiling->ptsRowElements;
        ptsRowAligned_ = tiling->ptsRowAligned;
        compareElements_ =
            Ceil(channelAligned_ * sizeof(int32_t), COMPARE_ALGIN_BYTE) * (COMPARE_ALGIN_BYTE / sizeof(int32_t));

        coreTask_ = tiling->coreTask;
        if (blockIdx_ < tiling->firstSmallCoreIdx) {
            coreTaskStart_ = blockIdx_ * coreTask_;
        } else {
            coreTaskStart_ =
                tiling->firstSmallCoreIdx * coreTask_ + (blockIdx_ - tiling->firstSmallCoreIdx) * (coreTask_ - 1);
            --coreTask_;
        }

        ptsIdxGm_.SetGlobalBuffer((__gm__ int32_t *)ptsIdx, static_cast<uint64_t>(totalTask_) * ptsRowElements_);
        argmaxGm_.SetGlobalBuffer((__gm__ int32_t *)argmax, static_cast<uint64_t>(totalTask_) * channels_);
        goutGm_.SetGlobalBuffer((__gm__ dataType *)gradOut, static_cast<uint64_t>(totalTask_) * channels_);
        ginGm_.SetGlobalBuffer((__gm__ dataType *)gradIn, static_cast<uint64_t>(nPoints_) * channels_);

        pipe_->InitBuffer(argmaxQue_, BUFFER_NUM,
            (singleLoopTask_ * channelAligned_ + compareElements_ - channelAligned_) * sizeof(int32_t));
        pipe_->InitBuffer(goutQue_, BUFFER_NUM, singleLoopTask_ * channelAligned_ * sizeof(dataType));
        pipe_->InitBuffer(ptsIdxQue_, BUFFER_NUM, singleLoopTask_ * ptsRowAligned_ * sizeof(int32_t));
        const uint32_t maskBytes = BUFFER_ALIGN_BYTE *
            Ceil(Ceil(compareElements_, AscendCUtils::GetBitSize(sizeof(uint8_t))), BUFFER_ALIGN_BYTE);
        pipe_->InitBuffer(maskBuf_, maskBytes);
        pipe_->InitBuffer(selectedBuf_, memberBatch_ * channelAligned_ * sizeof(dataType));
    }

    __aicore__ inline void Process() {
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
            ProcessBatch(currentTaskCount, coreTaskStart_ + nextTaskStart, nextTaskCount);
            processedTask = nextTaskStart;
            currentTaskCount = nextTaskCount;
        }
    }

  private:
    __aicore__ inline void CopyIn(uint32_t taskStart, uint32_t taskCount) {
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.AllocTensor<int32_t>();
        LocalTensor<dataType> goutLocal = goutQue_.AllocTensor<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.AllocTensor<int32_t>();

        const DataCopyExtParams argmaxParams{
            static_cast<uint16_t>(taskCount), static_cast<uint32_t>(channels_ * sizeof(int32_t)), 0, 0, 0};
        const DataCopyExtParams goutParams{
            static_cast<uint16_t>(taskCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
        const DataCopyExtParams ptsParams{
            static_cast<uint16_t>(taskCount), static_cast<uint32_t>(ptsRowElements_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(argmaxLocal, argmaxGm_[static_cast<uint64_t>(taskStart) * channels_], argmaxParams, argmaxPad_);
        DataCopyPad(goutLocal, goutGm_[static_cast<uint64_t>(taskStart) * channels_], goutParams, goutPad_);
        DataCopyPad(ptsIdxLocal, ptsIdxGm_[static_cast<uint64_t>(taskStart) * ptsRowElements_], ptsParams, ptsPad_);

        argmaxQue_.EnQue<int32_t>(argmaxLocal);
        goutQue_.EnQue<dataType>(goutLocal);
        ptsIdxQue_.EnQue<int32_t>(ptsIdxLocal);
    }

    __aicore__ inline void ProcessBatch(uint32_t taskCount, uint32_t nextTaskStart, uint32_t nextTaskCount) {
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.DeQue<int32_t>();
        LocalTensor<dataType> goutLocal = goutQue_.DeQue<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.DeQue<int32_t>();
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        LocalTensor<dataType> selectedLocal = selectedBuf_.Get<dataType>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (nextTaskCount != 0) {
            CopyIn(nextTaskStart, nextTaskCount);
        }

        SetAtomicAdd<dataType>();
        for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
            int32_t memberCount = ptsIdxLocal.GetValue(taskIdx * ptsRowAligned_);
            memberCount = max(memberCount, 0);
            memberCount = min(memberCount, static_cast<int32_t>(ptsRowElements_ - 1));
            const uint32_t argmaxOffset = taskIdx * channelAligned_;
            const uint32_t ptsOffset = taskIdx * ptsRowAligned_ + 1;

            for (int32_t memberStart = 0; memberStart < memberCount; memberStart += memberBatch_) {
                const uint32_t currentMemberCount = min(memberBatch_, static_cast<uint32_t>(memberCount - memberStart));
                for (uint32_t memberOffset = 0; memberOffset < currentMemberCount; ++memberOffset) {
                    const int32_t pointIdx = ptsIdxLocal.GetValue(ptsOffset + memberStart + memberOffset);
                    if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(nPoints_)) {
                        continue;
                    }
                    CompareScalar(maskLocal, argmaxLocal[argmaxOffset], pointIdx, CMPMODE::EQ, compareElements_);
                    PipeBarrier<PIPE_V>();
                    Select(selectedLocal[memberOffset * channelAligned_], maskLocal, goutLocal[argmaxOffset],
                        static_cast<dataType>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, channelAligned_);
                    PipeBarrier<PIPE_V>();
                }

                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                for (uint32_t memberOffset = 0; memberOffset < currentMemberCount; ++memberOffset) {
                    const int32_t pointIdx = ptsIdxLocal.GetValue(ptsOffset + memberStart + memberOffset);
                    if (pointIdx < 0 || pointIdx >= static_cast<int32_t>(nPoints_)) {
                        continue;
                    }
                    const DataCopyExtParams outputParams{
                        1, static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
                    DataCopyPad(ginGm_[static_cast<uint64_t>(pointIdx) * channels_],
                        selectedLocal[memberOffset * channelAligned_], outputParams);
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        SetAtomicNone();

        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        argmaxQue_.FreeTensor(argmaxLocal);
        goutQue_.FreeTensor(goutLocal);
        ptsIdxQue_.FreeTensor(ptsIdxLocal);
    }

  private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t channels_, channelAligned_, nPoints_, totalTask_;
    uint32_t coreTask_, coreTaskStart_, singleLoopTask_, memberBatch_;
    uint32_t ptsRowElements_, ptsRowAligned_, compareElements_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_, goutQue_, ptsIdxQue_;
    TBuf<TPosition::VECCALC> maskBuf_, selectedBuf_;
    GlobalTensor<int32_t> ptsIdxGm_, argmaxGm_;
    GlobalTensor<dataType> goutGm_, ginGm_;
    DataCopyPadExtParams<int32_t> argmaxPad_{true, 0, 0, 0};
    DataCopyPadExtParams<int32_t> ptsPad_{true, 0, 0, 0};
    DataCopyPadExtParams<dataType> goutPad_{true, 0, 0, 0};
};

extern "C" __global__ __aicore__ void roiaware_maxpool3d_grad(
    GM_ADDR pts_idx_of_voxels, GM_ADDR argmax, GM_ADDR grad_out, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)pts_idx_of_voxels;
    GET_TILING_DATA(tiling_data, tiling);
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelRoiawareMaxpool3dGradFast<float> op;
        op.Init(argmax, grad_out, grad_in, &pipe, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelRoiawareMaxpool3dGradFast2D<float, 8> op;
        op.Init(argmax, grad_out, grad_in, &pipe, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelRoiawareMaxpool3dGradFast2D<float, 32> op;
        op.Init(argmax, grad_out, grad_in, &pipe, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelRoiawareMaxpool3dGradMember<float> op;
        op.Init(pts_idx_of_voxels, argmax, grad_out, grad_in, &pipe, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelRoiawareMaxpool3dGrad<float, false> op;
        op.Init(argmax, grad_out, grad_in, &pipe, &tiling_data);
        op.Process();
    }
}
