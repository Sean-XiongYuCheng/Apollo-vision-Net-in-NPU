// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include <kernel_common.h>

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FREE_NUM = 1024;
constexpr int32_t THREAD_NUM = 1024;

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtProcess(
    int32_t loops, int32_t maxPoints, int32_t featNum, int32_t ptsStride, int32_t blockVoxIdx,
    __gm__ int32_t* uniArgsortIdxGm_, __gm__ int32_t* argsortVoxIdxGm, __gm__ float* ptsGm,
    __gm__ float* voxelsGm, __gm__ int32_t* uniIdxGm, __gm__ int32_t* uniLenGm, __gm__ int32_t* uniVoxGm,
    __gm__ int32_t* numPointsPerVoxelGm, __gm__ int32_t* sortedUniVoxGm)
{
    for (int32_t i = Simt::GetThreadIdx(); i < loops; i += Simt::GetThreadNum()) {
        int32_t idx = uniArgsortIdxGm_[i];
        int32_t uniIdx = uniIdxGm[idx];
        int32_t uniLen = uniLenGm[idx];
        int32_t uniVox = uniVoxGm[idx];

        uniLen = uniLen > maxPoints ? maxPoints : uniLen;
        int32_t curVoxIdx = blockVoxIdx + i;

        __gm__ int32_t* argsortVoxIdxPtr = argsortVoxIdxGm + uniIdx;
        __gm__ float* voxelsPtr = voxelsGm + curVoxIdx * ptsStride;

        for (int32_t j = 0; j < uniLen; j++) {
            int32_t n = argsortVoxIdxPtr[j] * featNum;
            
            for (int32_t k = 0; k < featNum; k++) {
                voxelsPtr[j * featNum + k] = ptsGm[n + k];
            }
        }

        numPointsPerVoxelGm[curVoxIdx] = uniLen;
        sortedUniVoxGm[curVoxIdx] = uniVox;
    }
}

class HardVoxelizeDiffKernel {
public:
    __aicore__ inline HardVoxelizeDiffKernel() = delete;
    __aicore__ inline ~HardVoxelizeDiffKernel() = default;
    __aicore__ inline HardVoxelizeDiffKernel(GM_ADDR uniIdxs, GM_ADDR uniLens, const HardVoxelizeTilingData& tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedDiffBlkNum), avgTasks_(tiling.avgDiffTasks),
          tailTasks_(tiling.tailDiffTasks), totalTasks_(tiling.totalDiffTasks), avgPts_(tiling.avgPts),
          tailPts_(tiling.tailPts), totalPts_(tiling.totalPts), numPts_(tiling.numPts)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curPtsIdx_ = curTaskIdx_ * avgPts_;

        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        adjOffset_ = avgPts_;

        cpParam_.blockLen = static_cast<uint16_t>(avgPts_ / B32_DATA_NUM_PER_BLOCK);
        cpTailParam_.blockLen = static_cast<uint32_t>(tailPts_ * B32_BYTE_SIZE);
        cpExtParam_.blockLen = static_cast<uint32_t>(tailPts_ * B32_BYTE_SIZE);

        uniIdxsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniIdxs));
        uniLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniLens));

        pipe_.InitBuffer(inQue_, BUFFER_NUM, avgPts_ * 2 * B32_BYTE_SIZE);
        pipe_.InitBuffer(outQue_, BUFFER_NUM, avgPts_ * B32_BYTE_SIZE);
    }

    __aicore__ inline void Process();

    __aicore__ inline void Done();

private:
    TPipe pipe_;
    GlobalTensor<int32_t> uniIdxsGm_, uniLensGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curPtsIdx_, curOutputIdx_, startOutputIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgPts_, tailPts_, totalPts_, numPts_; // here, avgPts_must be multiple of 64
    int32_t adjOffset_;
    DataCopyParams cpParam_;
    DataCopyExtParams cpExtParam_, cpTailParam_;
    BinaryRepeatParams binRptParam_ {1, 1, 1, 8, 8, 8};
    uint8_t rptTimes_;

private:
    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_tail>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_tail>
    __aicore__ inline void CopyOut();
};

__aicore__ inline void HardVoxelizeDiffKernel::Process()
{
    if (blkIdx_ >= usedBlkNum_) {
        return;
    }

    for (int32_t i = 0; i < coreTasks_ - 1; ++i) {
        DoProcess<false>();
        ++curTaskIdx_;
        curPtsIdx_ += avgPts_;
    }

    if (IsLastTask()) {
        DoProcess<true>();
    } else {
        DoProcess<false>();
    }
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::DoProcess()
{
    CopyIn<is_tail>();
    Compute<is_tail>();
    CopyOut<is_tail>();
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::CopyIn()
{
    LocalTensor<int32_t> inT = inQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> idxT0 = inT[0];
    LocalTensor<int32_t> idxT1 = inT[adjOffset_];
    if (is_tail) {
        DataCopyPad(idxT0, uniIdxsGm_[curPtsIdx_], cpTailParam_, {0, 0, 0, 0});
        DataCopyPad(idxT1, uniIdxsGm_[curPtsIdx_ + 1], cpTailParam_, {0, 0, 0, 0});
    } else {
        DataCopy(idxT0, uniIdxsGm_[curPtsIdx_], cpParam_);
        DataCopy(idxT1, uniIdxsGm_[curPtsIdx_ + 1], cpParam_);
    }
    inQue_.EnQue(inT);
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::Compute()
{
    LocalTensor<int32_t> idxT = inQue_.DeQue<int32_t>();
    LocalTensor<int32_t> idxT0 = idxT[0];
    LocalTensor<int32_t> idxT1 = idxT[adjOffset_];
    LocalTensor<int32_t> outT = outQue_.AllocTensor<int32_t>();
    if (is_tail) {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        idxT1.SetValue(tailPts_ - 1, numPts_);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
    }
    Sub<int32_t, false>(outT, idxT1, idxT0, MASK_PLACEHOLDER, rptTimes_, binRptParam_);
    outQue_.EnQue(outT);
    inQue_.FreeTensor(idxT);
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::CopyOut()
{
    LocalTensor<int32_t> outT = outQue_.DeQue<int32_t>();
    if (is_tail) {
        DataCopyPad(uniLensGm_[curPtsIdx_], outT, cpExtParam_);
    } else {
        DataCopy(uniLensGm_[curPtsIdx_], outT, cpParam_);
    }
    outQue_.FreeTensor(outT);
}

__aicore__ inline void HardVoxelizeDiffKernel::Done()
{
    pipe_.Destroy();
    SyncAll();
}

template<bool is_aligned>
class HardVoxelizeCopyKernel {
public:
    __aicore__ inline HardVoxelizeCopyKernel() = delete;
    __aicore__ inline ~HardVoxelizeCopyKernel() = default;
    __aicore__ inline HardVoxelizeCopyKernel(GM_ADDR points, GM_ADDR uniVoxels, GM_ADDR argsortVoxelIdxs,
        GM_ADDR uniArgsortIdxs, GM_ADDR uniIdxs, GM_ADDR uniLens, GM_ADDR voxels, GM_ADDR numPointsPerVoxel,
        GM_ADDR sortedUniVoxels, const HardVoxelizeTilingData& tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedCopyBlkNum), avgTasks_(tiling.avgCopyTasks),
          tailTasks_(tiling.tailCopyTasks), totalTasks_(tiling.totalCopyTasks), avgVoxs_(tiling.avgVoxs),
          tailVoxs_(tiling.tailVoxs), totalVoxs_(tiling.totalVoxs), featNum_(tiling.featNum),
          maxPoints_(tiling.maxPoints)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curVoxIdx_ = curTaskIdx_ * avgVoxs_;
        ptsStride_ = maxPoints_ * featNum_;

        // init global memory
        ptsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(points));
        uniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniVoxels));
        argsortVoxIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(argsortVoxelIdxs));
        uniArgsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniArgsortIdxs));
        uniIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniIdxs));
        uniLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniLens));
        voxelsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(voxels));
        numPointsPerVoxelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numPointsPerVoxel));
        sortedUniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedUniVoxels));
    }

    __aicore__ inline void Process();

private:
    TPipe pipe_;
    GlobalTensor<float> ptsGm_;
    GlobalTensor<int32_t> uniVoxGm_, argsortVoxIdxGm_, uniArgsortIdxGm_, uniIdxGm_, uniLenGm_;
    GlobalTensor<int32_t> numPointsPerVoxelGm_, sortedUniVoxGm_;
    GlobalTensor<float> voxelsGm_;

    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curVoxIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgVoxs_, tailVoxs_, totalVoxs_;
    int32_t featNum_;
    int32_t maxPoints_, ptsStride_;

private:
    __aicore__ inline bool IsLastTask() const
    {
        return blkIdx_ == usedBlkNum_ - 1;
    }
};

template<bool is_aligned>
__aicore__ inline void HardVoxelizeCopyKernel<is_aligned>::Process()
{
    auto loops = IsLastTask() ? tailVoxs_ + (coreTasks_ - 1) * avgVoxs_ : coreTasks_ * avgVoxs_;
    Simt::VF_CALL<SimtProcess>(
        Simt::Dim3(THREAD_NUM),
        loops, maxPoints_, featNum_, ptsStride_, curVoxIdx_,
        (__gm__ int32_t*)uniArgsortIdxGm_[curVoxIdx_].GetPhyAddr(),
        (__gm__ int32_t*)argsortVoxIdxGm_.GetPhyAddr(),
        (__gm__ float*)ptsGm_.GetPhyAddr(),
        (__gm__ float*)voxelsGm_.GetPhyAddr(),
        (__gm__ int32_t*)uniIdxGm_.GetPhyAddr(),
        (__gm__ int32_t*)uniLenGm_.GetPhyAddr(),
        (__gm__ int32_t*)uniVoxGm_.GetPhyAddr(),
        (__gm__ int32_t*)numPointsPerVoxelGm_.GetPhyAddr(),
        (__gm__ int32_t*)sortedUniVoxGm_.GetPhyAddr()
    );
}

extern "C" __global__ __aicore__ void hard_voxelize(GM_ADDR points, GM_ADDR uniVoxels, GM_ADDR argsortVoxelIdxs,
    GM_ADDR uniArgsortIdxs, GM_ADDR uniIdxs, GM_ADDR voxels, GM_ADDR numPointsPerVoxel, GM_ADDR sortedUniVoxels,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    // phase 1: calculate the length of voxels, i.e. num_per_voxel
    HardVoxelizeDiffKernel diffOp(uniIdxs, workspace, tilingData);
    diffOp.Process();
    diffOp.Done();
    // phase 2: group the points by the voxel index, sort by the point order.
    if (TILING_KEY_IS(0)) {
        HardVoxelizeCopyKernel<false> copyOp(points, uniVoxels, argsortVoxelIdxs, uniArgsortIdxs, uniIdxs, workspace,
            voxels, numPointsPerVoxel, sortedUniVoxels, tilingData);
        copyOp.Process();
    } else if (TILING_KEY_IS(1)) {
        HardVoxelizeCopyKernel<true> copyOp(points, uniVoxels, argsortVoxelIdxs, uniArgsortIdxs, uniIdxs, workspace,
            voxels, numPointsPerVoxel, sortedUniVoxels, tilingData);
        copyOp.Process();
    }
}
