/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;
using namespace MicroAPI;

class KernelSelectIdxWithMask {
  public:
    __aicore__ inline KernelSelectIdxWithMask() {}
    __aicore__ inline void Init(GM_ADDR poly_line, GM_ADDR min_idx, GM_ADDR pt, GM_ADDR back_idx, GM_ADDR out_min_idx,
        SelectIdxWithMaskTilingData *tiling) {
        batchSize = tiling->batchSize;
        numPoint = tiling->numPoint;
        numIdx = tiling->numIdx;
        compBatchNum = tiling->compBatchNum;
        numTaskPerCore = tiling->numTaskPerCore;
        numTaskTail = tiling->numTaskTail;

        uint32_t blockSize = 32;
        uint32_t dataSize = blockSize / sizeof(float);

        numPointAligned = (numPoint + dataSize - 1) / dataSize * dataSize;
        numIdxAligned = (numIdx + dataSize - 1) / dataSize * dataSize;
        numIdxRound64 = (numIdx + 63) / 64 * 64;

        uint32_t coreId = GetBlockIdx();
        if (coreId < numTaskTail) {
            numTaskPerCore = numTaskPerCore + 1;
            startTaskId = numTaskPerCore * coreId;
        } else {
            startTaskId = numTaskPerCore * coreId + numTaskTail;
        }
        compBatchNum = min(numTaskPerCore, compBatchNum);

        // arch35: 所有 idx 缓冲区按 numIdxRound64 对齐 (MicroAPI 每次读 64 元素)
        ubLengthBatchPointSize = compBatchNum * numPointAligned;
        ubLengthBatchIdxSize = compBatchNum * numIdxRound64;

        polyLineGm.SetGlobalBuffer((__gm__ float *)poly_line, batchSize * numPoint * 2);
        minIdxGm.SetGlobalBuffer((__gm__ int32_t *)min_idx, batchSize * numIdx);
        ptGm.SetGlobalBuffer((__gm__ float *)pt, batchSize * numIdx * 2);
        backIdxGm.SetGlobalBuffer((__gm__ int32_t *)back_idx, batchSize * numIdx);
        outMinIdxGm.SetGlobalBuffer((__gm__ int32_t *)out_min_idx, batchSize * numIdx);

        InitBuffer();
        InitLocalTensor();
    }

    __aicore__ inline void Process() {
        // 预计算 batch 偏移: batchIdxOffset[batchIdx] = batchIdx * numPoint
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(batchIdxOffset, (int32_t)((batchIdx - 1) * numPoint), batchIdx * numIdx);
        }

        for (uint32_t loopTimes = 0; loopTimes < numTaskPerCore; loopTimes += compBatchNum) {
            uint32_t batchIdx = loopTimes + startTaskId;
            uint32_t compBatchNumActual =
                loopTimes + compBatchNum <= numTaskPerCore ? compBatchNum : numTaskPerCore - loopTimes;

            uint32_t compBatchPointSize = compBatchNumActual * numPoint;
            uint32_t compBatchIdxSize = compBatchNumActual * numIdx;

            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

            DataCopyPad(polyLineLocal, polyLineGm[batchIdx * numPoint * 2],
                {1, static_cast<uint32_t>(compBatchPointSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(minIdxLocal, minIdxGm[batchIdx * numIdx],
                {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(ptLocal, ptGm[batchIdx * numIdx * 2],
                {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(backidxLocal, backIdxGm[batchIdx * numIdx],
                {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            BatchIdxSelect(frontPoint, minIdxLocal, polyLineLocal, ubLengthBatchIdxSize, compBatchIdxSize);
            BatchIdxSelect(backPoint, backidxLocal, polyLineLocal, ubLengthBatchIdxSize, compBatchIdxSize);

            {
                __local_mem__ float *ptInterleavedPtr = (__local_mem__ float *)ptLocal.GetPhyAddr();
                __local_mem__ float *frontPtr = (__local_mem__ float *)frontPoint.GetPhyAddr();
                __local_mem__ float *backPtr = (__local_mem__ float *)backPoint.GetPhyAddr();
                __local_mem__ int32_t *outPtr = (__local_mem__ int32_t *)outMinIdxLocal.GetPhyAddr();
                __local_mem__ int32_t *minIdxPtr = (__local_mem__ int32_t *)minIdxLocal.GetPhyAddr();
                uint16_t rpt1 = DivCeil(compBatchIdxSize, (uint16_t)64);

                PipeBarrier<PIPE_ALL>();
                __VEC_SCOPE__ {
                    MicroAPI::RegTensor<float> ptSrcReg1, ptSrcReg2, ptX, ptY;
                    MicroAPI::RegTensor<float> frontX, frontY, backX, backY;
                    MicroAPI::RegTensor<float> subPtX, subPtY, subBackX, subBackY, dotVal, tmpReg;
                    MicroAPI::RegTensor<int32_t> minIdxReg, subOneReg, outReg, negOneReg;
                    MicroAPI::RegTensor<float> minIdxFloatReg;
                    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MaskPattern::ALL>();
                    MicroAPI::MaskReg maskInt = MicroAPI::CreateMask<int32_t, MaskPattern::ALL>();
                    MicroAPI::MaskReg cmpGT0, dotGT0, andMask, eqMask, orMask;

                    MicroAPI::Duplicate(negOneReg, (int32_t)-1, maskInt);

                    for (uint16_t i = 0; i < rpt1; i++) {
                        uint32_t off = i * 64;
                        uint32_t off128 = i * 128;
                        uint32_t offY = ubLengthBatchIdxSize + off;

                        // DeInterleave: 从交错的 pt [x0,y0,x1,y1,...] 分离 x/y (替代 GatherMask)
                        MicroAPI::DataCopy(ptSrcReg1, ptInterleavedPtr + off128);
                        MicroAPI::DataCopy(ptSrcReg2, ptInterleavedPtr + off128 + 64);
                        MicroAPI::DeInterleave(ptX, ptY, ptSrcReg1, ptSrcReg2);

                        // 加载 front/back (已分离的 x/y 布局)
                        MicroAPI::DataCopy(frontX, frontPtr + off);
                        MicroAPI::DataCopy(frontY, frontPtr + offY);
                        MicroAPI::DataCopy(backX, backPtr + off);
                        MicroAPI::DataCopy(backY, backPtr + offY);

                        // pt = pt - front
                        MicroAPI::Sub(subPtX, ptX, frontX, mask);
                        MicroAPI::Sub(subPtY, ptY, frontY, mask);

                        // back = back - front
                        MicroAPI::Sub(subBackX, backX, frontX, mask);
                        MicroAPI::Sub(subBackY, backY, frontY, mask);

                        // dot = pt_x * back_x + pt_y * back_y
                        MicroAPI::Mul(dotVal, subPtX, subBackX, mask);
                        MicroAPI::Mul(tmpReg, subPtY, subBackY, mask);
                        MicroAPI::Add(dotVal, dotVal, tmpReg, mask);

                        MicroAPI::DataCopy(minIdxReg, minIdxPtr + off);
                        static constexpr CastTrait castTrait = {
                            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
                        MicroAPI::Cast<float, int32_t, castTrait>(minIdxFloatReg, minIdxReg, mask);
                        MicroAPI::Compares<float, CMPMODE::GT>(cmpGT0, minIdxFloatReg, 0.0f, mask);
                        MicroAPI::Compares<float, CMPMODE::GT>(dotGT0, dotVal, 0.0f, mask);
                        MicroAPI::And(andMask, cmpGT0, dotGT0, mask);
                        MicroAPI::Compares<int32_t, CMPMODE::EQ>(eqMask, minIdxReg, (int32_t)(numPoint - 1), maskInt);
                        MicroAPI::Or(orMask, andMask, eqMask, mask);
                        MicroAPI::Add(subOneReg, minIdxReg, negOneReg, maskInt);
                        MicroAPI::Select(outReg, subOneReg, minIdxReg, orMask);
                        MicroAPI::DataCopy(outPtr + off, outReg, maskInt);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(outMinIdxGm[batchIdx * numIdx], outMinIdxLocal,
                {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0});
        }
    }

  private:
    __aicore__ inline void InitBuffer() {
        pipe.InitBuffer(polyLineBuf, ubLengthBatchPointSize * 2 * sizeof(float));
        pipe.InitBuffer(minIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(ptBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(backIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(selectIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(batchIdxOffsetBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(frontPointBuf, 2 * ubLengthBatchIdxSize * sizeof(float));
        pipe.InitBuffer(backPointBuf, 2 * ubLengthBatchIdxSize * sizeof(float));
        pipe.InitBuffer(outMinIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
    }

    __aicore__ inline void InitLocalTensor() {
        polyLineLocal = polyLineBuf.Get<float>();
        minIdxLocal = minIdxBuf.Get<int32_t>();
        ptLocal = ptBuf.Get<float>();
        backidxLocal = backIdxBuf.Get<int32_t>();
        selectIdx = selectIdxBuf.Get<int32_t>();
        batchIdxOffset = batchIdxOffsetBuf.Get<int32_t>();
        frontPoint = frontPointBuf.Get<float>();
        backPoint = backPointBuf.Get<float>();
        outMinIdxLocal = outMinIdxBuf.Get<int32_t>();
    }

    __aicore__ inline void BatchIdxSelect(LocalTensor<float> dstLocal, LocalTensor<int32_t> idxLocal,
        LocalTensor<float> srcLocal, int32_t dstOffset, int32_t compNum) {
        Add(selectIdx, idxLocal, batchIdxOffset, compNum);
        Muls(selectIdx, selectIdx, (int32_t)(2 * sizeof(float)), compNum);
        Gather(dstLocal, srcLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compNum);
        Adds(selectIdx, selectIdx, (int32_t)(sizeof(float)), compNum);
        Gather(dstLocal[dstOffset], srcLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compNum);
    }

  private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> polyLineBuf, minIdxBuf, ptBuf, backIdxBuf, outMinIdxBuf;
    TBuf<QuePosition::VECCALC> frontPointBuf, backPointBuf;
    TBuf<QuePosition::VECCALC> selectIdxBuf, batchIdxOffsetBuf;
    GlobalTensor<float> polyLineGm;
    GlobalTensor<int32_t> minIdxGm;
    GlobalTensor<float> ptGm;
    GlobalTensor<int32_t> backIdxGm;
    GlobalTensor<int32_t> outMinIdxGm;

    LocalTensor<float> polyLineLocal, ptLocal;
    LocalTensor<int32_t> minIdxLocal, backidxLocal, outMinIdxLocal;
    LocalTensor<float> frontPoint, backPoint;
    LocalTensor<int32_t> selectIdx, batchIdxOffset;

    uint32_t batchSize, numPoint, numIdx;
    uint32_t numTaskPerCore, numTaskTail, startTaskId;
    uint32_t numPointAligned, numIdxAligned, numIdxRound64;
    uint32_t ubLengthBatchPointSize, ubLengthBatchIdxSize;
    uint32_t compBatchNum;
};

extern "C" __global__ __aicore__ void select_idx_with_mask(GM_ADDR poly_line, GM_ADDR min_idx, GM_ADDR pt,
    GM_ADDR back_idx, GM_ADDR out_min_idx, GM_ADDR workspace, GM_ADDR tiling_data) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    KernelSelectIdxWithMask op;
    op.Init(poly_line, min_idx, pt, back_idx, out_min_idx, &tiling);
    op.Process();
}
