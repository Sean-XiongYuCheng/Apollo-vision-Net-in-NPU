/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t FLOAT32_BYTES = 4;

class CalcPolyStartEndSl {
  public:
    __aicore__ inline CalcPolyStartEndSl() {}
    __aicore__ inline void Init(GM_ADDR min_idx, GM_ADDR poly_line, GM_ADDR points, GM_ADDR s_cum, GM_ADDR poly_start,
        GM_ADDR poly_end, GM_ADDR sl, CalcPolyStartEndSlTilingData *tiling) {
        batchSize = tiling->batchSize;
        npoint = tiling->npoints;
        numIdx = tiling->numIdx;
        numTaskPerCore = tiling->numTaskPerCore;
        numTaskRemained = tiling->numTaskRemained;
        compBatchNum = tiling->compBatchNum;

        uint32_t blockSize = 32;
        uint32_t dataSize = blockSize / sizeof(float);

        numPointAligned = (npoint + dataSize - 1) / dataSize * dataSize;
        numIdxAligned = (numIdx + dataSize - 1) / dataSize * dataSize;
        numIdxRound64 = (numIdx + 63) / 64 * 64;

        uint32_t coreId = GetBlockIdx();
        if (coreId < numTaskRemained) {
            numTaskPerCore = numTaskPerCore + 1;
            startTaskId = numTaskPerCore * coreId;
        } else {
            startTaskId = numTaskPerCore * coreId + numTaskRemained;
        }
        compBatchNum = min(numTaskPerCore, compBatchNum);

        // arch35: 所有 idx 缓冲区按 numIdxRound64 对齐
        ubLengthBatchPointSize = compBatchNum * numPointAligned;
        ubLengthBatchIdxSize = compBatchNum * numIdxRound64;

        polyLineGm.SetGlobalBuffer((__gm__ float *)poly_line, batchSize * npoint * 2);
        minIdxGm.SetGlobalBuffer((__gm__ int32_t *)min_idx, batchSize * numIdx);
        pointsGm.SetGlobalBuffer((__gm__ float *)points, batchSize * numIdx * 2);
        sCumGm.SetGlobalBuffer((__gm__ float *)s_cum, batchSize * npoint);
        polyStartGm.SetGlobalBuffer((__gm__ float *)poly_start, batchSize * numIdx * 2);
        polyEndGm.SetGlobalBuffer((__gm__ float *)poly_end, batchSize * numIdx * 2);
        slGm.SetGlobalBuffer((__gm__ float *)sl, batchSize * numIdx * 2);

        pipe.InitBuffer(PolyLineBuf, ubLengthBatchPointSize * 2 * sizeof(float));
        pipe.InitBuffer(MinIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(PointsBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(SCumBuf, ubLengthBatchPointSize * sizeof(float));
        pipe.InitBuffer(PolyStartBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(PolyEndBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(SlBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(minIdxLocalsCumBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(polyStartBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(polyEndBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(subPolyEndBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(dotValueBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(batchIdxOffsetBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(sCumBatchOffsetBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(selectIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(offsetBufX, ubLengthBatchIdxSize * 2 * sizeof(int32_t));
        pipe.InitBuffer(offsetBufY, ubLengthBatchIdxSize * 2 * sizeof(int32_t));
        pipe.InitBuffer(offsetFloatBuf, ubLengthBatchIdxSize * 2 * sizeof(float));

        offset = offsetBufX.Get<int32_t>();
        LocalTensor<int32_t> offsetY = offsetBufY.Get<int32_t>();
        LocalTensor<float> offsetFloat = offsetFloatBuf.Get<float>();

        polyLineLocal = PolyLineBuf.Get<float>();
        minIdxLocal = MinIdxBuf.Get<int32_t>();
        pointsLocal = PointsBuf.Get<float>();
        sCumLocal = SCumBuf.Get<float>();
        polyStart = PolyStartBuf.Get<float>();
        polyEnd = PolyEndBuf.Get<float>();
        slLocal = SlBuf.Get<float>();

        minIdxLocalscum = minIdxLocalsCumBuf.Get<int32_t>();
        polyStartX = polyStartBufX.Get<float>();
        polyEndX = polyEndBufX.Get<float>();
        subPolyEndX = subPolyEndBufX.Get<float>();
        dotLocalx = dotValueBufX.Get<float>();

        batchIdxOffset = batchIdxOffsetBuf.Get<int32_t>();
        sCumBatchOffset = sCumBatchOffsetBuf.Get<int32_t>();
        selectIdx = selectIdxBuf.Get<int32_t>();

        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        eventVMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

        // 预计算 offset 表 (传统 API, Init 中一次性完成)
        CreateVecIndex(offset, (int32_t)0, ubLengthBatchIdxSize * 2);
        Cast(offsetFloat, offset, RoundMode::CAST_CEIL, ubLengthBatchIdxSize * 2);
        Muls(offsetFloat, offsetFloat, (float)(5e-1), ubLengthBatchIdxSize * 2);
        Cast(offsetY, offsetFloat, RoundMode::CAST_FLOOR, ubLengthBatchIdxSize * 2);
        Muls(offsetY, offsetY, (int32_t)2, ubLengthBatchIdxSize * 2);
        Sub(offset, offset, offsetY, ubLengthBatchIdxSize * 2);
        Muls(offset, offset, (int32_t)(ubLengthBatchIdxSize * 4), ubLengthBatchIdxSize * 2);
        Muls(offsetY, offsetY, (int32_t)2, ubLengthBatchIdxSize * 2);
        Add(offset, offset, offsetY, ubLengthBatchIdxSize * 2);
    }

    __aicore__ inline void Process() {
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(batchIdxOffset, (int32_t)((batchIdx - 1) * npoint * 2 * FLOAT32_BYTES), batchIdx * numIdx);
        }
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(sCumBatchOffset, (int32_t)((batchIdx - 1) * npoint * FLOAT32_BYTES), batchIdx * numIdx);
        }

        for (uint32_t loopTimes = 0; loopTimes < numTaskPerCore; loopTimes += compBatchNum) {
            uint32_t batchIdx = loopTimes + startTaskId;
            uint32_t compBatchNumActual =
                loopTimes + compBatchNum <= numTaskPerCore ? compBatchNum : numTaskPerCore - loopTimes;

            uint32_t compBatchPointSize = compBatchNumActual * npoint;
            uint32_t compBatchIdxSize = compBatchNumActual * numIdx;

            SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

            DataCopyPad(polyLineLocal, polyLineGm[batchIdx * npoint * 2],
                {1, static_cast<uint32_t>(compBatchPointSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(sCumLocal, sCumGm[batchIdx * npoint],
                {1, static_cast<uint32_t>(compBatchPointSize * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(minIdxLocal, minIdxGm[batchIdx * numIdx],
                {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(pointsLocal, pointsGm[batchIdx * numIdx * 2],
                {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

            SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);

            Muls(minIdxLocalscum, minIdxLocal, (int32_t)4, compBatchIdxSize);
            Add(minIdxLocalscum, minIdxLocalscum, sCumBatchOffset, compBatchIdxSize);
            Muls(selectIdx, minIdxLocal, (int32_t)8, compBatchIdxSize);
            Add(selectIdx, selectIdx, batchIdxOffset, compBatchIdxSize);

            Gather(polyStartX, polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compBatchIdxSize);
            Gather(polyStartX[ubLengthBatchIdxSize], polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)4,
                compBatchIdxSize);
            Gather(polyEndX, polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)8, compBatchIdxSize);
            Gather(polyEndX[ubLengthBatchIdxSize], polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)12,
                compBatchIdxSize);

            {
                __local_mem__ float *ptInterleavedPtr = (__local_mem__ float *)pointsLocal.GetPhyAddr();
                __local_mem__ float *psPtr = (__local_mem__ float *)polyStartX.GetPhyAddr();
                __local_mem__ float *pePtr = (__local_mem__ float *)polyEndX.GetPhyAddr();
                __local_mem__ float *dotPtr = (__local_mem__ float *)dotLocalx.GetPhyAddr();
                uint16_t rpt1 = DivCeil(compBatchIdxSize, (uint16_t)64);

                PipeBarrier<PIPE_ALL>();
                __VEC_SCOPE__ {
                    MicroAPI::RegTensor<float> ptSrcReg1, ptSrcReg2, ptX, ptY;
                    MicroAPI::RegTensor<float> psX, psY, peX, peY;
                    MicroAPI::RegTensor<float> subX, subY, dotVal, crossVal, tmpReg, normReg;
                    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MaskPattern::ALL>();

                    for (uint16_t i = 0; i < rpt1; i++) {
                        uint32_t off = i * 64;
                        uint32_t off128 = i * 128;
                        uint32_t offY = ubLengthBatchIdxSize + off;

                        // DeInterleave: 从交错的 points [x0,y0,x1,y1,...] 分离 x/y
                        MicroAPI::DataCopy(ptSrcReg1, ptInterleavedPtr + off128);
                        MicroAPI::DataCopy(ptSrcReg2, ptInterleavedPtr + off128 + 64);
                        MicroAPI::DeInterleave(ptX, ptY, ptSrcReg1, ptSrcReg2);

                        // 加载 polyStart / polyEnd
                        MicroAPI::DataCopy(psX, psPtr + off);
                        MicroAPI::DataCopy(psY, psPtr + offY);
                        MicroAPI::DataCopy(peX, pePtr + off);
                        MicroAPI::DataCopy(peY, pePtr + offY);

                        // sub = polyEnd - polyStart
                        MicroAPI::Sub(subX, peX, psX, mask);
                        MicroAPI::Sub(subY, peY, psY, mask);

                        // pt = pt - polyStart
                        MicroAPI::Sub(ptX, ptX, psX, mask);
                        MicroAPI::Sub(ptY, ptY, psY, mask);

                        // dot = pt_x * sub_x + pt_y * sub_y
                        MicroAPI::Mul(dotVal, ptX, subX, mask);
                        MicroAPI::Mul(tmpReg, ptY, subY, mask);
                        MicroAPI::Add(dotVal, dotVal, tmpReg, mask);

                        // cross = pt_x * sub_y - pt_y * sub_x
                        MicroAPI::Mul(tmpReg, ptX, subY, mask);
                        MicroAPI::Mul(crossVal, ptY, subX, mask);
                        MicroAPI::Sub(crossVal, tmpReg, crossVal, mask);

                        // norm = sqrt(sub_x^2 + sub_y^2)
                        MicroAPI::Mul(subX, subX, subX, mask);
                        MicroAPI::Mul(subY, subY, subY, mask);
                        MicroAPI::Add(normReg, subX, subY, mask);
                        MicroAPI::Sqrt(normReg, normReg, mask);

                        // clip
                        MicroAPI::Mins(normReg, normReg, 1e10f, mask);
                        MicroAPI::Maxs(normReg, normReg, 1e-5f, mask);

                        // s = dot / norm, l = -cross / norm
                        MicroAPI::Div(dotVal, dotVal, normReg, mask);
                        MicroAPI::Div(crossVal, crossVal, normReg, mask);
                        MicroAPI::Muls(crossVal, crossVal, -1.0f, mask);

                        // 只写回最终结果
                        MicroAPI::DataCopy(dotPtr + off, dotVal, mask);
                        MicroAPI::DataCopy(dotPtr + offY, crossVal, mask);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }

            Gather(subPolyEndX, sCumLocal, minIdxLocalscum.ReinterpretCast<uint32_t>(), (uint32_t)0, compBatchIdxSize);
            Add(dotLocalx, dotLocalx, subPolyEndX, compBatchIdxSize);

            Gather(polyStart, polyStartX, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, ubLengthBatchIdxSize * 2);
            Gather(polyEnd, polyEndX, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, ubLengthBatchIdxSize * 2);
            Gather(slLocal, dotLocalx, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, ubLengthBatchIdxSize * 2);

            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

            DataCopyPad(polyStartGm[batchIdx * numIdx * 2], polyStart,
                {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0});
            DataCopyPad(polyEndGm[batchIdx * numIdx * 2], polyEnd,
                {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0});
            DataCopyPad(slGm[batchIdx * numIdx * 2], slLocal,
                {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0});
        }
    }

  private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> PolyLineBuf, MinIdxBuf, PointsBuf, SCumBuf, PolyStartBuf, PolyEndBuf, SlBuf;
    TBuf<QuePosition::VECCALC> minIdxLocalsCumBuf, polyStartBufX, polyEndBufX, subPolyEndBufX, dotValueBufX, offsetBufX,
        offsetBufY, offsetFloatBuf;
    TBuf<QuePosition::VECCALC> batchIdxOffsetBuf, sCumBatchOffsetBuf, selectIdxBuf;
    GlobalTensor<float> polyLineGm;
    GlobalTensor<int32_t> minIdxGm;
    GlobalTensor<float> pointsGm;
    GlobalTensor<float> sCumGm;
    GlobalTensor<float> polyStartGm;
    GlobalTensor<float> polyEndGm;
    GlobalTensor<float> slGm;

    LocalTensor<int32_t> offset;
    LocalTensor<float> polyLineLocal;
    LocalTensor<int32_t> minIdxLocal;
    LocalTensor<float> pointsLocal;
    LocalTensor<float> sCumLocal;
    LocalTensor<float> polyStart;
    LocalTensor<float> polyEnd;
    LocalTensor<float> slLocal;
    LocalTensor<int32_t> minIdxLocalscum;
    LocalTensor<float> polyStartX;
    LocalTensor<float> polyEndX;
    LocalTensor<float> subPolyEndX;
    LocalTensor<float> dotLocalx;
    LocalTensor<int32_t> batchIdxOffset, sCumBatchOffset, selectIdx;

    uint32_t batchSize, npoint, numIdx, numIdxAligned, numTaskPerCore, numTaskRemained, startTaskId;
    uint32_t compBatchNum;
    uint32_t ubLengthBatchPointSize, ubLengthBatchIdxSize;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_, eventVMTE2_;
    uint32_t numPointAligned, numIdxRound64;
};

extern "C" __global__ __aicore__ void calc_poly_start_end_sl(GM_ADDR min_idx, GM_ADDR poly_line, GM_ADDR pt,
    GM_ADDR s_cum, GM_ADDR poly_start, GM_ADDR poly_end, GM_ADDR sl, GM_ADDR workspace, GM_ADDR tiling_data) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CalcPolyStartEndSl op;
    op.Init(min_idx, poly_line, pt, s_cum, poly_start, poly_end, sl, &tiling);
    op.Process();
}
