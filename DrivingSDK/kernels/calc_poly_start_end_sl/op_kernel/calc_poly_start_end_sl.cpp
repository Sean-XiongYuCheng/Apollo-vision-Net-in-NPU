/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
constexpr uint32_t FLOAT32_BYTES = 4;
}

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
        // 此处是计算当前核心需要处理的任务总数和该任务的开始ID
        if (coreId < numTaskRemained) {
            numTaskPerCore = numTaskPerCore + 1;
            startTaskId = numTaskPerCore * coreId;
        } else {
            startTaskId = numTaskPerCore * coreId + numTaskRemained;
        }

        compBatchNum = min(numTaskPerCore, compBatchNum);

        ubLengthBatchPointSize = compBatchNum * numPointAligned;
        ubLengthBatchIdxSize = compBatchNum * numIdxAligned;
        ubLengthBatchIdxSizeRound64 = compBatchNum * numIdxRound64;

        polyLineGm.SetGlobalBuffer((__gm__ float *)poly_line, batchSize * npoint * 2);
        minIdxGm.SetGlobalBuffer((__gm__ int32_t *)min_idx, batchSize * numIdx);
        pointsGm.SetGlobalBuffer((__gm__ float *)points, batchSize * numIdx * 2);
        sCumGm.SetGlobalBuffer((__gm__ float *)s_cum, batchSize * npoint);
        polyStartGm.SetGlobalBuffer((__gm__ float *)poly_start, batchSize * numIdx * 2);
        polyEndGm.SetGlobalBuffer((__gm__ float *)poly_end, batchSize * numIdx * 2);
        slGm.SetGlobalBuffer((__gm__ float *)sl, batchSize * numIdx * 2);

        pipe.InitBuffer(PolyLineBuf, ubLengthBatchPointSize * 2 * sizeof(float));
        pipe.InitBuffer(MinIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(PointsBuf, ubLengthBatchIdxSizeRound64 * 2 * sizeof(float));
        pipe.InitBuffer(SCumBuf, ubLengthBatchPointSize * sizeof(float));
        pipe.InitBuffer(PolyStartBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(PolyEndBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(SlBuf, ubLengthBatchIdxSize * 2 * sizeof(float));

        // 用于给中间变量申请UBuf内存空间使用
        pipe.InitBuffer(minIdxLocalsCumBuf, ubLengthBatchIdxSize * sizeof(int32_t));

        pipe.InitBuffer(polyStartBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(polyEndBufX, ubLengthBatchIdxSize * 2 * sizeof(float));

        pipe.InitBuffer(subPolyEndBufX, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(pointBufX, ubLengthBatchIdxSizeRound64 * 2 * sizeof(float));
        pipe.InitBuffer(dotValueBufX, ubLengthBatchIdxSize * 2 * sizeof(float));

        // batch 偏移表
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
        ptLocalx = pointBufX.Get<float>();
        dotLocalx = dotValueBufX.Get<float>();

        batchIdxOffset = batchIdxOffsetBuf.Get<int32_t>();
        sCumBatchOffset = sCumBatchOffsetBuf.Get<int32_t>();
        selectIdx = selectIdxBuf.Get<int32_t>();

        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        eventVMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

        // 创建[0 1 2 3 4 5 6 7 8 .. 2N]
        CreateVecIndex(offset, (int32_t)0, ubLengthBatchIdxSize * 2);
        // 转成float
        Cast(offsetFloat, offset, RoundMode::CAST_CEIL, ubLengthBatchIdxSize * 2);
        // [0.0 1.0 2.0 3.0 4.0 5.0] * 0.5 -> [0 0.5 1.0 1.5 2.0 2.5]
        Muls(offsetFloat, offsetFloat, (float)(5e-1), ubLengthBatchIdxSize * 2);
        // 转成int32_t [0 0 1 1 2 2]
        Cast(offsetY, offsetFloat, RoundMode::CAST_FLOOR, ubLengthBatchIdxSize * 2);
        // [0 0 1 1 2 2] * 2 -> [0 0 2 2 4 4]
        Muls(offsetY, offsetY, (int32_t)2, ubLengthBatchIdxSize * 2);
        // [0 1 2 3 4 5] - [0 0 2 2 4 4] -> [0 1 0 1 0 1]
        Sub(offset, offset, offsetY, ubLengthBatchIdxSize * 2);
        // 得到[0 4N 0 4N 0 4N ... 0 4N]
        Muls(offset, offset, (int32_t)(ubLengthBatchIdxSize * 4), ubLengthBatchIdxSize * 2);
        // [0 0 4 4 8 8 ... 4n-4 4n-4]
        Muls(offsetY, offsetY, (int32_t)2, ubLengthBatchIdxSize * 2);
        // [0 4N 4 4N+4   ... 4n+4n]
        Add(offset, offset, offsetY, ubLengthBatchIdxSize * 2);
    }

    __aicore__ inline void Process() {
        uint64_t rsvdCnt = 0;

        //  预计算batch偏移表： batchIdxOffset: [0 0...0(共numIdx个)... batch序号*npoint*2*4 batch序号*npoint*2*4...batch序号*npoint*2*4(共numIdx个)... ]
        //  用于Gather时将batch内局部索引转为poly_line中的全局字节偏移
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(batchIdxOffset, (int32_t)((batchIdx - 1) * npoint * 2 * FLOAT32_BYTES), batchIdx * numIdx);
        }
        //  预计算s_cum的batch偏移：sCumBatchOffset[i] = i * npoint * 4（字节偏移，每个float=4字节）
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(sCumBatchOffset, (int32_t)((batchIdx - 1) * npoint * FLOAT32_BYTES), batchIdx * numIdx);
        }
        for (uint32_t loopTimes = 0; loopTimes < numTaskPerCore; loopTimes += compBatchNum) {
            uint32_t batchIdx = loopTimes + startTaskId;
            uint32_t compBatchNumActual =
                loopTimes + compBatchNum <= numTaskPerCore ? compBatchNum : numTaskPerCore - loopTimes;

            uint32_t compBatchPointSize = compBatchNumActual * npoint;
            uint32_t compBatchIdxSize = compBatchNumActual * numIdx;
            uint16_t gatherRepeatTimes = (2 * compBatchIdxSize + 63) / 64;

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

            //  s_cum: minIdx * 4 + batchOffset （每点4字节）
            Muls(minIdxLocalscum, minIdxLocal, (int32_t)4, compBatchIdxSize);
            Add(minIdxLocalscum, minIdxLocalscum, sCumBatchOffset, compBatchIdxSize);

            //  poly_line: minIdx * 8 + batchOffset (每点8字节)
            Muls(selectIdx, minIdxLocal, (int32_t)8, compBatchIdxSize);
            Add(selectIdx, selectIdx, batchIdxOffset, compBatchIdxSize);

            //  Gather 线段的起点和终点
            Gather(polyStartX, polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compBatchIdxSize);
            Gather(polyStartX[ubLengthBatchIdxSize], polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)4,
                compBatchIdxSize);
            Gather(polyEndX, polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)8, compBatchIdxSize);
            Gather(polyEndX[ubLengthBatchIdxSize], polyLineLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)12,
                compBatchIdxSize);

            //  Gather分离points的x/y
            GatherMask(ptLocalx, pointsLocal, 1, false, 0, {1, gatherRepeatTimes, 8, 8}, rsvdCnt);
            GatherMask(ptLocalx[ubLengthBatchIdxSize], pointsLocal, 2, false, 0, {1, gatherRepeatTimes, 8, 8}, rsvdCnt);

            Sub(ptLocalx, ptLocalx, polyStartX, ubLengthBatchIdxSize * 2);
            Sub(subPolyEndX, polyEndX, polyStartX, ubLengthBatchIdxSize * 2);
            // 点积
            Mul(dotLocalx, ptLocalx, subPolyEndX, ubLengthBatchIdxSize * 2);
            Add(dotLocalx, dotLocalx, dotLocalx[ubLengthBatchIdxSize], compBatchIdxSize);
            // 叉积
            Mul(ptLocalx, ptLocalx, subPolyEndX[ubLengthBatchIdxSize], compBatchIdxSize);
            Mul(ptLocalx[ubLengthBatchIdxSize], ptLocalx[ubLengthBatchIdxSize], subPolyEndX, compBatchIdxSize);
            Sub(dotLocalx[ubLengthBatchIdxSize], ptLocalx, ptLocalx[ubLengthBatchIdxSize], compBatchIdxSize);
            // 线段长度
            Mul(subPolyEndX, subPolyEndX, subPolyEndX, ubLengthBatchIdxSize * 2);
            Add(subPolyEndX, subPolyEndX, subPolyEndX[ubLengthBatchIdxSize], compBatchIdxSize);
            Sqrt(subPolyEndX, subPolyEndX, compBatchIdxSize);

            Mins(subPolyEndX, subPolyEndX, static_cast<float>(1e10), compBatchIdxSize);
            Maxs(subPolyEndX, subPolyEndX, static_cast<float>(1e-5), compBatchIdxSize);

            //  计算s l
            Div(dotLocalx, dotLocalx, subPolyEndX, compBatchIdxSize);
            Div(dotLocalx[ubLengthBatchIdxSize], dotLocalx[ubLengthBatchIdxSize], subPolyEndX, compBatchIdxSize);
            Muls(dotLocalx[ubLengthBatchIdxSize], dotLocalx[ubLengthBatchIdxSize], static_cast<float>(-1),
                compBatchIdxSize);

            Gather(subPolyEndX, sCumLocal, minIdxLocalscum.ReinterpretCast<uint32_t>(), (uint32_t)0, compBatchIdxSize);
            Add(dotLocalx, dotLocalx, subPolyEndX, compBatchIdxSize);

            //  用预计算 offset 将 [s0,s1,...,l0,l1,...] 交错为 [s0,l0,s1,l1,...] ----
            // s_cum[min_idx] 已在上方 Gather 并加到 s 上
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
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> PolyLineBuf, MinIdxBuf, PointsBuf, SCumBuf, PolyStartBuf, PolyEndBuf,
        SlBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> minIdxLocalsCumBuf, polyStartBufX, polyEndBufX, subPolyEndBufX,
        dotValueBufX, pointBufX, offsetBufX, offsetBufY, offsetFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> batchIdxOffsetBuf, sCumBatchOffsetBuf, selectIdxBuf;
    AscendC::GlobalTensor<float> polyLineGm;
    AscendC::GlobalTensor<int32_t> minIdxGm;
    AscendC::GlobalTensor<float> pointsGm;
    AscendC::GlobalTensor<float> sCumGm;
    AscendC::GlobalTensor<float> polyStartGm;
    AscendC::GlobalTensor<float> polyEndGm;
    AscendC::GlobalTensor<float> slGm;

    AscendC::LocalTensor<int32_t> offset;
    AscendC::LocalTensor<float> polyLineLocal;
    AscendC::LocalTensor<int32_t> minIdxLocal;
    AscendC::LocalTensor<float> pointsLocal;
    AscendC::LocalTensor<float> sCumLocal;
    AscendC::LocalTensor<float> polyStart;
    AscendC::LocalTensor<float> polyEnd;
    AscendC::LocalTensor<float> slLocal;
    AscendC::LocalTensor<int32_t> minIdxLocalscum;
    AscendC::LocalTensor<float> polyStartX;
    AscendC::LocalTensor<float> polyEndX;
    AscendC::LocalTensor<float> subPolyEndX;
    AscendC::LocalTensor<float> ptLocalx;
    AscendC::LocalTensor<float> dotLocalx;
    AscendC::LocalTensor<int32_t> batchIdxOffset, sCumBatchOffset, selectIdx;

    uint32_t batchSize, npoint, numIdx, numIdxAligned, numTaskPerCore, numTaskRemained, startTaskId;
    uint32_t compBatchNum;
    uint32_t ubLengthBatchPointSize, ubLengthBatchIdxSize, ubLengthBatchIdxSizeRound64;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_, eventVMTE2_;
    uint32_t numPointAligned, numIdxRound64;
};

extern "C" __global__ __aicore__ void calc_poly_start_end_sl(GM_ADDR min_idx, GM_ADDR poly_line, GM_ADDR pt,
    GM_ADDR s_cum, GM_ADDR poly_start, GM_ADDR poly_end, GM_ADDR sl, GM_ADDR workspace, GM_ADDR tiling_data) {
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CalcPolyStartEndSl op;
    op.Init(min_idx, poly_line, pt, s_cum, poly_start, poly_end, sl, &tiling);
    op.Process();
}
