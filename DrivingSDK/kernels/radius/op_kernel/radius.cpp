/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
constexpr uint32_t GROUP_NUM = 2; //Inputs only contain group X and group Y
constexpr uint32_t ALIGN_NUM_8 = 8;
constexpr uint32_t ALIGN_NUM_64 = 64; // CompareScalar function requires 256B alignment.
}

class KernelRadius {
  public:
    __aicore__ inline KernelRadius() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR outFinal,
        GM_ADDR numNeighbors, GM_ADDR usrWorkspace, RadiusTilingData *tiling_data) {
        InitTilingParams(tiling_data);
        InitGlobalTensor(x, y, ptrX, ptrY, outFinal, numNeighbors, usrWorkspace, tiling_data->batchSize);
        InitUbBuffers(tiling_data->coreIndexSize, tiling_data->batchPointsSize);
        InitLocalTensor();
        SyncAll();
    }

    __aicore__ inline void Process() {
        // Input batch address pointer
        LoadBatchPointers();
        for (uint32_t i = 0; i < batchThisCore; ++i) {
            int32_t ptrXLeft = ptrXLocal.GetValue(i);
            int32_t ptrYLeft = ptrYLocal.GetValue(i);
            int32_t ptrXRight = ptrXLocal.GetValue(i + 1);
            int32_t ptrYRight = ptrYLocal.GetValue(i + 1);
            uint32_t numBatchPointsX = ptrXRight - ptrXLeft;
            uint32_t numBatchPointsY = ptrYRight - ptrYLeft;
            if (numBatchPointsX == 0 || numBatchPointsY == 0) {
                continue;
            }
            SetFlag<HardEvent::S_MTE2>(0);
            WaitFlag<HardEvent::S_MTE2>(0);
            // According to the batch address pointer, points within a single batch are moved in
            LoadBatchPoints(ptrXLeft, ptrYLeft, numBatchPointsX, numBatchPointsY);
            FindBatchNeighbors(ptrXLeft, ptrYLeft, numBatchPointsX, numBatchPointsY);
        }
        AccumulateCoreOutputOffsets();
        SyncAll();
        // Rearrange non-contiguous data to ensure continuous distribution in memory
        CompactWorkspaceOutput();
    }

  private:
    __aicore__ inline void InitTilingParams(RadiusTilingData *tiling_data) {
        coordinateDim = tiling_data->coordinateDim;
        numPointsX = tiling_data->numPointsX;
        numPointsY = tiling_data->numPointsY;
        maxNumNeighbors = tiling_data->maxNumNeighbors;
        coreOutputOffsetSlots = tiling_data->coreOutputOffsetSlots;
        coreIndexCount = tiling_data->coreIndexCount;
        batchPointsCount = tiling_data->batchPointsCount;
        usedCoreNum = tiling_data->usedCoreNum;
        rSquare = tiling_data->rSquare;
        blockIdx = GetBlockIdx();
        numOutputPoints = 0;

        uint32_t headCoreNum = tiling_data->headCoreNum;
        uint32_t batchPerCore = tiling_data->batchPerCore;
        uint32_t batchPerCoreTail = tiling_data->batchPerCoreTail;
        if (blockIdx < headCoreNum) {
            batchThisCore = batchPerCore;
            ptrAddrOffset = blockIdx * batchPerCore;
        } else {
            batchThisCore = batchPerCoreTail;
            ptrAddrOffset = blockIdx * batchPerCore - (blockIdx - headCoreNum);
        }
    }

    __aicore__ inline void InitGlobalTensor(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR outFinal,
        GM_ADDR numNeighbors, GM_ADDR usrWorkspace, uint32_t batchSize) {
        uint32_t outTempDataNum = GROUP_NUM * numPointsY * maxNumNeighbors;

        xGm.SetGlobalBuffer((__gm__ float *)x, coordinateDim * numPointsX);
        yGm.SetGlobalBuffer((__gm__ float *)y, coordinateDim * numPointsY);
        ptrXGm.SetGlobalBuffer((__gm__ int32_t *)ptrX, batchSize + 1);
        ptrYGm.SetGlobalBuffer((__gm__ int32_t *)ptrY, batchSize + 1);
        outFinalGm.SetGlobalBuffer((__gm__ int32_t *)outFinal, outTempDataNum);
        numNeighborsGm.SetGlobalBuffer(
            (__gm__ int32_t *)numNeighbors, coreOutputOffsetSlots * ALIGN_NUM_8); // [slots, 8]

        outTempWsp.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, outTempDataNum);

        // numNeighborsGm is also used as per-core prefix workspace and needs to be initialized to 0.
        if (blockIdx == 0) {
            InitOutput<int32_t>(numNeighborsGm, (usedCoreNum + 1) * ALIGN_NUM_8, 0);
        }
    }

    __aicore__ inline void InitUbBuffers(uint32_t coreIndexSize, uint32_t batchPointsSize) {
        pipe.InitBuffer(ptrXBuf, coreIndexSize);
        pipe.InitBuffer(ptrYBuf, coreIndexSize);
        pipe.InitBuffer(xBuf, batchPointsSize);
        pipe.InitBuffer(yBuf, batchPointsSize);
        pipe.InitBuffer(distBuf, batchPointsCount * sizeof(float));
        pipe.InitBuffer(tempBuf, 2 * batchPointsCount * sizeof(float));
        pipe.InitBuffer(maskBuf, batchPointsCount * sizeof(float) / sizeof(int32_t));
        pipe.InitBuffer(baseIndexBuf, batchPointsCount * sizeof(int32_t));
        pipe.InitBuffer(indexXBuf, batchPointsCount * sizeof(float));
        pipe.InitBuffer(indexYBuf, batchPointsCount * sizeof(float));
        pipe.InitBuffer(numNeighborsBuf, ALIGN_NUM_8 * sizeof(int32_t));
        // Store each core's neighbor count and one extra prefix slot, with 8 int32 values per slot.
        pipe.InitBuffer(numNeighborsCoreBuf, (usedCoreNum + 1) * ALIGN_NUM_8 * sizeof(int32_t));
    }

    __aicore__ inline void InitLocalTensor() {
        ptrXLocal = ptrXBuf.Get<int32_t>();
        ptrYLocal = ptrYBuf.Get<int32_t>();
        pointsXLocal = xBuf.Get<float>();
        pointsYLocal = yBuf.Get<float>();
        distLocal = distBuf.Get<float>();
        tempLocal = tempBuf.Get<float>();
        maskUint8 = maskBuf.Get<uint8_t>();
        baseIndexTensor = baseIndexBuf.Get<int32_t>();
        indexXTensor = indexXBuf.Get<int32_t>();
        indexYTensor = indexYBuf.Get<int32_t>();
        numNeighborsTensor = numNeighborsBuf.Get<int32_t>();
        numNeighborsCoreTensor = numNeighborsCoreBuf.Get<int32_t>();
    }

    __aicore__ inline void LoadBatchPointers() {
        DataCopyParams copyParams{1, static_cast<uint16_t>(coreIndexCount * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(ptrXLocal, ptrXGm[ptrAddrOffset], copyParams, padParams);
        DataCopyPad(ptrYLocal, ptrYGm[ptrAddrOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        outputWspOffsetY = ptrYLocal.GetValue(0) * maxNumNeighbors;
        outputWspOffsetX = outputWspOffsetY + numPointsY * maxNumNeighbors;
    }

    __aicore__ inline void LoadBatchPoints(
        int32_t ptrXLeft, int32_t ptrYLeft, uint32_t numBatchPointsX, uint32_t numBatchPointsY) {
        DataCopyParams copyParamsX{1, static_cast<uint16_t>(numBatchPointsX * sizeof(float)), 0, 0};
        DataCopyParams copyParamsY{1, static_cast<uint16_t>(numBatchPointsY * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        for (int32_t dim = 0; dim < coordinateDim; ++dim) {
            DataCopyPad(pointsXLocal[batchPointsCount * dim], xGm[numPointsX * dim + ptrXLeft], copyParamsX, padParams);
            DataCopyPad(pointsYLocal[batchPointsCount * dim], yGm[numPointsY * dim + ptrYLeft], copyParamsY, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void FindBatchNeighbors(
        int32_t ptrXLeft, int32_t ptrYLeft, uint32_t numBatchPointsX, uint32_t numBatchPointsY) {
        uint32_t numBatchPointsVecAlignedX = AlignUp(numBatchPointsX, ALIGN_NUM_8);
        uint32_t numBatchPointsCmpAlignedX = AlignUp(numBatchPointsX, ALIGN_NUM_64);
        CreateVecIndex(baseIndexTensor, ptrXLeft, numBatchPointsVecAlignedX);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        SetFlag<HardEvent::MTE3_V>(0);

        if (coordinateDim == 1) {
            for (int32_t pointIdx = 0; pointIdx < numBatchPointsY; ++pointIdx) {
                float yi0 = -1.0F * pointsYLocal.GetValue(pointIdx);
                SetFlag<HardEvent::S_V>(0);
                WaitFlag<HardEvent::S_V>(0);
                Adds(distLocal, pointsXLocal, yi0, numBatchPointsVecAlignedX);
                Mul(distLocal, distLocal, distLocal, numBatchPointsVecAlignedX);
                SelectAndOutput(pointIdx + ptrYLeft, numBatchPointsX, numBatchPointsCmpAlignedX);
            }
        } else if (coordinateDim == 2) {
            for (int32_t pointIdx = 0; pointIdx < numBatchPointsY; ++pointIdx) {
                CalcDistance2D(pointIdx, numBatchPointsVecAlignedX);
                SelectAndOutput(pointIdx + ptrYLeft, numBatchPointsX, numBatchPointsCmpAlignedX);
            }
        } else if (coordinateDim == 3) {
            for (int32_t pointIdx = 0; pointIdx < numBatchPointsY; ++pointIdx) {
                CalcDistance3D(pointIdx, numBatchPointsVecAlignedX);
                SelectAndOutput(pointIdx + ptrYLeft, numBatchPointsX, numBatchPointsCmpAlignedX);
            }
        } else {
            for (int32_t pointIdx = 0; pointIdx < numBatchPointsY; ++pointIdx) {
                CalcDistanceGeneric(pointIdx, numBatchPointsVecAlignedX);
                SelectAndOutput(pointIdx + ptrYLeft, numBatchPointsX, numBatchPointsCmpAlignedX);
            }
        }
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void SelectAndOutput(
        int32_t pointIdxAbs, uint32_t numBatchPointsX, uint32_t numBatchPointsCmpAlignedX) {
        uint64_t selectCnt = 0;
        PipeBarrier<PIPE_V>();
        CompareScalar(maskUint8, distLocal, rSquare, CMPMODE::LT, numBatchPointsCmpAlignedX);
        LocalTensor<uint32_t> maskUint32 = maskUint8.ReinterpretCast<uint32_t>();
        WaitFlag<HardEvent::MTE3_V>(0);
        GatherMask(indexXTensor, baseIndexTensor, maskUint32, true, numBatchPointsX, {1, 1, 0, 0}, selectCnt);

        if (selectCnt > maxNumNeighbors) {
            selectCnt = maxNumNeighbors;
        }
        if (selectCnt > 0) {
            Duplicate(indexYTensor, pointIdxAbs, AlignUp(static_cast<uint32_t>(selectCnt), ALIGN_NUM_8));
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (selectCnt > 0) {
            DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(selectCnt * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(outTempWsp[outputWspOffsetX + numOutputPoints], indexXTensor, outCopyParams);
            DataCopyPad(outTempWsp[outputWspOffsetY + numOutputPoints], indexYTensor, outCopyParams);
            numOutputPoints += selectCnt;
        }
        SetFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void CalcDistance2D(int32_t pointIdx, uint32_t numBatchPointsVecAlignedX) {
        float yi0 = -1.0F * pointsYLocal.GetValue(pointIdx);
        float yi1 = -1.0F * pointsYLocal.GetValue(batchPointsCount + pointIdx);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Adds(distLocal, pointsXLocal, yi0, numBatchPointsVecAlignedX);
        Adds(tempLocal, pointsXLocal[batchPointsCount], yi1, numBatchPointsVecAlignedX);
        Mul(distLocal, distLocal, distLocal, numBatchPointsVecAlignedX);
        Mul(tempLocal, tempLocal, tempLocal, numBatchPointsVecAlignedX);
        Add(distLocal, distLocal, tempLocal, numBatchPointsVecAlignedX);
    }

    __aicore__ inline void CalcDistance3D(int32_t pointIdx, uint32_t numBatchPointsVecAlignedX) {
        float yi0 = -1.0F * pointsYLocal.GetValue(pointIdx);
        float yi1 = -1.0F * pointsYLocal.GetValue(batchPointsCount + pointIdx);
        float yi2 = -1.0F * pointsYLocal.GetValue(2 * batchPointsCount + pointIdx);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Adds(distLocal, pointsXLocal, yi0, numBatchPointsVecAlignedX);
        Adds(tempLocal, pointsXLocal[batchPointsCount], yi1, numBatchPointsVecAlignedX);
        Adds(tempLocal[batchPointsCount], pointsXLocal[2 * batchPointsCount], yi2, numBatchPointsVecAlignedX);
        Mul(distLocal, distLocal, distLocal, numBatchPointsVecAlignedX);
        Mul(tempLocal, tempLocal, tempLocal, numBatchPointsVecAlignedX);
        Mul(tempLocal[batchPointsCount], tempLocal[batchPointsCount], tempLocal[batchPointsCount],
            numBatchPointsVecAlignedX);
        Add(distLocal, distLocal, tempLocal, numBatchPointsVecAlignedX);
        Add(distLocal, distLocal, tempLocal[batchPointsCount], numBatchPointsVecAlignedX);
    }

    __aicore__ inline void CalcDistanceGeneric(int32_t pointIdx, uint32_t numBatchPointsVecAlignedX) {
        float yi = -1.0F * pointsYLocal.GetValue(pointIdx);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Adds(distLocal, pointsXLocal, yi, numBatchPointsVecAlignedX);
        Mul(distLocal, distLocal, distLocal, numBatchPointsVecAlignedX);

        uint32_t dim = 1;
        for (; dim + 1 < coordinateDim; dim += 2) {
            float yiPing = -1.0F * pointsYLocal.GetValue(batchPointsCount * dim + pointIdx);
            float yiPong = -1.0F * pointsYLocal.GetValue(batchPointsCount * (dim + 1) + pointIdx);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Adds(tempLocal, pointsXLocal[batchPointsCount * dim], yiPing, numBatchPointsVecAlignedX);
            Adds(tempLocal[batchPointsCount], pointsXLocal[batchPointsCount * (dim + 1)], yiPong,
                numBatchPointsVecAlignedX);
            Mul(tempLocal, tempLocal, tempLocal, numBatchPointsVecAlignedX);
            Mul(tempLocal[batchPointsCount], tempLocal[batchPointsCount], tempLocal[batchPointsCount],
                numBatchPointsVecAlignedX);
            Add(distLocal, distLocal, tempLocal, numBatchPointsVecAlignedX);
            Add(distLocal, distLocal, tempLocal[batchPointsCount], numBatchPointsVecAlignedX);
        }
        if (dim < coordinateDim) {
            yi = -1.0F * pointsYLocal.GetValue(batchPointsCount * dim + pointIdx);
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            Adds(tempLocal, pointsXLocal[batchPointsCount * dim], yi, numBatchPointsVecAlignedX);
            Mul(tempLocal, tempLocal, tempLocal, numBatchPointsVecAlignedX);
            Add(distLocal, distLocal, tempLocal, numBatchPointsVecAlignedX);
        }
    }

    __aicore__ inline void AccumulateCoreOutputOffsets() {
        Duplicate(numNeighborsCoreTensor, numOutputPoints, usedCoreNum * ALIGN_NUM_8);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        SetAtomicAdd<int32_t>();
        DataCopyParams outCopyParams{
            1, static_cast<uint16_t>((usedCoreNum - blockIdx) * ALIGN_NUM_8 * sizeof(int32_t)), 0, 0};
        DataCopyPad(numNeighborsGm, numNeighborsCoreTensor, outCopyParams);
        SetAtomicNone();
    }
    __aicore__ inline void CompactWorkspaceOutput() {
        uint32_t numIters = DivCeil(numOutputPoints, static_cast<uint32_t>(batchPointsCount));
        uint32_t numPointsTail = numOutputPoints % batchPointsCount;

        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyParams copyParamsNumNeighbors{1, static_cast<uint16_t>(ALIGN_NUM_8 * sizeof(int32_t)), 0, 0};
        DataCopyPad(numNeighborsTensor, numNeighborsGm[(usedCoreNum - blockIdx) * ALIGN_NUM_8], copyParamsNumNeighbors,
            padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);

        uint32_t addrsPtrY = numNeighborsTensor.GetValue(0);
        uint32_t addrsPtrX = addrsPtrY + numPointsY * maxNumNeighbors;
        numOutputPoints += addrsPtrY;
        SetFlag<HardEvent::S_MTE2>(0);
        WaitFlag<HardEvent::S_MTE2>(0);
        DataCopyParams copyParams{1, static_cast<uint16_t>(batchPointsCount * sizeof(int32_t)), 0, 0};
        DataCopyParams outCopyParams{1, static_cast<uint16_t>(batchPointsCount * sizeof(int32_t)), 0, 0};
        DataCopyParams copyParamsTail{1, static_cast<uint16_t>(numPointsTail * sizeof(int32_t)), 0, 0};
        DataCopyParams outCopyParamsTail{1, static_cast<uint16_t>(numPointsTail * sizeof(int32_t)), 0, 0};

        for (uint32_t i = 0; i < numIters; ++i) {
            bool isFullTile = (i < numIters - 1) || (numPointsTail == 0);
            if (isFullTile) {
                DataCopyPad(indexXTensor, outTempWsp[outputWspOffsetX], copyParams, padParams);
                DataCopyPad(indexYTensor, outTempWsp[outputWspOffsetY], copyParams, padParams);
            } else {
                DataCopyPad(indexXTensor, outTempWsp[outputWspOffsetX], copyParamsTail, padParams);
                DataCopyPad(indexYTensor, outTempWsp[outputWspOffsetY], copyParamsTail, padParams);
            }
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            if (isFullTile) {
                DataCopyPad(outFinalGm[addrsPtrX], indexXTensor, outCopyParams);
                DataCopyPad(outFinalGm[addrsPtrY], indexYTensor, outCopyParams);
            } else {
                DataCopyPad(outFinalGm[addrsPtrX], indexXTensor, outCopyParamsTail);
                DataCopyPad(outFinalGm[addrsPtrY], indexYTensor, outCopyParamsTail);
            }
            outputWspOffsetX += batchPointsCount;
            outputWspOffsetY += batchPointsCount;
            addrsPtrX += batchPointsCount;
            addrsPtrY += batchPointsCount;
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
    }

  private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf, yBuf, ptrXBuf, ptrYBuf, numNeighborsBuf, numNeighborsCoreBuf;
    TBuf<TPosition::VECCALC> distBuf, tempBuf, maskBuf, baseIndexBuf, indexXBuf, indexYBuf;

    LocalTensor<int32_t> ptrXLocal, ptrYLocal, baseIndexTensor, indexXTensor, indexYTensor, numNeighborsTensor,
        numNeighborsCoreTensor;
    LocalTensor<float> pointsXLocal, pointsYLocal, distLocal, tempLocal;
    LocalTensor<uint8_t> maskUint8;
    GlobalTensor<float> xGm, yGm;
    GlobalTensor<int32_t> ptrXGm, ptrYGm, outTempWsp, outFinalGm, numNeighborsGm;

    uint32_t blockIdx, batchThisCore, ptrAddrOffset, coordinateDim, maxNumNeighbors, coreOutputOffsetSlots, usedCoreNum;
    uint32_t outputWspOffsetX, outputWspOffsetY;
    int32_t numPointsX, numPointsY, batchPointsCount, coreIndexCount, numOutputPoints;
    float rSquare;
};

extern "C" __global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR outFinal,
    GM_ADDR numTotalNeighbors, GM_ADDR workspace, GM_ADDR tiling) {
#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    KernelRadius op;
    op.Init(x, y, ptrX, ptrY, outFinal, numTotalNeighbors, usrWorkspace, &tiling_data);
    op.Process();
}
