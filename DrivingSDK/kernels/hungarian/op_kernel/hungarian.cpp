/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr float FLOAT_MAX = 3.40282347e+38;

class HungarianKernel {
  public:
    __aicore__ inline HungarianKernel() {}

    __aicore__ inline void Init(
        GM_ADDR cost_matrix, GM_ADDR row_ind, GM_ADDR col_ind, GM_ADDR workspace, HungarianTilingData *tiling_data) {
        numRows_ = tiling_data->numRows;
        numCols_ = tiling_data->numCols;
        transpose_ = tiling_data->transposed;

        rowAlign_ = AlignUp(numRows_, 128);
        colAlign_ = AlignUp(numCols_, 128);
        ubAlign_ = max(rowAlign_, colAlign_);

        int32_t totalBufSize = 40 * 1024;
        int32_t usedBufSize = 24 * ubAlign_;
        singleCopyRows = (totalBufSize - usedBufSize) / AlignUp(numCols_, 8);
        singleCopyRows = min(singleCopyRows, numRows_);
        startRows = -1;
        endRows = 0;

        costGm_.SetGlobalBuffer((__gm__ float *)cost_matrix);
        rowIndGm_.SetGlobalBuffer((__gm__ int32_t *)row_ind);
        colIndGm_.SetGlobalBuffer((__gm__ int32_t *)col_ind);

        pipe.InitBuffer(singleCostBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(MultiCostBuf_, singleCopyRows * AlignUp(numCols_, 8) * sizeof(float));

        pipe.InitBuffer(uBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(vBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(flagRowBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(flagColBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(col4rowBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(row4colBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(shortestPathCostsBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(pathBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(remainingBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(scatterIdxBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(resultBuf_, B32_DATA_NUM_PER_REPEAT * sizeof(float));

        pipe.InitBuffer(tempBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(idxBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(shortestPathCostsTmpBuf_, ubAlign_ * sizeof(float));
        pipe.InitBuffer(pathTmpBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(biasIdxBuf_, ubAlign_ * sizeof(int32_t));
        pipe.InitBuffer(biasBuf_, ubAlign_ * sizeof(float));

        pipe.InitBuffer(sortDstBuf_, 2 * ubAlign_ * sizeof(float));
        pipe.InitBuffer(sortTmpBuf_, 4 * ubAlign_ * sizeof(float));

        pipe.InitBuffer(maskBuf_, B32_DATA_NUM_PER_REPEAT * sizeof(uint32_t));
        pipe.InitBuffer(indexMaskBuf_, B32_DATA_NUM_PER_REPEAT * sizeof(uint32_t));

        pipe.InitBuffer(sharedTmpBuf_, ubAlign_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> uLocal = uBuf_.template Get<float>();
        LocalTensor<float> vLocal = vBuf_.template Get<float>();
        LocalTensor<int32_t> flagRowLocal = flagRowBuf_.template Get<int32_t>();
        LocalTensor<int32_t> flagColLocal = flagColBuf_.template Get<int32_t>();

        LocalTensor<int32_t> col4rowLocal = col4rowBuf_.template Get<int32_t>();
        LocalTensor<int32_t> row4colLocal = row4colBuf_.template Get<int32_t>();
        LocalTensor<float> shortestPathCostsLocal = shortestPathCostsBuf_.template Get<float>();
        LocalTensor<int32_t> pathLocal = pathBuf_.template Get<int32_t>();
        LocalTensor<int32_t> remainingLocal = remainingBuf_.template Get<int32_t>();
        LocalTensor<int32_t> scatterIdxLocal = scatterIdxBuf_.template Get<int32_t>();

        LocalTensor<float> resultLocal = resultBuf_.template Get<float>();

        Duplicate(col4rowLocal, -1, rowAlign_);
        Duplicate(row4colLocal, -1, colAlign_);
        Duplicate(pathLocal, -1, colAlign_);
        Duplicate(uLocal, 0.0f, rowAlign_);
        Duplicate(vLocal, 0.0f, colAlign_);

        for (uint32_t rowIdx = 0; rowIdx < numRows_; rowIdx++) {
            CreateVecIndex(remainingLocal, 0, colAlign_);
            Muls(remainingLocal, remainingLocal, -1, colAlign_);
            Adds(remainingLocal, remainingLocal, numCols_ - 1, colAlign_);
            Adds(scatterIdxLocal, remainingLocal, 0, colAlign_);
            Duplicate(flagRowLocal, 0, rowAlign_);
            Duplicate(flagColLocal, 0, colAlign_);
            Duplicate(shortestPathCostsLocal, FLOAT_MAX, colAlign_);

            int32_t curIdx = rowIdx;
            float minValue = 0;
            int32_t tempSize = numCols_;
            int32_t sink = -1;

            while (sink == -1) {
                LocalTensor<float> costLocal = GetRowCost(curIdx);

                Augmenting(uLocal, vLocal, costLocal, pathLocal, shortestPathCostsLocal, resultLocal, remainingLocal,
                    scatterIdxLocal, curIdx, minValue, tempSize);

                minValue = resultLocal.GetValue(0);
                int32_t index = resultLocal.ReinterpretCast<int32_t>().GetValue(1);

                flagRowLocal.SetValue(curIdx, 1);
                int32_t tempIdx = remainingLocal.GetValue(index);
                if (row4colLocal.GetValue(tempIdx) == -1) {
                    sink = tempIdx;
                } else {
                    curIdx = row4colLocal.GetValue(tempIdx);
                }
                flagColLocal.SetValue(tempIdx, 1);
                tempSize -= 1;

                int32_t swapGatherIdx1 = remainingLocal.GetValue(index);
                int32_t swapGatherIdx2 = remainingLocal.GetValue(tempSize);
                remainingLocal.SetValue(index, swapGatherIdx2);
                remainingLocal.SetValue(tempSize, swapGatherIdx1);

                int32_t swapScatterIdx1 = scatterIdxLocal.GetValue(swapGatherIdx1);
                int32_t swapScatterIdx2 = scatterIdxLocal.GetValue(swapGatherIdx2);
                scatterIdxLocal.SetValue(swapGatherIdx1, swapScatterIdx2);
                scatterIdxLocal.SetValue(swapGatherIdx2, swapScatterIdx1);
            }
            UpdateParam(
                flagRowLocal, flagColLocal, col4rowLocal, shortestPathCostsLocal, uLocal, vLocal, rowIdx, minValue);

            int32_t tempIdx = sink;
            while (1) {
                int32_t nextIdx = pathLocal.GetValue(tempIdx);
                row4colLocal.SetValue(tempIdx, nextIdx);
                int32_t swapIdx = col4rowLocal.GetValue(nextIdx);
                col4rowLocal.SetValue(nextIdx, tempIdx);
                tempIdx = swapIdx;
                if (nextIdx == rowIdx) {
                    break;
                }
            }
        }

        if (transpose_) {
            uint16_t sort32RepeatTimes = rowAlign_ / 32;
            uint16_t sort32MergeRepeatTimes = rowAlign_ / 128;
            uint16_t sortCountList[4] = {(uint16_t)32, (uint16_t)32, (uint16_t)32, (uint16_t)32};

            LocalTensor<float> sortDstLocal = sortDstBuf_.template Get<float>();
            LocalTensor<float> sortTmpLocal = sortTmpBuf_.template Get<float>();

            Duplicate(uLocal, -1 * FLOAT_MAX, rowAlign_);
            Muls(col4rowLocal, col4rowLocal, -1, numRows_);
            Cast(uLocal, col4rowLocal, RoundMode::CAST_RINT, numRows_);
            CreateVecIndex(col4rowLocal, 0, numRows_);
            Sort<float, true>(
                sortDstLocal, uLocal, col4rowLocal.ReinterpretCast<uint32_t>(), sortTmpLocal, sort32RepeatTimes);
            Extract(uLocal, col4rowLocal.ReinterpretCast<uint32_t>(), sortDstLocal, sort32RepeatTimes);
            Cast(flagRowLocal, uLocal, RoundMode::CAST_RINT, numRows_);
            Muls(flagRowLocal, flagRowLocal, -1, numRows_);
        } else {
            CreateVecIndex(flagRowLocal, 0, numRows_);
        }

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyOut(flagRowLocal, col4rowLocal);
    }

  private:
    __aicore__ inline void CopyOut(const LocalTensor<int32_t> &rowIdxLocal, const LocalTensor<int32_t> &colIdxLocal) {
        DataCopyPad(rowIndGm_, rowIdxLocal, {1, static_cast<uint32_t>(numRows_ * sizeof(float)), 0, 0, 0});
        DataCopyPad(colIndGm_, colIdxLocal, {1, static_cast<uint32_t>(numRows_ * sizeof(float)), 0, 0, 0});
    }

    __aicore__ inline LocalTensor<float> GetRowCost(int32_t curIdx) {
        if (curIdx >= endRows) {
            uint16_t copyRows = min(numRows_ - curIdx, singleCopyRows);
            startRows = curIdx;
            endRows = curIdx + copyRows;
            LocalTensor<float> costLocal = MultiCostBuf_.template Get<float>();
            DataCopyPad(costLocal, costGm_[curIdx * numCols_],
                {copyRows, static_cast<uint32_t>(numCols_ * sizeof(float)), 0, 0, 0}, {});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            return costLocal;
        } else if (curIdx < startRows) {
            LocalTensor<float> costLocal = singleCostBuf_.template Get<float>();
            DataCopyPad(costLocal, costGm_[curIdx * numCols_],
                {1, static_cast<uint32_t>(numCols_ * sizeof(float)), 0, 0, 0}, {});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            return costLocal;
        } else {
            LocalTensor<float> costLocal = MultiCostBuf_.template Get<float>();
            return costLocal[(curIdx - startRows) * AlignUp(numCols_, 8)];
        }
    }

    __aicore__ inline void Augmenting(const LocalTensor<float> &uLocal, const LocalTensor<float> &vLocal,
        const LocalTensor<float> &costLocal, const LocalTensor<int32_t> &path,
        const LocalTensor<float> &shortestPathCosts, const LocalTensor<float> &minValLocal,
        const LocalTensor<int32_t> &remainingLocal, const LocalTensor<int32_t> &scatterIdxLocal, int32_t rowIdx,
        float minVal, int32_t compSize) {
        uint64_t rsvdCnt;
        uint32_t compSizeAlign = AlignUp(compSize, B32_DATA_NUM_PER_REPEAT);

        LocalTensor<float> tempLocal = tempBuf_.template Get<float>();
        LocalTensor<int32_t> idxLocal = idxBuf_.template Get<int32_t>();
        LocalTensor<float> shortestPathCostsTmp = shortestPathCostsTmpBuf_.template Get<float>();
        LocalTensor<int32_t> pathTmp = pathTmpBuf_.template Get<int32_t>();
        LocalTensor<float> sharedTmpLocal = sharedTmpBuf_.template Get<float>();

        LocalTensor<uint8_t> maskLocal = maskBuf_.template Get<uint8_t>();
        LocalTensor<uint8_t> indexMaskLocal = indexMaskBuf_.template Get<uint8_t>();

        Sub(tempLocal, costLocal, vLocal, colAlign_);
        Adds(tempLocal, tempLocal, minVal - uLocal.GetValue(rowIdx), colAlign_);
        Compare(maskLocal, tempLocal, shortestPathCosts, CMPMODE::LT, colAlign_);
        Cast(idxLocal.ReinterpretCast<float>(), scatterIdxLocal, RoundMode::CAST_RINT, colAlign_);
        CompareScalar(indexMaskLocal, idxLocal.ReinterpretCast<float>(), (float)compSize, CMPMODE::LT, colAlign_);
        And(maskLocal.ReinterpretCast<uint16_t>(), maskLocal.ReinterpretCast<uint16_t>(),
            indexMaskLocal.ReinterpretCast<uint16_t>(), 128);
        Select(shortestPathCosts, maskLocal, tempLocal, shortestPathCosts, SELMODE::VSEL_TENSOR_TENSOR_MODE, colAlign_);
        Duplicate(pathTmp, rowIdx, colAlign_);
        Select(path.ReinterpretCast<float>(), maskLocal, pathTmp.ReinterpretCast<float>(),
            path.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, colAlign_);

        Muls(idxLocal, remainingLocal, int32_t(sizeof(float)), compSize);
        Gather(shortestPathCostsTmp, shortestPathCosts, idxLocal.ReinterpretCast<uint32_t>(), 0, compSize);
        ReduceMin(minValLocal, shortestPathCostsTmp, sharedTmpLocal, compSize, true);
    }

    __aicore__ inline void UpdateParam(const LocalTensor<int32_t> flagRowLocal, const LocalTensor<int32_t> flagColLocal,
        const LocalTensor<int32_t> col4rowLocal, const LocalTensor<float> shortestPathCosts,
        const LocalTensor<float> uLocal, const LocalTensor<float> vLocal, int32_t rowIdx, float minVal) {
        LocalTensor<int32_t> idxLocal = idxBuf_.template Get<int32_t>();
        LocalTensor<uint8_t> maskLocal = maskBuf_.template Get<uint8_t>();
        LocalTensor<uint8_t> indexMaskLocal = indexMaskBuf_.template Get<uint8_t>();
        LocalTensor<int32_t> biasIdxLocal = biasIdxBuf_.template Get<int32_t>();
        LocalTensor<float> biasLocal = biasBuf_.template Get<float>();

        CreateVecIndex(idxLocal, 0, numRows_);
        CompareScalar(maskLocal, idxLocal, rowIdx, CMPMODE::EQ, rowAlign_);
        Not(indexMaskLocal.ReinterpretCast<uint16_t>(), maskLocal.ReinterpretCast<uint16_t>(), 128);
        CompareScalar(maskLocal, flagRowLocal, 1, CMPMODE::EQ, rowAlign_);
        And(indexMaskLocal, indexMaskLocal, maskLocal, 256);

        Select(biasIdxLocal.ReinterpretCast<float>(), indexMaskLocal, col4rowLocal.ReinterpretCast<float>(), 0.0f,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, numRows_);
        Muls(biasIdxLocal, biasIdxLocal, int32_t(sizeof(float)), numRows_);
        Gather(biasLocal, shortestPathCosts, biasIdxLocal.ReinterpretCast<uint32_t>(), 0, numRows_);
        Select(biasLocal, indexMaskLocal, biasLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, numRows_);
        Sub(uLocal, uLocal, biasLocal, numRows_);
        Duplicate(biasLocal, minVal, numRows_);
        Select(biasLocal, maskLocal, biasLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, numRows_);
        Add(uLocal, uLocal, biasLocal, numRows_);

        CompareScalar(maskLocal, flagColLocal, 1, CMPMODE::EQ, colAlign_);
        Duplicate(biasLocal, minVal, numCols_);
        Select(biasLocal, maskLocal, biasLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, numCols_);
        Sub(vLocal, vLocal, biasLocal, numCols_);
        Select(shortestPathCosts, maskLocal, shortestPathCosts, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, numCols_);
        Add(vLocal, vLocal, shortestPathCosts, numCols_);
    }

  private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> uBuf_, vBuf_;
    TBuf<TPosition::VECCALC> flagRowBuf_, flagColBuf_;
    TBuf<TPosition::VECCALC> col4rowBuf_, row4colBuf_;
    TBuf<TPosition::VECCALC> shortestPathCostsBuf_, pathBuf_;
    TBuf<TPosition::VECCALC> remainingBuf_, scatterIdxBuf_;
    TBuf<TPosition::VECCALC> resultBuf_;
    TBuf<TPosition::VECCALC> singleCostBuf_, MultiCostBuf_;
    TBuf<TPosition::VECCALC> tempBuf_, idxBuf_;
    TBuf<TPosition::VECCALC> maskBuf_, indexMaskBuf_;
    TBuf<TPosition::VECCALC> shortestPathCostsTmpBuf_, pathTmpBuf_;
    TBuf<TPosition::VECCALC> biasIdxBuf_, biasBuf_;
    TBuf<TPosition::VECCALC> sortDstBuf_, sortTmpBuf_;
    TBuf<TPosition::VECCALC> sharedTmpBuf_;

    GlobalTensor<float> costGm_;
    GlobalTensor<int32_t> rowIndGm_;
    GlobalTensor<int32_t> colIndGm_;

    int32_t numRows_, rowAlign_, numCols_, colAlign_, ubAlign_;
    int32_t singleCopyRows, startRows, endRows, costBlockStride;
    bool transpose_;
};

extern "C" __global__ __aicore__ void hungarian(
    GM_ADDR cost_matrix, GM_ADDR row_ind, GM_ADDR col_ind, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    HungarianKernel op;
    op.Init(cost_matrix, row_ind, col_ind, workspace, &tiling_data);
    op.Process();
}
