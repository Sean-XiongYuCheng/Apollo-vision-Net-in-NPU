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
        costGm_.SetGlobalBuffer((__gm__ float *)cost_matrix);

        rowAlign_ = AlignUp(numRows_, 64);
        colAlign_ = AlignUp(numCols_, 64);

        rowIndGm_.SetGlobalBuffer((__gm__ int32_t *)row_ind);
        colIndGm_.SetGlobalBuffer((__gm__ int32_t *)col_ind);

        pipe.InitBuffer(uBuf_, rowAlign_ * sizeof(float));
        pipe.InitBuffer(vBuf_, colAlign_ * sizeof(float));
        pipe.InitBuffer(flagRowBuf_, rowAlign_ * sizeof(int32_t));
        pipe.InitBuffer(flagColBuf_, colAlign_ * sizeof(int32_t));
        pipe.InitBuffer(col4rowBuf_, rowAlign_ * sizeof(int32_t));
        pipe.InitBuffer(row4colBuf_, colAlign_ * sizeof(int32_t));
        pipe.InitBuffer(shortestPathCostsBuf_, colAlign_ * sizeof(float));
        pipe.InitBuffer(pathBuf_, colAlign_ * sizeof(int32_t));
        pipe.InitBuffer(remainingBuf_, colAlign_ * sizeof(int32_t));
        pipe.InitBuffer(costBuf_, numCols_ * sizeof(float));
        pipe.InitBuffer(indexBuf_, B32_DATA_NUM_PER_REPEAT * sizeof(int32_t));
        pipe.InitBuffer(resultBuf_, B32_DATA_NUM_PER_REPEAT * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> costLocal = costBuf_.template Get<float>();
        LocalTensor<float> uLocal = uBuf_.template Get<float>();
        LocalTensor<float> vLocal = vBuf_.template Get<float>();
        LocalTensor<int32_t> flagRowLocal = flagRowBuf_.template Get<int32_t>();
        LocalTensor<int32_t> flagColLocal = flagColBuf_.template Get<int32_t>();

        LocalTensor<int32_t> col4rowLocal = col4rowBuf_.template Get<int32_t>();
        LocalTensor<int32_t> row4colLocal = row4colBuf_.template Get<int32_t>();
        LocalTensor<float> shortestPathCostsLocal = shortestPathCostsBuf_.template Get<float>();
        LocalTensor<int32_t> pathLocal = pathBuf_.template Get<int32_t>();
        LocalTensor<int32_t> remainingLocal = remainingBuf_.template Get<int32_t>();

        LocalTensor<int32_t> indexLocal = indexBuf_.template Get<int32_t>();
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
            Duplicate(flagRowLocal, 0, rowAlign_);
            Duplicate(flagColLocal, 0, colAlign_);
            Duplicate(shortestPathCostsLocal, FLOAT_MAX, colAlign_);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);

            int32_t curIdx = rowIdx;
            float minValue = 0;
            int32_t tempSize = numCols_;
            int32_t sink = -1;

            while (sink == -1) {
                DataCopyPad(costLocal, costGm_[curIdx * numCols_],
                    {1, static_cast<uint32_t>(numCols_ * sizeof(float)), 0, 0, 0}, {});
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                AugmentingVF(uLocal, vLocal, costLocal, pathLocal, shortestPathCostsLocal, resultLocal, indexLocal,
                    remainingLocal.ReinterpretCast<uint32_t>(), row4colLocal, curIdx, minValue, tempSize);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
                minValue = resultLocal.GetValue(0);
                int32_t index = indexLocal.GetValue(0);
                flagRowLocal.SetValue(curIdx, 1);
                int32_t tempIdx = remainingLocal.GetValue(index);
                if (row4colLocal.GetValue(tempIdx) == -1) {
                    sink = tempIdx;
                } else {
                    curIdx = row4colLocal.GetValue(tempIdx);
                }
                flagColLocal.SetValue(tempIdx, 1);
                tempSize -= 1;
                remainingLocal.SetValue(index, remainingLocal.GetValue(tempSize));
            }
            UpdateParamVF(flagRowLocal, flagColLocal, col4rowLocal.ReinterpretCast<uint32_t>(), shortestPathCostsLocal,
                uLocal, vLocal, rowIdx, minValue);
            PipeBarrier<PIPE_ALL>();
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
            PipeBarrier<PIPE_ALL>();
        }

        if (transpose_) {
            static constexpr AscendC::SortConfig config = {AscendC::SortType::RADIX_SORT, false};
            LocalTensor<int32_t> sortVal = flagRowLocal.ReinterpretCast<int32_t>();
            LocalTensor<uint32_t> sortIdx = col4rowLocal.ReinterpretCast<uint32_t>();
            Sort<int32_t, false, config>(sortVal, sortIdx, col4rowLocal, numRows_);
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

    __aicore__ inline void AugmentingVF(const LocalTensor<float> &uLocal, const LocalTensor<float> &vLocal,
        const LocalTensor<float> &costLocal, const LocalTensor<int32_t> &path,
        const LocalTensor<float> &shortestPathCosts, const LocalTensor<float> &minValLocal,
        const LocalTensor<int32_t> minIdxLocal, const LocalTensor<uint32_t> &remainingLocal,
        const LocalTensor<int32_t> row4colLocal, uint32_t rowIdx, float minVal, uint32_t compSize) {
        __local_mem__ float *uLocalPtr = (__local_mem__ float *)uLocal.GetPhyAddr();
        __local_mem__ float *vLocalPtr = (__local_mem__ float *)vLocal.GetPhyAddr();
        __local_mem__ float *costLocalPtr = (__local_mem__ float *)costLocal.GetPhyAddr();
        __local_mem__ float *shortestPathCostsPtr = (__local_mem__ float *)shortestPathCosts.GetPhyAddr();
        __local_mem__ int32_t *pathPtr = (__local_mem__ int32_t *)path.GetPhyAddr();
        __local_mem__ uint32_t *remainingPtr = (__local_mem__ uint32_t *)remainingLocal.GetPhyAddr();
        __local_mem__ uint32_t *row4colPtr = (__local_mem__ uint32_t *)row4colLocal.GetPhyAddr();

        __local_mem__ float *minValLocalPtr = (__local_mem__ float *)minValLocal.GetPhyAddr();
        __local_mem__ int32_t *minIdxLocalPtr = (__local_mem__ int32_t *)minIdxLocal.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<float> uReg, vReg, costReg, shortestPathCostsReg, minReg;
            MicroAPI::RegTensor<int32_t> pathReg, indexReg, arangeReg, tempReg, row4colReg;
            MicroAPI::RegTensor<uint32_t> remainingReg;
            MicroAPI::RegTensor<float> minValReg, minValInReg;
            MicroAPI::RegTensor<int32_t> minIdxReg;
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loops = DivCeil(compSize, B32_DATA_NUM_PER_REPEAT);
            uint32_t totalNum = compSize;

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(uReg, uLocalPtr + rowIdx);
            MicroAPI::Duplicate(minValReg, FLOAT_MAX, mask);

            for (uint16_t i = 0; i < loops; ++i) {
                int32_t offset = i * B32_DATA_NUM_PER_REPEAT;
                MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float>(totalNum);
                MicroAPI::DataCopy(remainingReg, remainingPtr + offset);

                MicroAPI::Gather(pathReg, pathPtr, remainingReg, validMask);
                MicroAPI::Gather(shortestPathCostsReg, shortestPathCostsPtr, remainingReg, validMask);
                MicroAPI::Gather(vReg, vLocalPtr, remainingReg, validMask);
                MicroAPI::Gather(costReg, costLocalPtr, remainingReg, validMask);

                MicroAPI::Duplicate(minValInReg, minVal, validMask);
                MicroAPI::Sub(costReg, costReg, vReg, validMask);
                MicroAPI::Sub(minValInReg, minValInReg, uReg, validMask);
                MicroAPI::Add(costReg, costReg, minValInReg, validMask);

                MicroAPI::MaskReg cmpMask;
                MicroAPI::Duplicate(tempReg, rowIdx, validMask);
                MicroAPI::Compare<float, CMPMODE::LT>(cmpMask, costReg, shortestPathCostsReg, validMask);
                MicroAPI::Select(shortestPathCostsReg, costReg, shortestPathCostsReg, cmpMask);
                MicroAPI::Select(pathReg, tempReg, pathReg, cmpMask);

                MicroAPI::Scatter(shortestPathCostsPtr, shortestPathCostsReg, remainingReg, validMask);
                MicroAPI::Scatter(pathPtr, pathReg, remainingReg, validMask);

                MicroAPI::MaskReg minMask;
                MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(minReg, shortestPathCostsReg, validMask);
                MicroAPI::DeInterleave(minReg, (MicroAPI::RegTensor<float> &)indexReg, minReg, minReg);
                MicroAPI::Adds(indexReg, indexReg, offset, mask);
                MicroAPI::Compare<float, CMPMODE::LT>(minMask, minReg, minValReg, mask);
                MicroAPI::Select(minValReg, minReg, minValReg, minMask);
                MicroAPI::Select(minIdxReg, indexReg, minIdxReg, minMask);
            }
            MicroAPI::UnalignReg uValReg, uIdxReg;
            MicroAPI::DataCopyUnAlign((__local_mem__ float *&)minValLocalPtr, minValReg, uValReg, 1);
            MicroAPI::DataCopyUnAlign((__local_mem__ int32_t *&)minIdxLocalPtr, minIdxReg, uIdxReg, 1);
            MicroAPI::DataCopyUnAlignPost((__local_mem__ float *&)minValLocalPtr, uValReg, 0);
            MicroAPI::DataCopyUnAlignPost((__local_mem__ int32_t *&)minIdxLocalPtr, uIdxReg, 0);
        }
    }

    __aicore__ inline void UpdateParamVF(const LocalTensor<int32_t> flagRowLocal,
        const LocalTensor<int32_t> flagColLocal, const LocalTensor<uint32_t> col4rowLocal,
        const LocalTensor<float> shortestPathCosts, const LocalTensor<float> uLocal, const LocalTensor<float> vLocal,
        uint32_t rowIdx, float minVal) {
        __local_mem__ int32_t *flagRowPtr = (__local_mem__ int32_t *)flagRowLocal.GetPhyAddr();
        __local_mem__ int32_t *flagColPtr = (__local_mem__ int32_t *)flagColLocal.GetPhyAddr();
        __local_mem__ uint32_t *col4rowPtr = (__local_mem__ uint32_t *)col4rowLocal.GetPhyAddr();
        __local_mem__ float *uLocalPtr = (__local_mem__ float *)uLocal.GetPhyAddr();
        __local_mem__ float *vLocalPtr = (__local_mem__ float *)vLocal.GetPhyAddr();
        __local_mem__ float *shortestPathCostsPtr = (__local_mem__ float *)shortestPathCosts.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<float> uReg, vReg, shortestPathCostsReg, biasReg, temp1Reg, temp2Reg;
            MicroAPI::RegTensor<int32_t> indexReg, flagRowReg, flagColReg;
            MicroAPI::RegTensor<uint32_t> col4rowReg;
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopsRow = DivCeil(numRows_, B32_DATA_NUM_PER_REPEAT);
            uint16_t loopsCol = DivCeil(numCols_, B32_DATA_NUM_PER_REPEAT);

            uint32_t totalRows = numRows_;
            uint32_t totalCols = numCols_;

            for (uint16_t i = 0; i < loopsRow; i++) {
                int32_t offset = i * B32_DATA_NUM_PER_REPEAT;
                MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float>(totalRows);

                MicroAPI::DataCopy(uReg, uLocalPtr + offset);
                MicroAPI::DataCopy(col4rowReg, col4rowPtr + offset);
                MicroAPI::DataCopy(flagRowReg, flagRowPtr + offset);

                MicroAPI::MaskReg filterMask, indexMask;
                MicroAPI::Arange(indexReg, offset);
                MicroAPI::Compares<int32_t, CMPMODE::EQ>(filterMask, flagRowReg, 1, mask);
                MicroAPI::And(filterMask, filterMask, validMask, mask);
                MicroAPI::Compares<int32_t, CMPMODE::NE>(indexMask, indexReg, rowIdx, mask);
                MicroAPI::And(indexMask, indexMask, filterMask, mask);
                MicroAPI::Gather(biasReg, shortestPathCostsPtr, col4rowReg, indexMask);
                MicroAPI::Duplicate(temp1Reg, minVal, filterMask);
                MicroAPI::Sub(temp1Reg, temp1Reg, biasReg, mask);
                MicroAPI::Add(uReg, uReg, temp1Reg, mask);

                MicroAPI::DataCopy(uLocalPtr + offset, uReg, validMask);
            }

            for (uint16_t i = 0; i < loopsCol; i++) {
                int32_t offset = i * B32_DATA_NUM_PER_REPEAT;
                MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float>(totalCols);

                MicroAPI::DataCopy(vReg, vLocalPtr + offset);
                MicroAPI::DataCopy(flagColReg, flagColPtr + offset);
                MicroAPI::DataCopy(shortestPathCostsReg, shortestPathCostsPtr + offset);

                MicroAPI::MaskReg filterMask;
                MicroAPI::Compares<int32_t, CMPMODE::EQ>(filterMask, flagColReg, 1, mask);
                MicroAPI::And(filterMask, filterMask, validMask, mask);
                MicroAPI::Duplicate(temp2Reg, minVal, filterMask);
                MicroAPI::Sub(temp2Reg, temp2Reg, shortestPathCostsReg, filterMask);
                MicroAPI::Sub(vReg, vReg, temp2Reg, mask);

                MicroAPI::DataCopy(vLocalPtr + offset, vReg, validMask);
            }
        }
    }

  private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> uBuf_, vBuf_;
    TBuf<TPosition::VECCALC> flagRowBuf_, flagColBuf_;
    TBuf<TPosition::VECCALC> col4rowBuf_, row4colBuf_;
    TBuf<TPosition::VECCALC> shortestPathCostsBuf_;
    TBuf<TPosition::VECCALC> pathBuf_;
    TBuf<TPosition::VECCALC> remainingBuf_;
    TBuf<TPosition::VECCALC> indexBuf_, resultBuf_;
    TBuf<TPosition::VECCALC> costBuf_;

    GlobalTensor<float> costGm_;
    GlobalTensor<int32_t> rowIndGm_;
    GlobalTensor<int32_t> colIndGm_;

    uint32_t numRows_, rowAlign_, numCols_, colAlign_;
    bool transpose_;
};

extern "C" __global__ __aicore__ void hungarian(
    GM_ADDR cost_matrix, GM_ADDR row_ind, GM_ADDR col_ind, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    HungarianKernel op;
    op.Init(cost_matrix, row_ind, col_ind, workspace, &tiling_data);
    op.Process();
}
