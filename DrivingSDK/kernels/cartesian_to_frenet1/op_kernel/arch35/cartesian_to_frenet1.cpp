#include "kernel_operator.h"
using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t PT_BYTE_SIZE = sizeof(float);
constexpr uint32_t IDX_BYTE_SIZE = sizeof(int32_t);
constexpr uint32_t DATA_BLOCK_SIZE = 32;
constexpr uint32_t PT_ALIGN_NUM = DATA_BLOCK_SIZE / PT_BYTE_SIZE;
class CartesianToFrenet1 {
  public:
    __aicore__ inline CartesianToFrenet1() {}
    __aicore__ inline void Init(const GM_ADDR dist_vec, const GM_ADDR min_idx, const GM_ADDR back_idx,
        const CartesianToFrenet1TilingData *tiling) {
        InitTiling(tiling);
        InitGM(dist_vec, min_idx, back_idx);
        InitUB();
        InitEvent();
    }
    __aicore__ inline void Process() {
        if (coreId < usedCoreNum) {
            tileNum = tileRemainder > 0 ? tileNum + 1 : tileNum;
            for (int32_t i = 0; i < tileNum; i++) {
                if (i == tileNum - 1 && tileRemainder > 0) {
                    tileTaskNum = tileRemainder;
                }
                SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
                WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

                CopyIn(i, tileTaskNum, taskSize);

                SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

                SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
                WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);

                Compute(tileTaskNum, numPolyLinePoints);

                SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
                WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

                CopyOut(i, tileTaskNum);
            }
        }
    }

  private:
    __aicore__ inline void InitTiling(const CartesianToFrenet1TilingData *tiling) {
        this->copyInAlignNum = tiling->copyInAlignNum;
        this->dstStride = tiling->dstStride;
        this->rightPadding = tiling->rightPadding;
        this->numPolyLinePoints = tiling->numPolyLinePoints;
        this->taskSize = tiling->taskSize;
        this->taskSizeElem = tiling->taskSizeElem;
        this->taskSizeAligned = tiling->taskSizeAligned;
        this->bigCoreCount = tiling->bigCoreCount;
        this->usedCoreNum = tiling->usedCoreNum;
        this->avgTaskNum = tiling->avgTaskNum;
        this->coreId = GetBlockIdx();

        this->tileNum = coreId < bigCoreCount ? tiling->formerTileNum : tiling->tailTileNum;
        this->tileRemainder = coreId < bigCoreCount ? tiling->formerTileRemainder : tiling->tailTileRemainder;
        this->taskResultSizeAligned =
            coreId < bigCoreCount ? tiling->taskResultSizeAligned_b : tiling->taskResultSizeAligned_s;
        this->numTaskCurCore = coreId < bigCoreCount ? tiling->numTaskCurCore_b : tiling->avgTaskNum;
        this->taskLengthCurCore = coreId < bigCoreCount ? tiling->taskLengthCurCore_b : tiling->taskLengthCurCore_s;
        this->tileTaskNum = coreId < bigCoreCount ? tiling->tileTaskNum_b : tiling->tileTaskNum_s;

        this->formerTileTaskNum = this->tileTaskNum;
        this->tileLength = tileTaskNum * taskSizeAligned;

        if (coreId < bigCoreCount) {
            startTaskId = taskLengthCurCore * coreId;
            startResId = numTaskCurCore * coreId;
        } else {
            startTaskId = (avgTaskNum + 1) * taskSizeElem * bigCoreCount +
                numTaskCurCore * taskSizeElem * (coreId - bigCoreCount);
            startResId = (avgTaskNum + 1) * bigCoreCount + avgTaskNum * (coreId - bigCoreCount);
        }
    }
    __aicore__ inline void InitEvent() {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        eventVMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    }
    __aicore__ inline void InitGM(const GM_ADDR distVec, const GM_ADDR minIdx, const GM_ADDR backIdx) {
        this->distVecGm.SetGlobalBuffer((__gm__ float *)distVec + startTaskId, taskLengthCurCore);
        this->minIdxGm.SetGlobalBuffer((__gm__ int32_t *)minIdx + startResId, numTaskCurCore);
        this->backIdxGm.SetGlobalBuffer((__gm__ int32_t *)backIdx + startResId, numTaskCurCore);
    }
    __aicore__ inline void InitUB() {
        uint32_t taskResultSizeAligned64 =
            DivCeil(taskResultSizeAligned, (uint32_t)256) * 256; // 确保最后一次 256 字节读取不越界
        uint32_t tileLengthPadded = DivCeil(tileLength, (uint32_t)512) * 512; // 确保最后一次 128 元素读取不越界。
        uint32_t numPolyLinePointsAlignedEach = AlignUp(numPolyLinePoints, PT_ALIGN_NUM);
        uint32_t xyBufferSize =
            tileTaskNum * numPolyLinePointsAlignedEach * PT_BYTE_SIZE + 256; // 保证最后一个 task 的 64 元素读取不越界

        pipe.InitBuffer(inQueueDistVec, tileLengthPadded);
        pipe.InitBuffer(outQueueMinIdx, taskResultSizeAligned64);
        pipe.InitBuffer(outQueueBackIdx, taskResultSizeAligned64);
        pipe.InitBuffer(xBuffer, xyBufferSize);

        minIdxLocal = outQueueMinIdx.Get<int32_t>();
        backIdxLocal = outQueueBackIdx.Get<int32_t>();
        distVecLocal = inQueueDistVec.Get<float>();
        xLocal = xBuffer.Get<float>();
    }
    __aicore__ inline void CopyIn(int32_t curTile, uint32_t tileTaskNum, uint32_t taskSize) {
        DataCopyPad(distVecLocal, distVecGm[curTile * formerTileTaskNum * taskSizeElem],
            {(uint16_t)tileTaskNum, (uint32_t)taskSize, 0, dstStride, 0},
            {true, 0, static_cast<uint8_t>(rightPadding), 0});
    }
    __aicore__ inline void Compute(uint32_t tileTaskNum, uint32_t numPolyLinePoints) {
        uint32_t numPolyLinePointsDimAligned = tileTaskNum * copyInAlignNum / PT_BYTE_SIZE;
        uint32_t numPolyLinePointsAlignedEach = AlignUp(numPolyLinePoints, PT_ALIGN_NUM);

        constexpr uint16_t B32_DATA_NUM_PER_REPEAT = 64;
        constexpr uint16_t DEINTERLEAVE_INPUT_SIZE = 128;

        //  平方 → DeInterleave → Add → Sqrt → 写距离到 xLocal
        {
            __local_mem__ float *distVecPtr = (__local_mem__ float *)distVecLocal.GetPhyAddr();
            __local_mem__ float *xPtr = (__local_mem__ float *)xLocal.GetPhyAddr();
            uint16_t numDeInterleave =
                static_cast<uint16_t>(DivCeil(numPolyLinePointsDimAligned, DEINTERLEAVE_INPUT_SIZE));

            PipeBarrier<PIPE_ALL>();
            __VEC_SCOPE__ {
                MicroAPI::RegTensor<float> srcReg1, srcReg2, xReg, yReg;
                MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

                for (uint16_t i = 0; i < numDeInterleave; i++) {
                    // 加载 128 个交错元素 [x0,y0,x1,y1,...,x63,y63]
                    MicroAPI::DataCopy(srcReg1, distVecPtr + i * DEINTERLEAVE_INPUT_SIZE);
                    MicroAPI::DataCopy(srcReg2, distVecPtr + i * DEINTERLEAVE_INPUT_SIZE + B32_DATA_NUM_PER_REPEAT);

                    // 步骤1: 逐元素平方 (在寄存器内完成, 不写回 UB)
                    MicroAPI::Mul(srcReg1, srcReg1, srcReg1, mask);
                    MicroAPI::Mul(srcReg2, srcReg2, srcReg2, mask);

                    // 步骤2: DeInterleave 分离 x²/y² (在寄存器内完成)
                    MicroAPI::DeInterleave(xReg, yReg, srcReg1, srcReg2);

                    // 步骤3: Add(x²+y²) + Sqrt → 欧氏距离 (在寄存器内完成)
                    MicroAPI::Add(xReg, xReg, yReg, mask);
                    MicroAPI::Sqrt(xReg, xReg, mask);

                    // 只写一次: 距离结果到 xLocal
                    MicroAPI::DataCopy(xPtr + i * B32_DATA_NUM_PER_REPEAT, xReg, mask);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }

        // Reduce<MIN> 求最小值及索引
        {
            __local_mem__ float *xPtr = (__local_mem__ float *)xLocal.GetPhyAddr();
            __local_mem__ int32_t *minIdxPtr = (__local_mem__ int32_t *)minIdxLocal.GetPhyAddr();

            uint16_t reduceLoops = static_cast<uint16_t>(DivCeil(numPolyLinePoints, B32_DATA_NUM_PER_REPEAT));
            uint16_t tileTaskNum16 = static_cast<uint16_t>(tileTaskNum);

            __VEC_SCOPE__ {
                MicroAPI::RegTensor<float> srcReg, reduceDstReg;
                MicroAPI::RegTensor<float> curReg, curValReg;
                MicroAPI::RegTensor<float> bestValReg;
                MicroAPI::RegTensor<int32_t> curIdxReg, bestIdxReg;
                MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
                MicroAPI::MaskReg allMask1 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
                MicroAPI::MaskReg cmpMask;
                uint32_t oneCountLocal = 1;
                MicroAPI::MaskReg oneMask = MicroAPI::UpdateMask<int32_t>(oneCountLocal);
                MicroAPI::UnalignReg uIdxReg;

                for (uint16_t i = 0; i < tileTaskNum16; i++) {
                    __local_mem__ float *taskPtr = xPtr + i * numPolyLinePointsAlignedEach;

                    // 首块归约
                    MicroAPI::DataCopy(srcReg, taskPtr);
                    uint32_t taskCount = numPolyLinePoints;
                    MicroAPI::MaskReg firstMask = MicroAPI::UpdateMask<int32_t>(taskCount);
                    MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(reduceDstReg, srcReg, firstMask);
                    MicroAPI::DeInterleave(
                        bestValReg, (MicroAPI::RegTensor<float> &)bestIdxReg, reduceDstReg, reduceDstReg);

                    // 后续块 (reduceLoops == 1 时此循环不执行)
                    for (uint16_t j = 1; j < reduceLoops; j++) {
                        int32_t chunkStart = j * B32_DATA_NUM_PER_REPEAT;
                        uint32_t remaining = numPolyLinePoints - chunkStart;

                        MicroAPI::DataCopy(srcReg, taskPtr + chunkStart);
                        MicroAPI::MaskReg chunkMask = MicroAPI::UpdateMask<int32_t>(remaining);
                        MicroAPI::Reduce<MicroAPI::ReduceType::MIN>(curReg, srcReg, chunkMask);
                        MicroAPI::DeInterleave(curValReg, (MicroAPI::RegTensor<float> &)curIdxReg, curReg, curReg);
                        MicroAPI::Adds(curIdxReg, curIdxReg, chunkStart, allMask1);
                        MicroAPI::Compare<float, CMPMODE::LT>(cmpMask, curValReg, bestValReg, allMask);
                        MicroAPI::Select(bestValReg, curValReg, bestValReg, cmpMask);
                        MicroAPI::Select(bestIdxReg, curIdxReg, bestIdxReg, cmpMask);
                    }

                    MicroAPI::DataCopyUnAlign((__local_mem__ int32_t *&)minIdxPtr, bestIdxReg, uIdxReg, 1);
                }
                MicroAPI::DataCopyUnAlignPost((__local_mem__ int32_t *&)minIdxPtr, uIdxReg, 0);
            }
        }

        // back_idx = max(min_idx - 1, 0)
        {
            __local_mem__ int32_t *minIdxPtr = (__local_mem__ int32_t *)minIdxLocal.GetPhyAddr();
            __local_mem__ int32_t *backIdxPtr = (__local_mem__ int32_t *)backIdxLocal.GetPhyAddr();
            uint16_t repeatTimes = static_cast<uint16_t>(DivCeil(tileTaskNum, B32_DATA_NUM_PER_REPEAT));

            __VEC_SCOPE__ {
                MicroAPI::RegTensor<int32_t> minIdxReg;
                MicroAPI::RegTensor<int32_t> onesReg;
                MicroAPI::RegTensor<int32_t> backIdxReg;
                MicroAPI::RegTensor<int32_t> zeroReg;
                MicroAPI::MaskReg mask = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();

                MicroAPI::Duplicate(onesReg, (int32_t)1, mask);
                MicroAPI::Duplicate(zeroReg, (int32_t)0, mask);

                for (uint16_t i = 0; i < repeatTimes; i++) {
                    MicroAPI::DataCopy(minIdxReg, minIdxPtr + i * B32_DATA_NUM_PER_REPEAT);
                    MicroAPI::Sub(backIdxReg, minIdxReg, onesReg, mask);
                    MicroAPI::Max(backIdxReg, backIdxReg, zeroReg, mask);
                    MicroAPI::DataCopy(backIdxPtr + i * B32_DATA_NUM_PER_REPEAT, backIdxReg, mask);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void CopyOut(uint32_t curTile, uint32_t tileTaskNum) {
        DataCopyPad(minIdxGm[curTile * formerTileTaskNum], minIdxLocal,
            {1, static_cast<uint32_t>(tileTaskNum * IDX_BYTE_SIZE), 0, 0, 0});
        DataCopyPad(backIdxGm[curTile * formerTileTaskNum], backIdxLocal,
            {1, static_cast<uint32_t>(tileTaskNum * IDX_BYTE_SIZE), 0, 0, 0});
    }

  private:
    TPipe pipe;
    GlobalTensor<float> distVecGm;
    GlobalTensor<int32_t> minIdxGm;
    GlobalTensor<int32_t> backIdxGm;
    LocalTensor<int32_t> minIdxLocal, backIdxLocal;
    LocalTensor<float> distVecLocal, xLocal;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_, eventVMTE2_;
    uint32_t startTaskId, startResId;
    uint32_t copyInAlignNum, dstStride, rightPadding, numPolyLinePoints, taskSize, taskSizeElem, taskSizeAligned,
        bigCoreCount, usedCoreNum, avgTaskNum, coreId, tileNum, tileRemainder, taskResultSizeAligned, numTaskCurCore,
        taskLengthCurCore, tileTaskNum, formerTileTaskNum, tileLength;
    TBuf<TPosition::VECCALC> xBuffer, outQueueMinIdx, outQueueBackIdx, inQueueDistVec;
};
extern "C" __global__ __aicore__ void cartesian_to_frenet1(
    GM_ADDR distVec, GM_ADDR minIdx, GM_ADDR backIdx, GM_ADDR workspace, GM_ADDR tiling_data) {
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CartesianToFrenet1 op;
    op.Init(distVec, minIdx, backIdx, &tiling);
    op.Process();
}
