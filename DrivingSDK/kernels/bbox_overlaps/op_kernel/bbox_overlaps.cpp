/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

constexpr int32_t BYTE_ALIGN_SIZE = 32;
constexpr uint32_t MODEFLAG_IOU = 0;
constexpr uint32_t MODEFLAG_IOF = 1;
constexpr uint32_t MODEFLAG_GIOU = 2;

template <typename T, int N> class KernelBboxOverlaps {
  public:
    __aicore__ inline KernelBboxOverlaps() {}

    __aicore__ inline void Init(
        GM_ADDR boxes_a, GM_ADDR boxes_b, GM_ADDR res, const BboxOverlapsTilingData *tilingData, TPipe *tmpPipe) {
        pipe_ = tmpPipe;
        blkIdx_ = GetBlockIdx();
        auto ubSize = tilingData->availableUBSize;
        modeFlag_ = tilingData->modeFlag;
        margin_ = tilingData->margin;
        nSize_ = tilingData->nSize;

        auto totalBufferSize = (2 * (8 + 1) + 6) * sizeof(T);
        if (modeFlag_ == MODEFLAG_GIOU) {
            totalBufferSize += 3 * sizeof(T);
        }
        auto nAlignedMax = ubSize / totalBufferSize;
        nAligned_ = (nAlignedMax / BYTE_ALIGN_SIZE) * BYTE_ALIGN_SIZE;
        if (blkIdx_ < tilingData->bigCoreCount) {
            coreTaskCount_ = tilingData->blockCount + 1;
            globalTaskOffset_ = static_cast<int64_t>(blkIdx_) * coreTaskCount_;
        } else {
            coreTaskCount_ = tilingData->blockCount;
            globalTaskOffset_ = static_cast<int64_t>(blkIdx_) * coreTaskCount_ + tilingData->bigCoreCount;
        }
        if (globalTaskOffset_ >= nSize_) {
            return;
        }

        for (int i = 0; i < N; ++i) {
            boxesAGM_[i].SetGlobalBuffer((__gm__ T *)boxes_a + i * nSize_ + globalTaskOffset_);
            boxesBGM_[i].SetGlobalBuffer((__gm__ T *)boxes_b + i * nSize_ + globalTaskOffset_);
        }
        resGM_.SetGlobalBuffer((__gm__ T *)res + globalTaskOffset_);

        // boxes_a + boxes_b
        pipe_->InitBuffer(boxesAQueue_, 2, 4 * nAligned_ * sizeof(T));
        pipe_->InitBuffer(boxesBQueue_, 2, 4 * nAligned_ * sizeof(T));
        // res
        pipe_->InitBuffer(resQueue_, 2, nAligned_ * sizeof(T));
        // tmpbuf
        pipe_->InitBuffer(tmpBuf_, nAligned_ * sizeof(T));
        // area1+area2+w+h+overlap
        pipe_->InitBuffer(iouBuf_, 5 * nAligned_ * sizeof(T));
        if (modeFlag_ == MODEFLAG_GIOU) {
            // enclose_w + enclose_h + enclose_area
            pipe_->InitBuffer(giouBuf_, 3 * nAligned_ * sizeof(T));
        }
    }

    __aicore__ inline void Process() {
        if (globalTaskOffset_ >= nSize_) {
            return;
        }
        auto taskCount = coreTaskCount_;
        int64_t offset = 0;
        int32_t length = 0;
        while (taskCount > 0) {
            if (taskCount > nAligned_) {
                length = nAligned_;
            } else {
                length = taskCount;
            }
            CopyIn(offset, length);
            Compute(length);
            CopyOut(offset, length);
            offset += length;
            taskCount -= length;
        }
    }

  private:
    __aicore__ inline void CopyIn(int64_t offset, int32_t length) {
        LocalTensor<T> boxesALocal = boxesAQueue_.AllocTensor<T>();
        LocalTensor<T> boxesBLocal = boxesBQueue_.AllocTensor<T>();
        for (int i = 0; i < N; ++i) {
            DataCopyPad(boxesALocal[i * nAligned_], boxesAGM_[i][offset],
                {1, length * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(boxesBLocal[i * nAligned_], boxesBGM_[i][offset],
                {1, length * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        }
        boxesAQueue_.EnQue(boxesALocal);
        boxesBQueue_.EnQue(boxesBLocal);
    }

    __aicore__ inline void Compute(int32_t length) {
        LocalTensor<T> boxesALocal = boxesAQueue_.DeQue<T>();
        LocalTensor<T> boxesBLocal = boxesBQueue_.DeQue<T>();
        LocalTensor<T> resLocal = resQueue_.AllocTensor<T>();

        LocalTensor<T> x11Local = boxesALocal;
        LocalTensor<T> y11Local = boxesALocal[nAligned_];
        LocalTensor<T> x12Local = boxesALocal[2 * nAligned_];
        LocalTensor<T> y12Local = boxesALocal[3 * nAligned_];
        LocalTensor<T> x21Local = boxesBLocal;
        LocalTensor<T> y21Local = boxesBLocal[nAligned_];
        LocalTensor<T> x22Local = boxesBLocal[2 * nAligned_];
        LocalTensor<T> y22Local = boxesBLocal[3 * nAligned_];
        LocalTensor<T> iouLocal = iouBuf_.Get<T>();
        LocalTensor<T> area1Local = iouLocal;
        LocalTensor<T> area2Local = iouLocal[nAligned_];
        LocalTensor<T> wLocal = iouLocal[2 * nAligned_];
        LocalTensor<T> hLocal = iouLocal[3 * nAligned_];
        LocalTensor<T> overlapLocal = iouLocal[4 * nAligned_];
        LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();

        // area1
        AscendC::Sub(tmpLocal, x12Local, x11Local, length);
        AscendC::Sub(area1Local, y12Local, y11Local, length);
        AscendC::Mul(area1Local, area1Local, tmpLocal, length);

        // area2
        AscendC::Sub(tmpLocal, x22Local, x21Local, length);
        AscendC::Sub(area2Local, y22Local, y21Local, length);
        AscendC::Mul(area2Local, area2Local, tmpLocal, length);

        // width
        AscendC::Max(tmpLocal, x11Local, x21Local, length); // lt
        AscendC::Min(wLocal, x12Local, x22Local, length); // rb
        AscendC::Sub(wLocal, wLocal, tmpLocal, length); // rb - lt
        AscendC::ClampMin(wLocal, wLocal, static_cast<float>(0.0), length);

        // height
        AscendC::Max(tmpLocal, y11Local, y21Local, length); // lt
        AscendC::Min(hLocal, y12Local, y22Local, length); // rb
        AscendC::Sub(hLocal, hLocal, tmpLocal, length); // rb - lt
        AscendC::ClampMin(hLocal, hLocal, static_cast<float>(0.0), length);

        // overlap
        AscendC::Mul(overlapLocal, wLocal, hLocal, length);

        //union = area1 + area2 - overlap, use area1Local as union
        if (modeFlag_ == MODEFLAG_GIOU || modeFlag_ == MODEFLAG_IOU) {
            AscendC::Add(area1Local, area1Local, area2Local, length);
            AscendC::Sub(area1Local, area1Local, overlapLocal, length); // rb
        }
        AscendC::ClampMin(area1Local, area1Local, margin_, length);

        // ious = overlap / union
        AscendC::Div(resLocal, overlapLocal, area1Local, length);

        if (modeFlag_ == MODEFLAG_GIOU) {
            LocalTensor<T> giouLocal = giouBuf_.Get<T>();
            LocalTensor<T> enclose_w = giouLocal;
            LocalTensor<T> enclose_h = giouLocal[nAligned_];
            LocalTensor<T> enclose_area = giouLocal[2 * nAligned_];

            // width
            AscendC::Min(tmpLocal, x11Local, x21Local, length); // lt
            AscendC::Max(enclose_w, x12Local, x22Local, length); // rb
            AscendC::Sub(enclose_w, enclose_w, tmpLocal, length); // rb - lt
            AscendC::ClampMin(enclose_w, enclose_w, static_cast<float>(0.0), length);

            // height
            AscendC::Min(tmpLocal, y11Local, y21Local, length); // lt
            AscendC::Max(enclose_h, y12Local, y22Local, length); // rb
            AscendC::Sub(enclose_h, enclose_h, tmpLocal, length); // rb - lt
            AscendC::ClampMin(enclose_h, enclose_h, static_cast<float>(0.0), length);

            // enclose_area
            AscendC::Mul(enclose_area, enclose_w, enclose_h, length);

            // gious = ious - (enclose_area - union) / enclose_area
            AscendC::Sub(tmpLocal, enclose_area, area1Local, length);
            AscendC::Div(tmpLocal, tmpLocal, enclose_area, length);
            AscendC::Sub(resLocal, resLocal, tmpLocal, length);
        }

        resQueue_.EnQue(resLocal);
        boxesAQueue_.FreeTensor(boxesALocal);
        boxesBQueue_.FreeTensor(boxesBLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int32_t length) {
        LocalTensor<T> resLocal = resQueue_.DeQue<T>();
        DataCopyPad(resGM_[offset], resLocal, {1, length * static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        resQueue_.FreeTensor(resLocal);
    }

  private:
    TPipe *pipe_;
    TQue<TPosition::VECIN, 2> boxesAQueue_;
    TQue<TPosition::VECIN, 2> boxesBQueue_;
    TQue<TPosition::VECOUT, 2> resQueue_;
    TBuf<TPosition::VECCALC> iouBuf_;
    TBuf<TPosition::VECCALC> giouBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<T> boxesAGM_[N];
    GlobalTensor<T> boxesBGM_[N];
    GlobalTensor<T> resGM_;

    int32_t blkIdx_;
    int32_t coreTaskCount_;
    int64_t globalTaskOffset_;
    int64_t nSize_;
    int32_t nAligned_;
    uint32_t modeFlag_;
    float margin_;
};

extern "C" __global__ __aicore__ void bbox_overlaps(
    GM_ADDR boxes_a, GM_ADDR boxes_b, GM_ADDR res, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KernelBboxOverlaps<DTYPE_BOXES_A, 4> op;
    op.Init(boxes_a, boxes_b, res, &tilingData, &pipe);
    op.Process();
}
