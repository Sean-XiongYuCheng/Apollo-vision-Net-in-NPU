/*

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

#define DTYPE_F_BOXS float

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t BOX_SHAPE_N = 5;
constexpr int32_t BOX_SHAPE_N_PADDING = 3;
constexpr int32_t BOX_ALIGN_NUM = 8;

// (x_center, y_center, width, height, angle)
constexpr uint32_t ROTATED_XCENTER_OFFSET = 0;
constexpr uint32_t ROTATED_YCENTER_OFFSET = 1;
constexpr uint32_t ROTATED_WIDTH_OFFSET = 2;
constexpr uint32_t ROTATED_HEIGHT_OFFSET = 3;
constexpr uint32_t ROTATED_ANGLE_OFFSET = 4;

template <typename T> struct RotatedBox {
    T x_ctr, y_ctr, w, h, a;
};

template <typename T> class KernelNmsRotatedDist {
  public:
    __aicore__ inline KernelNmsRotatedDist(TPipe *pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR mask, const NmsRotatedDistTilingData *__restrict tiling_data) {
        usedCoreNum = tiling_data->usedCoreNum;
        boxNum = tiling_data->boxNum;
        maskNum = tiling_data->maskNum;
        overlapThresh = tiling_data->threshold;

        boxNumPerHeadCore = tiling_data->boxNumPerHeadCore;
        boxNumPerTailCore = tiling_data->boxNumPerTailCore;

        uint32_t core_id = GetBlockIdx();
        if (core_id < tiling_data->headCoreNum) {
            loopTime = boxNumPerHeadCore / dataAlign;
            remainBoxNum = boxNumPerHeadCore % dataAlign;
            startBoxNumIdx = core_id * boxNumPerHeadCore;
        } else {
            loopTime = boxNumPerTailCore / dataAlign;
            remainBoxNum = boxNumPerTailCore % dataAlign;
            startBoxNumIdx =
                tiling_data->headCoreNum * boxNumPerHeadCore + (core_id - tiling_data->headCoreNum) * boxNumPerTailCore;
        }

        boxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(boxes), static_cast<uint64_t>(boxNum) * BOX_SHAPE_N);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(mask), static_cast<uint64_t>(maskNum) * boxNum);

        pPipe->InitBuffer(inQueueCur, BUFFER_NUM, dataAlign * sizeof(T));
        pPipe->InitBuffer(inQueueBox, BUFFER_NUM, dataAlign * BOX_ALIGN_NUM * sizeof(T));

        pPipe->InitBuffer(outQueueMask, BUFFER_NUM, dataAlign * sizeof(int16_t));
        pPipe->InitBuffer(oneMask, BUFFER_NUM, dataAlign * sizeof(int16_t));

        pPipe->InitBuffer(comBuf, dataAlign * sizeof(float));

        pPipe->InitBuffer(angleBuf_, dataAlign * sizeof(DTYPE_F_BOXS));
        pPipe->InitBuffer(sinBuf_, dataAlign * sizeof(DTYPE_F_BOXS));
        pPipe->InitBuffer(cosBuf_, dataAlign * sizeof(DTYPE_F_BOXS));

        angleLocalT_ = angleBuf_.Get<DTYPE_F_BOXS>();
        sinLocalT_ = sinBuf_.Get<DTYPE_F_BOXS>();
        cosLocalT_ = cosBuf_.Get<DTYPE_F_BOXS>();
    }

    __aicore__ inline void Process() {
        LocalTensor<int16_t> oneLocal = oneMask.AllocTensor<int16_t>();
        Duplicate(oneLocal, static_cast<int16_t>(1), dataAlign);
        for (size_t i = 0; i < boxNum; ++i) {
            CopyAndGetCurBox(i);
            for (size_t j = 0; j < loopTime; ++j) {
                uint32_t start = startBoxNumIdx + dataAlign * j;
                if (i >= start + dataAlign) {
                    DataCopy(maskGm[i * maskNum + start], oneLocal, dataAlign);
                    continue;
                }

                LoopProcess(i, start, dataAlign);
            }

            if (remainBoxNum > 0) {
                uint32_t start = startBoxNumIdx + dataAlign * loopTime;
                if (i >= start + remainBoxNum) {
                    DataCopyParams copyParams1{1, (uint16_t)(remainBoxNum * sizeof(int16_t)), 0, 0};
                    DataCopyPad(maskGm[i * maskNum + start], oneLocal, copyParams1);
                    continue;
                }
                LoopProcess(i, start, remainBoxNum);
            }
        }
        oneMask.FreeTensor(oneLocal);
    }

    __aicore__ inline void LoopProcess(uint32_t index, uint32_t start, uint32_t multiBoxNum) {
        CopyIn(start, multiBoxNum);
        Compute(index, start, multiBoxNum);
        CopyOut(index, start, multiBoxNum);
    }

  private:
    __aicore__ inline void CopyIn(int32_t com_box, uint32_t multiBoxNum) {
        LocalTensor<T> boxLocal = inQueueBox.AllocTensor<T>();
        DataCopyPadParams padParams{true, 0, BOX_SHAPE_N_PADDING, 0};
        DataCopyParams copyParams1{(uint16_t)multiBoxNum, (uint16_t)(BOX_SHAPE_N * sizeof(T)), 0, 0};
        DataCopyPad(boxLocal, boxGm[static_cast<uint64_t>(com_box) * BOX_SHAPE_N], copyParams1, padParams);
        inQueueBox.EnQue(boxLocal);
    }

    __aicore__ inline void CopyAndGetCurBox(uint32_t cur_box) {
        LocalTensor<T> curLocal = inQueueCur.AllocTensor<T>();
        DataCopyParams copyParams{1, (uint16_t)(BOX_SHAPE_N * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{true, 0, BOX_SHAPE_N_PADDING, 0};
        DataCopyPad(curLocal, boxGm[static_cast<uint64_t>(cur_box) * BOX_SHAPE_N], copyParams, padParams);
        inQueueCur.EnQue(curLocal);
        curTemp = inQueueCur.DeQue<T>();
        curBox.x_ctr = curTemp.GetValue(ROTATED_XCENTER_OFFSET);
        curBox.y_ctr = curTemp.GetValue(ROTATED_YCENTER_OFFSET);
        curBox.w = curTemp.GetValue(ROTATED_WIDTH_OFFSET);
        curBox.h = curTemp.GetValue(ROTATED_HEIGHT_OFFSET);
        curBox.a = curTemp.GetValue(ROTATED_ANGLE_OFFSET);
        inQueueCur.FreeTensor(curTemp);

        // Compute cos/sin once per curBox to avoid redundant calculation in dist_check
        angleLocalT_.SetValue(0, curBox.a);
        Cos(cosLocalT_, angleLocalT_);
        Sin(sinLocalT_, angleLocalT_);
        curBoxCos = cosLocalT_.GetValue(0);
        curBoxSin = sinLocalT_.GetValue(0);
    }

    __aicore__ inline void Compute(int32_t cur_box, int32_t com_box, uint32_t cmpNum) {
        LocalTensor<float> comLocal = comBuf.Get<float>();
        boxTemp = inQueueBox.DeQue<T>();

        LocalTensor<int16_t> outLocal = outQueueMask.AllocTensor<int16_t>();
        for (size_t i = 0; i < cmpNum; i++) {
            if (cur_box >= com_box + i) {
                outLocal.SetValue(i, 1);
                continue;
            }

            DataCopy(comLocal, boxTemp[i * BOX_ALIGN_NUM], BOX_ALIGN_NUM);
            auto flag = dist_check(comLocal);
            if (flag) {
                outLocal.SetValue(i, 0);
            } else {
                outLocal.SetValue(i, 1);
            }
        }
        outQueueMask.EnQue<int16_t>(outLocal);
        inQueueBox.FreeTensor(boxTemp);
    }

    __aicore__ inline void CopyOut(int32_t cur_box, int32_t com_box, uint32_t multiBoxNum) {
        LocalTensor<int16_t> outLocal = outQueueMask.DeQue<int16_t>();
        DataCopyParams copyParams1{1, (uint16_t)(multiBoxNum * sizeof(int16_t)), 0, 0};
        DataCopyPad(
            maskGm[static_cast<uint64_t>(cur_box) * maskNum + static_cast<uint64_t>(com_box)], outLocal, copyParams1);
        outQueueMask.FreeTensor(outLocal);
    }

    __aicore__ inline bool dist_check(const LocalTensor<float> &boxBTensor) {
        DTYPE_F_BOXS xCtrB = boxBTensor.GetValue(ROTATED_XCENTER_OFFSET);
        DTYPE_F_BOXS yCtrB = boxBTensor.GetValue(ROTATED_YCENTER_OFFSET);

        DTYPE_F_BOXS dist_x = curBox.x_ctr - xCtrB;
        DTYPE_F_BOXS dist_y = curBox.y_ctr - yCtrB;

        // Use pre-computed cos/sin from CopyAndGetCurBox()
        // DTYPE_F_BOXS dist_w = Abs(dist_x * curBoxCos - dist_y * curBoxSin);
        // DTYPE_F_BOXS dist_h = Abs(dist_x * curBoxSin + dist_y * curBoxCos);

        DTYPE_F_BOXS val_w = dist_x * curBoxCos - dist_y * curBoxSin;
        DTYPE_F_BOXS dist_w = (val_w >= 0) ? val_w : -val_w;
        DTYPE_F_BOXS val_h = dist_x * curBoxSin + dist_y * curBoxCos;
        DTYPE_F_BOXS dist_h = (val_h >= 0) ? val_h : -val_h;

        bool inside = (dist_w <= curBox.w && dist_h <= curBox.h);
        return inside && (overlapThresh < 0.5f);
    }

  private:
    TPipe *pPipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCur, inQueueBox;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask, oneMask;
    TBuf<TPosition::VECCALC> comBuf;

    TBuf<TPosition::VECCALC> angleBuf_, sinBuf_, cosBuf_;

    GlobalTensor<T> boxGm;
    GlobalTensor<int16_t> maskGm;
    LocalTensor<float> curTemp, boxTemp;

    LocalTensor<DTYPE_F_BOXS> angleLocalT_, sinLocalT_, cosLocalT_;

    RotatedBox<DTYPE_F_BOXS> curBox;
    DTYPE_F_BOXS curBoxCos;
    DTYPE_F_BOXS curBoxSin;

    uint32_t usedCoreNum;
    uint32_t loopTime;

    uint32_t boxNum;
    uint32_t maskNum;
    uint32_t dataAlign = 16;
    uint32_t boxNumPerHeadCore;
    uint32_t boxNumPerTailCore;
    uint32_t startBoxNumIdx;

    uint32_t remainBoxNum;

    float overlapThresh;
};

extern "C" __global__ __aicore__ void nms_rotated_dist(GM_ADDR boxes, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    const NmsRotatedDistTilingData *__restrict tilingDevice = &tilingData;
    TPipe pipe;

    KernelNmsRotatedDist<float> op(&pipe);
    op.Init(boxes, mask, tilingDevice);
    op.Process();
}
