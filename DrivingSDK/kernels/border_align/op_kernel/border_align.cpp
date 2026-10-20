/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include <cmath>
using namespace std;
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BOX_INFO_NUM = 4;

class BorderAlign {
  public:
    __aicore__ inline BorderAlign() {}
    __aicore__ inline void Init(
        GM_ADDR featureMap, GM_ADDR rois, GM_ADDR featureOut, const BorderAlignTilingData *tiling_data) {
        batchSize = tiling_data->batchSize; // 输入特征的batch大小
        channels = tiling_data->channels; // 输入特征的通道长度
        inputH = tiling_data->inputH; // 输入特征的高度
        inputW = tiling_data->inputW; // 输入特征的宽度
        pooledSize = tiling_data->pooledSize; // 在每条边计算采样点个数
        roisNumAligned = tiling_data->roisNumAligned; // 对齐之后的RoI总数
        tailNum = tiling_data->tailNum; // tailNum = roisNumAligned - roisNum
        roisNumPerLcore = tiling_data->roisNumPerLcore; // 每个大核分配的RoI数
        roisNumPerScore = tiling_data->roisNumPerScore; // 每个小核分配的RoI数
        lcoreNum = tiling_data->lcoreNum; // 大核数量
        scoreNum = tiling_data->scoreNum; // 小核数量
        inputBufferSize = tiling_data->inputBufferSize; // 搬运特征向量的Tque Buffer大小
        roisBufferSize = tiling_data->roisBufferSize; // 搬运RoI的Tque Buffer大小
        roisNumPerLoop = tiling_data->roisNumPerLoop; // 每次搬运到UB的RoI数量
        totalOutputLength =
            static_cast<uint64_t>(batchSize) * inputH * inputW * (pooledSize + 1) * channels; // 输出的Feature Map长度
        totalInputLength = static_cast<uint64_t>(batchSize) * inputH * inputW * channels; // 输入的Feature Map长度
        moveInLength = tiling_data->moveInLength; // 每次搬入的Feature长度
        moveOutLength = tiling_data->moveOutLength; // 每次搬出的Feature长度
        roisNum = roisNumAligned - tailNum;

        if (GetBlockIdx() < lcoreNum) {
            roisNumPerCore = roisNumPerLcore;
            roisStartAddr = GetBlockIdx() * roisNumPerLcore * BOX_INFO_NUM;
        } else {
            roisNumPerCore = roisNumPerScore;
            roisStartAddr = (lcoreNum * roisNumPerLcore + (GetBlockIdx() - lcoreNum) * roisNumPerScore) * BOX_INFO_NUM;
        }
        totalLoop = (roisNumPerCore + roisNumPerLoop - 1) / roisNumPerLoop;

        featureGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)featureMap, totalInputLength * sizeof(DTYPE_INPUT));
        roisGm.SetGlobalBuffer((__gm__ DTYPE_ROIS *)rois, roisNumAligned * BOX_INFO_NUM * sizeof(DTYPE_ROIS));
        outputGm.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)featureOut, totalOutputLength * sizeof(DTYPE_OUTPUT));

        pipe.InitBuffer(inQueueBox, BUFFER_NUM, roisBufferSize);
        pipe.InitBuffer(atomicAddBuffer, BUFFER_NUM, inputBufferSize);
        pipe.InitBuffer(inQueueFeatureFloorRow, BUFFER_NUM, 2 * inputBufferSize);
        pipe.InitBuffer(inQueueFeatureCeilRow, BUFFER_NUM, 2 * inputBufferSize);
        pipe.InitBuffer(outQueueFeature, BUFFER_NUM, inputBufferSize);
        pipe.InitBuffer(tmpValueBuffer, BUFFER_NUM, 8 * inputBufferSize);
    }
    __aicore__ inline void Process() {
        // 用于搬运双线性插值需要的四个点对应的特征向量
        featureFloorRow = inQueueFeatureFloorRow.AllocTensor<float>();
        featureCeilRow = inQueueFeatureCeilRow.AllocTensor<float>();
        tmpValue = tmpValueBuffer.AllocTensor<float>(); // 用于双线性插值的中间特征计算
        outFeature = outQueueFeature.AllocTensor<float>(); // 用于特征搬出
        boxLocal = inQueueBox.AllocTensor<float>(); // 用于搬入RoI
        for (int32_t loopIdx = 0; loopIdx < totalLoop; loopIdx++) {
            uint32_t currentRois = GetCurrentRois(loopIdx);
            CopyInBoxes(loopIdx, currentRois);
            Compute(loopIdx, currentRois);
        }
        inQueueFeatureFloorRow.FreeTensor(featureFloorRow);
        inQueueFeatureCeilRow.FreeTensor(featureCeilRow);
        tmpValueBuffer.FreeTensor(tmpValue);
        outQueueFeature.FreeTensor(outFeature);
        inQueueBox.FreeTensor(boxLocal);
    }

  private:
    __aicore__ inline uint32_t GetCurrentRois(int32_t loopIdx) {
        uint32_t processedRois = static_cast<uint32_t>(loopIdx) * roisNumPerLoop;
        uint32_t remainingRois = roisNumPerCore - processedRois;
        return remainingRois < roisNumPerLoop ? remainingRois : roisNumPerLoop;
    }

    __aicore__ inline void CopyInBoxes(int32_t loopIdx, uint32_t currentRois) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(currentRois * BOX_INFO_NUM * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(boxLocal, roisGm[static_cast<uint32_t>(loopIdx) * roisNumPerLoop * BOX_INFO_NUM + roisStartAddr],
            copyParams, padParams);
        inQueueBox.EnQue(boxLocal);
        boxLocal = inQueueBox.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(int32_t loopIdx, uint32_t currentRois) {
        for (int32_t boxIdx = 0; boxIdx < currentRois; boxIdx++) {
            ComputeOneBox(loopIdx, boxIdx);
        }
    }

    __aicore__ inline void ComputeOneBox(int32_t loopIdx, int32_t boxIdx) {
        int32_t boxIdx_ = loopIdx * roisNumPerLoop + boxIdx + roisStartAddr / BOX_INFO_NUM;
        // boxIdx_表示第几个box，如果超过了本核需要计算的RoI总数
        if (boxIdx_ >= roisNumPerCore + roisStartAddr / BOX_INFO_NUM) {
            return;
        }
        // 或者大于所有RoI的总数，那么就跳过计算
        if (boxIdx_ >= roisNum) {
            return;
        }

        float xLoc;
        float yLoc;
        float channelIdx;
        float x1 = boxLocal.GetValue(BOX_INFO_NUM * boxIdx);
        float y1 = boxLocal.GetValue(BOX_INFO_NUM * boxIdx + 1);
        float x2 = boxLocal.GetValue(BOX_INFO_NUM * boxIdx + 2);
        float y2 = boxLocal.GetValue(BOX_INFO_NUM * boxIdx + 3);
        float dx = (x2 - x1) / static_cast<float>(pooledSize);
        float dy = (y2 - y1) / static_cast<float>(pooledSize);
        int32_t batchIdx = boxIdx_ / (inputH * inputW);
        int32_t xBox = boxIdx_ % inputW;
        int32_t yBox = boxIdx_ / inputW % inputH;
        uint64_t baseAddrCopyIn = static_cast<uint64_t>(batchIdx) * channels * inputH * inputW;
        uint64_t baseAddrCopyOut = static_cast<uint64_t>(batchIdx) * inputH * inputW * (pooledSize + 1) * channels +
            yBox * inputW * (pooledSize + 1) * channels + xBox * (pooledSize + 1) * channels;
        // 遍历上边缘， 起始点为（x1, y1），对应特征的通道为[0, channels / 4]
        xLoc = x1;
        yLoc = y1;
        channelIdx = 0;
        for (int32_t poolIdx = 0; poolIdx < pooledSize + 1; poolIdx++) {
            BilinearInterpolate(xLoc, yLoc, baseAddrCopyIn, channelIdx);
            FeatureCopyOut(baseAddrCopyOut, poolIdx, channelIdx);
            xLoc = xLoc + dx;
        }
        // 遍历左边缘， 起始点为（x1, y1），对应特征的通道为[channels / 4, channels / 2]
        xLoc = x1;
        yLoc = y1;
        channelIdx = 1;
        for (int32_t poolIdx = 0; poolIdx < pooledSize + 1; poolIdx++) {
            BilinearInterpolate(xLoc, yLoc, baseAddrCopyIn, channelIdx);
            FeatureCopyOut(baseAddrCopyOut, poolIdx, channelIdx);
            yLoc = yLoc + dy;
        }
        // 遍历下边缘， 起始点为（x2, y2），对应特征的通道为[channels / 2, channels * 3 / 4]
        xLoc = x2;
        yLoc = y2;
        channelIdx = 2;
        for (int32_t poolIdx = 0; poolIdx < pooledSize + 1; poolIdx++) {
            BilinearInterpolate(xLoc, yLoc, baseAddrCopyIn, channelIdx);
            FeatureCopyOut(baseAddrCopyOut, poolIdx, channelIdx);
            xLoc = xLoc - dx;
        }
        // 遍历右边缘， 起始点为（x2, y2），对应特征的通道为[channels * 3 / 4, channels]
        xLoc = x2;
        yLoc = y2;
        channelIdx = 3;
        for (int32_t poolIdx = 0; poolIdx < pooledSize + 1; poolIdx++) {
            BilinearInterpolate(xLoc, yLoc, baseAddrCopyIn, channelIdx);
            FeatureCopyOut(baseAddrCopyOut, poolIdx, channelIdx);
            yLoc = yLoc - dy;
        }
    }

    __aicore__ inline void BilinearInterpolate(float xLoc, float yLoc, uint64_t baseAddrCopyIn, int32_t channelIdx) {
        if (yLoc < -1 || yLoc > inputH || xLoc < -1 || xLoc > inputW) {
            float zero_factor = 0;
            Muls(outFeature, outFeature, zero_factor, moveInLength);
        } else {
            int32_t xFloor = static_cast<int32_t>(xLoc);
            int32_t yFloor = static_cast<int32_t>(yLoc);
            int32_t xCeil = xFloor + 1;
            int32_t yCeil = yFloor + 1;
            if (xFloor >= (inputW - 1)) {
                xCeil = inputW - 1;
                xFloor = xCeil;
                xLoc = static_cast<float>(xCeil);
            }
            if (yFloor >= (inputH - 1)) {
                yCeil = inputH - 1;
                yFloor = yCeil;
                yLoc = static_cast<float>(yCeil);
            }

            float lx = xLoc - static_cast<float>(xFloor);
            float ly = yLoc - static_cast<float>(yFloor);
            float hx = static_cast<float>(1) - lx;
            float hy = static_cast<float>(1) - ly;
            float weightP1 = static_cast<float>(hy * hx);
            float weightP2 = static_cast<float>(hy * lx);
            float weightP3 = static_cast<float>(ly * hx);
            float weightP4 = static_cast<float>(ly * lx);
            uint64_t baseAddrCopyIn_ = baseAddrCopyIn + channelIdx * channels / 4;

            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            bool copyTwoColumns = xFloor != xCeil;
            bool copyTwoRows = yFloor != yCeil;
            CopyInFeatureRow(
                featureFloorRow, baseAddrCopyIn_ + yFloor * inputW * channels + xFloor * channels, copyTwoColumns);
            inQueueFeatureFloorRow.EnQue(featureFloorRow);
            featureFloorRow = inQueueFeatureFloorRow.DeQue<float>();
            if (copyTwoRows) {
                CopyInFeatureRow(
                    featureCeilRow, baseAddrCopyIn_ + yCeil * inputW * channels + xFloor * channels, copyTwoColumns);
                inQueueFeatureCeilRow.EnQue(featureCeilRow);
                featureCeilRow = inQueueFeatureCeilRow.DeQue<float>();
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            LocalTensor<float> featureFloorFloor = featureFloorRow;
            LocalTensor<float> featureFloorCeil = featureFloorRow;
            if (copyTwoColumns) {
                featureFloorCeil = featureFloorRow[moveInLength];
            }
            LocalTensor<float> featureCeilFloor = featureFloorFloor;
            LocalTensor<float> featureCeilCeil = featureFloorCeil;
            if (copyTwoRows) {
                featureCeilFloor = featureCeilRow;
                featureCeilCeil = featureCeilRow;
                if (copyTwoColumns) {
                    featureCeilCeil = featureCeilRow[moveInLength];
                }
            }
            Muls(tmpValue[moveInLength * 0], featureFloorFloor, weightP1, moveInLength);
            Muls(tmpValue[moveInLength * 1], featureFloorCeil, weightP2, moveInLength);
            Add(tmpValue[moveInLength * 0], tmpValue[moveInLength * 0], tmpValue[moveInLength * 1], moveInLength);
            Muls(tmpValue[moveInLength * 2], featureCeilFloor, weightP3, moveInLength);
            Muls(tmpValue[moveInLength * 3], featureCeilCeil, weightP4, moveInLength);
            Add(tmpValue[moveInLength * 2], tmpValue[moveInLength * 2], tmpValue[moveInLength * 3], moveInLength);
            Add(outFeature, tmpValue[moveInLength * 0], tmpValue[moveInLength * 2], moveInLength);
        }
    }

    __aicore__ inline void CopyInFeatureRow(LocalTensor<float> &dst, uint64_t srcOffset, bool copyTwoColumns) {
        if (!copyTwoColumns) {
            CopyInFeature(dst, srcOffset);
            return;
        }
        if (moveOutLength == inputBufferSize) {
            DataCopyParams copyParams;
            copyParams.blockCount = 2;
            copyParams.blockLen = moveInLength / 8;
            copyParams.srcStride = (channels - channels / 4) / 8;
            copyParams.dstStride = 0;
            DataCopy(dst, featureGm[srcOffset], copyParams);
            return;
        }
        DataCopyParams copyParams{2, static_cast<uint16_t>(moveOutLength),
            static_cast<uint16_t>((channels - channels / 4) * sizeof(float)), 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(dst, featureGm[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyInFeature(LocalTensor<float> &dst, uint64_t srcOffset) {
        if (moveOutLength == inputBufferSize) {
            DataCopy(dst, featureGm[srcOffset], moveInLength);
            return;
        }
        DataCopyParams copyParams{1, static_cast<uint16_t>(moveOutLength), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(dst, featureGm[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void FeatureCopyOut(uint64_t baseAddrCopyOut, int32_t poolIdx, int32_t channelIdx) {
        outQueueFeature.EnQue(outFeature);
        outFeature = outQueueFeature.DeQue<float>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        DataCopyExtParams copyParams{1, moveOutLength, 0, 0, 0};
        DataCopyPad(outputGm[baseAddrCopyOut + poolIdx * channels + channelIdx * channels / 4], outFeature, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }

  private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFeatureFloorRow, inQueueFeatureCeilRow, inQueueBox;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueFeature;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpValueBuffer, atomicAddBuffer;
    GlobalTensor<float> featureGm, roisGm, outputGm;
    LocalTensor<float> featureFloorRow, featureCeilRow, tmpValue, outFeature, boxLocal;

    // 使用64位整数避免上溢出
    uint64_t totalOutputLength, totalInputLength;

    uint32_t batchSize, inputH, inputW, channels, roisNumAligned, tailNum, roisNum;
    uint32_t roisNumPerCore, roisNumPerLoop, roisStartAddr, moveInLength, totalLoop;
    uint32_t roisNumPerLcore, roisNumPerScore, lcoreNum, scoreNum, inputBufferSize, roisBufferSize, moveOutLength;
    int32_t pooledSize;
};

extern "C" __global__ __aicore__ void border_align(
    GM_ADDR featureMap, GM_ADDR rois, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BorderAlign op;
    op.Init(featureMap, rois, output, &tiling_data);
    op.Process();
}
