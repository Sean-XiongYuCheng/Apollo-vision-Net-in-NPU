/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;

template <typename T1, typename T2> class KernelAdamwParamUpdate {
  public:
    __aicore__ inline KernelAdamwParamUpdate() = delete;

    __aicore__ inline KernelAdamwParamUpdate(GM_ADDR combined_param, GM_ADDR combined_grad, GM_ADDR exp_avg,
        GM_ADDR exp_avg_sq, GM_ADDR max_exp_avg_sq, const AdamwParamUpdateTilingData &tiling_data, TPipe *pipe)
        : pipe_(pipe) {
        InitTask(tiling_data);
        InitGM(combined_param, combined_grad, exp_avg, exp_avg_sq, max_exp_avg_sq);
        InitBuffer();
    }

    __aicore__ inline void Process();

  private:
    __aicore__ inline void InitTask(const AdamwParamUpdateTilingData &tiling) {
        coreId = GetBlockIdx();
        usedCoreNum = tiling.usedCoreNum;
        totalCoreTaskNum = tiling.totalCoreTaskNum;
        lr = tiling.lr;
        beta1 = tiling.beta1;
        beta2 = tiling.beta2;
        decay = tiling.decay;
        eps = tiling.eps;
        biasCorrection1 = tiling.biasCorrection1;
        biasCorrection2 = tiling.biasCorrection2;
        amsgrad = tiling.amsgrad;

        uint32_t avgTasks = totalCoreTaskNum / usedCoreNum;
        uint32_t remainTasks = totalCoreTaskNum % usedCoreNum;
        startOffset_ = avgTasks * coreId + (coreId < remainTasks ? coreId : remainTasks);
        endOffset_ = startOffset_ + avgTasks + (coreId < remainTasks ? 1 : 0);
        taskLoops_ = DivCeil(endOffset_ - startOffset_, singleProcessTaskNum);
    }

    __aicore__ inline void InitGM(
        GM_ADDR combined_param, GM_ADDR combined_grad, GM_ADDR exp_avg, GM_ADDR exp_avg_sq, GM_ADDR max_exp_avg_sq) {
        combinedParamGm.SetGlobalBuffer((__gm__ T2 *)(combined_param));
        combinedGradGm.SetGlobalBuffer((__gm__ T2 *)(combined_grad));
        expAvgGm.SetGlobalBuffer((__gm__ T2 *)(exp_avg));
        expAvgSqGm.SetGlobalBuffer((__gm__ T2 *)(exp_avg_sq));
        if (amsgrad) {
            maxExpAvgSqGm.SetGlobalBuffer((__gm__ T2 *)(max_exp_avg_sq));
        }
    }

    __aicore__ inline void InitBuffer() {
        InitInputBufferWithPingPong<T2>(combinedParamBuf, combinedParamDtypeBuf);
        InitInputBufferWithPingPong<T2>(combinedGradBuf, combinedGradDtypeBuf);
        InitInputBufferWithPingPong<T2>(expAvgBuf, expAvgDtypeBuf);
        InitInputBufferWithPingPong<T2>(expAvgSqBuf, expAvgSqDtypeBuf);
        InitInputBufferWithPingPong<T2>(maxExpAvgSqBuf, maxExpAvgSqDtypeBuf);
        pipe_->InitBuffer(denomBuf, singleProcessTaskNum * sizeof(T1));
        pipe_->InitBuffer(tempBuf, singleProcessTaskNum * sizeof(T1));
    }

    template <typename Dtype>
    __aicore__ inline void InitInputBufferWithPingPong(
        TBuf<TPosition::VECCALC> &bufferT, TBuf<TPosition::VECCALC> &bufferD) {
        if constexpr (std::is_same<T1, Dtype>::value) {
            pipe_->InitBuffer(bufferT, 2 * singleProcessTaskNum * sizeof(T1));
        } else {
            pipe_->InitBuffer(bufferT, singleProcessTaskNum * sizeof(T1));
            pipe_->InitBuffer(bufferD, 2 * singleProcessTaskNum * sizeof(Dtype));
        }
    }

    template <typename Dtype>
    __aicore__ inline LocalTensor<T1> CopyInSingle(int32_t taskIdx, int32_t taskOffset, int32_t actualTaskNum,
        TBuf<TPosition::VECCALC> &bufferT, TBuf<TPosition::VECCALC> &bufferD, GlobalTensor<Dtype> globalData) {
        int32_t bufferIdx = taskIdx % 2;
        if constexpr (std::is_same<T1, Dtype>::value) {
            LocalTensor<T1> inputLocalT = bufferT.Get<T1>();
            LocalTensor<T1> srcLocal = inputLocalT[bufferIdx * singleProcessTaskNum];
            DataCopyPad(srcLocal, globalData[taskOffset],
                {1, static_cast<uint32_t>(actualTaskNum * sizeof(Dtype)), 0, 0, 0}, {});
            SetFlag<HardEvent::MTE2_V>(bufferIdx);
            WaitFlag<HardEvent::MTE2_V>(bufferIdx);
            return srcLocal;
        } else {
            LocalTensor<T1> inputLocalT = bufferT.Get<T1>();
            LocalTensor<Dtype> inputLocalD = bufferD.Get<Dtype>();
            LocalTensor<Dtype> srcLocal = inputLocalD[bufferIdx * singleProcessTaskNum];
            DataCopyPad(srcLocal, globalData[taskOffset],
                {1, static_cast<uint32_t>(actualTaskNum * sizeof(Dtype)), 0, 0, 0}, {});
            SetFlag<HardEvent::MTE2_V>(bufferIdx);
            WaitFlag<HardEvent::MTE2_V>(bufferIdx);
            Cast(inputLocalT, srcLocal, RoundMode::CAST_NONE, actualTaskNum);
            return inputLocalT;
        }
    }

    template <typename Dtype>
    __aicore__ inline void CopyOutSingle(int32_t taskIdx, int32_t taskOffset, int32_t actualTaskNum,
        TBuf<TPosition::VECCALC> &bufferT, TBuf<TPosition::VECCALC> &bufferD, GlobalTensor<Dtype> globalData,
        LocalTensor<T1> localData) {
        int32_t bufferIdx = taskIdx % 2;
        if constexpr (std::is_same<T1, Dtype>::value) {
            SetFlag<HardEvent::V_MTE3>(bufferIdx);
            WaitFlag<HardEvent::V_MTE3>(bufferIdx);
            DataCopyPad(
                globalData[taskOffset], localData, {1, static_cast<uint32_t>(actualTaskNum * sizeof(Dtype)), 0, 0, 0});
        } else {
            LocalTensor<Dtype> inputLocalD = bufferD.Get<Dtype>();
            LocalTensor<Dtype> srcLocal = inputLocalD[bufferIdx * singleProcessTaskNum];
            Cast(srcLocal, localData, RoundMode::CAST_RINT, actualTaskNum);
            SetFlag<HardEvent::V_MTE3>(bufferIdx);
            WaitFlag<HardEvent::V_MTE3>(bufferIdx);
            DataCopyPad(
                globalData[taskOffset], srcLocal, {1, static_cast<uint32_t>(actualTaskNum * sizeof(Dtype)), 0, 0, 0});
        }
    }

    __aicore__ inline void ProcessSingle(int32_t taskIdx, int32_t taskOffset, int32_t actualTaskNum) {
        int32_t taskAlign = AlignUp(actualTaskNum, 32 / sizeof(T1));
        int32_t bufferIdx = taskIdx % 2;

        LocalTensor<T1> denom = denomBuf.Get<T1>();
        LocalTensor<T1> temp = tempBuf.Get<T1>();
        LocalTensor<T1> combinedParam, combinedGrad, expAvg, expAvgSq, maxExpAvgSq;

        WaitFlag<HardEvent::V_MTE2>(bufferIdx);
        WaitFlag<HardEvent::MTE3_MTE2>(bufferIdx);

        combinedParam = CopyInSingle<T2>(
            taskIdx, taskOffset, actualTaskNum, combinedParamBuf, combinedParamDtypeBuf, combinedParamGm);
        combinedGrad =
            CopyInSingle<T2>(taskIdx, taskOffset, actualTaskNum, combinedGradBuf, combinedGradDtypeBuf, combinedGradGm);
        expAvg = CopyInSingle<T2>(taskIdx, taskOffset, actualTaskNum, expAvgBuf, expAvgDtypeBuf, expAvgGm);
        expAvgSq = CopyInSingle<T2>(taskIdx, taskOffset, actualTaskNum, expAvgSqBuf, expAvgSqDtypeBuf, expAvgSqGm);
        if (amsgrad) {
            maxExpAvgSq = CopyInSingle<T2>(
                taskIdx, taskOffset, actualTaskNum, maxExpAvgSqBuf, maxExpAvgSqDtypeBuf, maxExpAvgSqGm);
        }

        T1 decayFactor = static_cast<T1>(1.0f - lr * decay);
        Muls(combinedParam, combinedParam, decayFactor, taskAlign);

        Muls(temp, expAvg, static_cast<T1>(beta1), taskAlign);
        Muls(expAvg, combinedGrad, static_cast<T1>(1.0f - beta1), taskAlign);
        Add(expAvg, temp, expAvg, taskAlign);

        Mul(temp, combinedGrad, combinedGrad, taskAlign);
        Muls(temp, temp, static_cast<T1>(1.0f - beta2), taskAlign);
        Muls(expAvgSq, expAvgSq, static_cast<T1>(beta2), taskAlign);
        Add(expAvgSq, expAvgSq, temp, taskAlign);

        if (amsgrad) {
            Max(maxExpAvgSq, maxExpAvgSq, expAvgSq, taskAlign);
            Sqrt(denom, maxExpAvgSq, taskAlign);
        } else {
            Sqrt(denom, expAvgSq, taskAlign);
        }

        T1 sqrtBiasCorrection2 = static_cast<T1>(sqrt(biasCorrection2));
        Duplicate(temp, sqrtBiasCorrection2, taskAlign);
        Div(denom, denom, temp, taskAlign);
        Adds(denom, denom, static_cast<T1>(eps), taskAlign);

        T1 stepSize = static_cast<T1>(-1 * lr / biasCorrection1);
        Div(temp, expAvg, denom, taskAlign);
        Muls(temp, temp, stepSize, taskAlign);
        Add(combinedParam, combinedParam, temp, taskAlign);

        CopyOutSingle<T2>(taskIdx, taskOffset, actualTaskNum, combinedParamBuf, combinedParamDtypeBuf, combinedParamGm,
            combinedParam);
        CopyOutSingle<T2>(taskIdx, taskOffset, actualTaskNum, expAvgBuf, expAvgDtypeBuf, expAvgGm, expAvg);
        CopyOutSingle<T2>(taskIdx, taskOffset, actualTaskNum, expAvgSqBuf, expAvgSqDtypeBuf, expAvgSqGm, expAvgSq);
        if (amsgrad) {
            CopyOutSingle<T2>(
                taskIdx, taskOffset, actualTaskNum, maxExpAvgSqBuf, maxExpAvgSqDtypeBuf, maxExpAvgSqGm, maxExpAvgSq);
        }

        SetFlag<HardEvent::V_MTE2>(bufferIdx);
        SetFlag<HardEvent::MTE3_MTE2>(bufferIdx);
    }

  private:
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> combinedParamBuf, combinedGradBuf, expAvgBuf, expAvgSqBuf, maxExpAvgSqBuf, denomBuf,
        tempBuf;
    TBuf<TPosition::VECCALC> combinedParamDtypeBuf, combinedGradDtypeBuf, expAvgDtypeBuf, expAvgSqDtypeBuf,
        maxExpAvgSqDtypeBuf;
    GlobalTensor<T2> combinedParamGm;
    GlobalTensor<T2> combinedGradGm;
    GlobalTensor<T2> expAvgGm;
    GlobalTensor<T2> expAvgSqGm;
    GlobalTensor<T2> maxExpAvgSqGm;
    float lr, beta1, beta2, decay, eps, biasCorrection1, biasCorrection2;
    bool amsgrad;
    int32_t coreId, usedCoreNum, totalCoreTaskNum, coreProcessTaskNum;
    int32_t singleProcessTaskNum = 4096;
    int32_t startOffset_, endOffset_, taskLoops_;
};

template <typename T1, typename T2> __aicore__ inline void KernelAdamwParamUpdate<T1, T2>::Process() {
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);
    for (int32_t taskIdx = 0; taskIdx < taskLoops_; taskIdx++) {
        int32_t taskOffset = startOffset_ + taskIdx * singleProcessTaskNum;
        int32_t actualTaskNum = min(endOffset_ - taskOffset, singleProcessTaskNum);
        ProcessSingle(taskIdx, taskOffset, actualTaskNum);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);
}

extern "C" __global__ __aicore__ void adamw_param_update(GM_ADDR combined_param, GM_ADDR combined_grad, GM_ADDR exp_avg,
    GM_ADDR exp_avg_sq, GM_ADDR max_exp_avg_sq, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR y4, GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        KernelAdamwParamUpdate<float, float> op(
            combined_param, combined_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelAdamwParamUpdate<float, half> op(
            combined_param, combined_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelAdamwParamUpdate<float, bfloat16_t> op(
            combined_param, combined_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, tiling_data, &pipe);
        op.Process();
    }
}
