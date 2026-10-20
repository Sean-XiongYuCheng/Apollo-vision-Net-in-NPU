#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace std;

constexpr int32_t ALIGN_NUM = 8;
constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t SRC_SHAPE_DIM = 8;
constexpr int32_t ONE_VALUE = 1;
constexpr int32_t ZERO_VALUE = 0;
constexpr float ZERO_FLOAT_VALUE = 0.0f;

// ═══ TILING_KEY_IS(0): Scattered — no reduce_sum, grad = y*dy - y*y*dy ═══

class GraphSoftmaxGradScattered {
  public:
    __aicore__ inline GraphSoftmaxGradScattered() {}
    __aicore__ inline void Init(GM_ADDR index, GM_ADDR softmax_output, GM_ADDR grad_output, GM_ADDR reduce_sum,
        GM_ADDR src_grad, const GraphSoftmaxGradTilingData *tiling_data, TPipe *pipe) {
        ASSERT(GetBlockNum() != 0);
        blockIndex = GetBlockIdx();
        _pipe = pipe;
        GetTilingData(tiling_data);
        uint64_t bufferSize = taskNumPerLoop * SRC_SHAPE_DIM * FLOAT_SIZE;
        uint64_t castEdgeNum = static_cast<uint64_t>(edgeNum);
        uint64_t gmSize = castEdgeNum * SRC_SHAPE_DIM;
        indexGM.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, castEdgeNum);
        softmaxGM.SetGlobalBuffer((__gm__ DTYPE_SOFTMAX_OUTPUT *)softmax_output, gmSize);
        gradGM.SetGlobalBuffer((__gm__ DTYPE_GRAD_OUTPUT *)grad_output, gmSize);
        srcgradGM.SetGlobalBuffer((__gm__ DTYPE_SRC_GRAD *)src_grad, gmSize);
        _pipe->InitBuffer(SoftmaxOutputBuffer, bufferSize);
        _pipe->InitBuffer(IndexTensorBuffer, taskNumPerLoop * FLOAT_SIZE);
        _pipe->InitBuffer(GradOutputBuffer, bufferSize);
        _pipe->InitBuffer(SrcGradTensorBuffer, bufferSize);
    }

    __aicore__ inline void Process() {
        IndexTensor = IndexTensorBuffer.Get<DTYPE_INDEX>();
        SoftmaxOutputTensor = SoftmaxOutputBuffer.Get<DTYPE_SOFTMAX_OUTPUT>();
        GradOutputTensor = GradOutputBuffer.Get<DTYPE_GRAD_OUTPUT>();
        SrcGradTensor = SrcGradTensorBuffer.Get<DTYPE_SRC_GRAD>();
        if (taskLoop == 1) {
            CopyIn(GetCopyIndex(0), GetTaskNum(0));
            ComputeGrad(GetTaskNum(0));
            CopyOut(GetCopyIndex(0), GetTaskNum(0));
        } else {
            for (uint32_t i = 0; i < taskLoop; i++) {
                CopyIn(GetCopyIndex(i), GetTaskNum(i));
                ComputeGrad(GetTaskNum(i));
                CopyOut(GetCopyIndex(i), GetTaskNum(i));
            }
        }
    }

  private:
    __aicore__ inline void GetTilingData(const GraphSoftmaxGradTilingData *t) {
        edgeNum = t->edgeNum;
        alignTaskNum = t->alignTaskNum;
        tailNum = t->tailNum;
        nodeNum = t->nodeNum;
        taskNumPerLoop = t->taskNumPerLoop;
        taskLoop = t->taskLoop;
        blockDim = t->blockDim;
        tailCoreNum = t->tailCoreNum;
        taskNumPerCore = t->taskNumPerCore;
        ubTotalSize = t->ubTotalSize;
    }
    __aicore__ inline uint64_t GetCopyIndex(uint32_t i) {
        uint64_t ci = blockIndex * taskNumPerCore + i * taskNumPerLoop;
        if (blockIndex >= tailCoreNum) {
            ci -= (blockIndex - tailCoreNum);
        }
        return ci;
    }
    __aicore__ inline int32_t GetTaskNum(uint32_t i) {
        if (i == taskLoop - 1) {
            int32_t n = taskNumPerCore - i * taskNumPerLoop;
            if (blockIndex >= tailCoreNum) {
                n -= 1;
            }
            return n;
        }
        return taskNumPerLoop;
    }
    __aicore__ inline void CopyIn(uint64_t copyIndex, int32_t copyLength) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        int32_t idxLen = (copyLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        DataCopy(IndexTensor, indexGM[copyIndex], idxLen);
        DataCopy(SoftmaxOutputTensor, softmaxGM[copyIndex * SRC_SHAPE_DIM], copyLength * SRC_SHAPE_DIM);
        DataCopy(GradOutputTensor, gradGM[copyIndex * SRC_SHAPE_DIM], copyLength * SRC_SHAPE_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor, GradOutputTensor.GetSize());
    }
    __aicore__ inline void ComputeGrad(int32_t egTotalNum) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        Mul(SrcGradTensor, GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor.GetSize());
        Sub(SrcGradTensor, GradOutputTensor, SrcGradTensor, SrcGradTensor.GetSize());
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    __aicore__ inline void CopyOut(uint64_t copyIndex, int32_t copyLength) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(srcgradGM[copyIndex * SRC_SHAPE_DIM], SrcGradTensor, copyLength * SRC_SHAPE_DIM);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    TPipe *_pipe;
    TBuf<TPosition::VECCALC> IndexTensorBuffer, SoftmaxOutputBuffer, GradOutputBuffer, SrcGradTensorBuffer;
    LocalTensor<int32_t> IndexTensor;
    LocalTensor<float> SoftmaxOutputTensor, GradOutputTensor, SrcGradTensor;
    GlobalTensor<DTYPE_INDEX> indexGM;
    GlobalTensor<DTYPE_SOFTMAX_OUTPUT> softmaxGM;
    GlobalTensor<DTYPE_GRAD_OUTPUT> gradGM;
    GlobalTensor<DTYPE_SRC_GRAD> srcgradGM;
    int32_t edgeNum, nodeNum, alignTaskNum, tailNum, taskLoop, taskNumPerLoop, tailCoreNum;
    uint32_t blockDim, taskNumPerCore;
    uint64_t blockIndex, ubTotalSize;
};

// ═══ TILING_KEY_IS(1): General path (original baseline) ═══

class GraphSoftmaxGrad {
  public:
    __aicore__ inline GraphSoftmaxGrad() {}
    __aicore__ inline void Init(GM_ADDR index, GM_ADDR softmax_output, GM_ADDR grad_output, GM_ADDR reduce_sum,
        GM_ADDR src_grad, const GraphSoftmaxGradTilingData *tiling_data, TPipe *pipe) {
        ASSERT(GetBlockNum() != 0);
        this->blockIndex = GetBlockIdx();
        this->_pipe = pipe;
        GetTilingData(tiling_data);
        uint64_t bufferSize = taskNumPerLoop * SRC_SHAPE_DIM * FLOAT_SIZE;
        uint64_t castEdgeNum = static_cast<uint64_t>(edgeNum);
        uint64_t gmSize = castEdgeNum * SRC_SHAPE_DIM;
        indexGM.SetGlobalBuffer((__gm__ DTYPE_INDEX *)index, castEdgeNum);
        softmaxGM.SetGlobalBuffer((__gm__ DTYPE_SOFTMAX_OUTPUT *)softmax_output, gmSize);
        gradGM.SetGlobalBuffer((__gm__ DTYPE_GRAD_OUTPUT *)grad_output, gmSize);
        srcgradGM.SetGlobalBuffer((__gm__ DTYPE_SRC_GRAD *)src_grad, gmSize);
        reducesumGM.SetGlobalBuffer((__gm__ DTYPE_REDUCE_SUM *)reduce_sum, nodeNum * SRC_SHAPE_DIM);
        this->_pipe->InitBuffer(SoftmaxOutputBuffer, bufferSize);
        this->_pipe->InitBuffer(IndexTensorBuffer, taskNumPerLoop * FLOAT_SIZE);
        this->_pipe->InitBuffer(GradOutputBuffer, bufferSize);
        this->_pipe->InitBuffer(SrcGradTensorBuffer, bufferSize);
        this->_pipe->InitBuffer(TmpIndexSelectBuffer, bufferSize);
    }
    __aicore__ inline void Process() {
        AllocLocalTensors();
        if (taskLoop == 1) {
            SingleLoopComputing();
        } else {
            MultiLoopComputing();
        }
    }

  private:
    __aicore__ inline void GetTilingData(const GraphSoftmaxGradTilingData *t) {
        edgeNum = t->edgeNum;
        alignTaskNum = t->alignTaskNum;
        tailNum = t->tailNum;
        nodeNum = t->nodeNum;
        taskNumPerLoop = t->taskNumPerLoop;
        taskLoop = t->taskLoop;
        blockDim = t->blockDim;
        tailCoreNum = t->tailCoreNum;
        taskNumPerCore = t->taskNumPerCore;
        ubTotalSize = t->ubTotalSize;
    }
    __aicore__ inline uint64_t GetCopyIndex(uint32_t i) {
        uint64_t ci = blockIndex * taskNumPerCore + i * taskNumPerLoop;
        if (blockIndex >= tailCoreNum) {
            ci -= (blockIndex - tailCoreNum);
        }
        return ci;
    }
    __aicore__ inline int32_t GetTaskNum(uint32_t i) {
        if (i == taskLoop - 1) {
            int32_t n = taskNumPerCore - i * taskNumPerLoop;
            if (blockIndex >= tailCoreNum) {
                n -= 1;
            }
            return n;
        }
        return taskNumPerLoop;
    }
    __aicore__ inline void SingleLoopComputing() {
        CopyIn(GetCopyIndex(0), GetTaskNum(0));
        ReduceSum(GetTaskNum(0));
        SyncAll();
        SingleLoopIndexSelect(GetTaskNum(0));
        CopyOut(GetCopyIndex(0), GetTaskNum(0));
    }
    __aicore__ inline void MultiLoopComputing() {
        for (uint32_t i = 0; i < taskLoop; i++) {
            CopyIn(GetCopyIndex(i), GetTaskNum(i));
            ReduceSum(GetTaskNum(i));
        }
        SyncAll();
        for (uint32_t i = 0; i < taskLoop; i++) {
            uint64_t ci = GetCopyIndex(i);
            int32_t n = GetTaskNum(i);
            DataCopy(IndexTensor, indexGM[ci], (n + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM);
            MultiLoopIndexSelect(ci, n);
            CopyOut(ci, n);
        }
    }
    __aicore__ inline void AllocLocalTensors() {
        IndexTensor = IndexTensorBuffer.Get<DTYPE_INDEX>();
        SoftmaxOutputTensor = SoftmaxOutputBuffer.Get<DTYPE_SOFTMAX_OUTPUT>();
        GradOutputTensor = GradOutputBuffer.Get<DTYPE_GRAD_OUTPUT>();
        SrcGradTensor = SrcGradTensorBuffer.Get<DTYPE_SRC_GRAD>();
        TmpIndexSelectTensor = TmpIndexSelectBuffer.Get<DTYPE_SRC_GRAD>();
        Duplicate(GradOutputTensor, ZERO_FLOAT_VALUE, taskNumPerLoop * SRC_SHAPE_DIM);
        Duplicate(TmpIndexSelectTensor, ZERO_FLOAT_VALUE, taskNumPerLoop * SRC_SHAPE_DIM);
    }
    __aicore__ inline void CopyIn(uint64_t ci, int32_t n) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(IndexTensor, indexGM[ci], (n + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM);
        DataCopy(SoftmaxOutputTensor, softmaxGM[ci * SRC_SHAPE_DIM], n * SRC_SHAPE_DIM);
        DataCopy(GradOutputTensor, gradGM[ci * SRC_SHAPE_DIM], n * SRC_SHAPE_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor, GradOutputTensor.GetSize());
    }
    __aicore__ inline void ReduceSum(int32_t n) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetAtomicAdd<float>();
        for (int32_t i = 0; i < n; i++) {
            int32_t g = IndexTensor.GetValue(i);
            DataCopy(reducesumGM[g * SRC_SHAPE_DIM], GradOutputTensor[i * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    __aicore__ inline void SingleLoopIndexSelect(int32_t n) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t i = 0; i < n; i++) {
            int32_t g = IndexTensor.GetValue(i);
            DataCopy(TmpIndexSelectTensor[i * SRC_SHAPE_DIM], reducesumGM[g * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(TmpIndexSelectTensor, TmpIndexSelectTensor, SoftmaxOutputTensor, TmpIndexSelectTensor.GetSize());
        Sub(SrcGradTensor, GradOutputTensor, TmpIndexSelectTensor, TmpIndexSelectTensor.GetSize());
    }
    __aicore__ inline void MultiLoopIndexSelect(uint64_t ci, int32_t n) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (int32_t i = 0; i < n; i++) {
            int32_t g = IndexTensor.GetValue(i);
            DataCopy(TmpIndexSelectTensor[i * SRC_SHAPE_DIM], reducesumGM[g * SRC_SHAPE_DIM], SRC_SHAPE_DIM);
        }
        DataCopy(SoftmaxOutputTensor, softmaxGM[ci * SRC_SHAPE_DIM], n * SRC_SHAPE_DIM);
        DataCopy(GradOutputTensor, gradGM[ci * SRC_SHAPE_DIM], n * SRC_SHAPE_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(GradOutputTensor, SoftmaxOutputTensor, GradOutputTensor, GradOutputTensor.GetSize());
        Mul(TmpIndexSelectTensor, TmpIndexSelectTensor, SoftmaxOutputTensor, TmpIndexSelectTensor.GetSize());
        Sub(SrcGradTensor, GradOutputTensor, TmpIndexSelectTensor, TmpIndexSelectTensor.GetSize());
    }
    __aicore__ inline void CopyOut(uint64_t ci, int32_t n) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(srcgradGM[ci * SRC_SHAPE_DIM], SrcGradTensor, n * SRC_SHAPE_DIM);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    TPipe *_pipe;
    TBuf<TPosition::VECCALC> IndexTensorBuffer, SoftmaxOutputBuffer, GradOutputBuffer, SrcGradTensorBuffer,
        TmpIndexSelectBuffer;
    LocalTensor<int32_t> IndexTensor;
    LocalTensor<float> SoftmaxOutputTensor, GradOutputTensor, SrcGradTensor, TmpIndexSelectTensor;
    GlobalTensor<DTYPE_INDEX> indexGM;
    GlobalTensor<DTYPE_SOFTMAX_OUTPUT> softmaxGM;
    GlobalTensor<DTYPE_GRAD_OUTPUT> gradGM;
    GlobalTensor<DTYPE_SRC_GRAD> srcgradGM;
    GlobalTensor<DTYPE_REDUCE_SUM> reducesumGM;
    int32_t edgeNum, nodeNum, alignTaskNum, tailNum, taskLoop, taskNumPerLoop, tailCoreNum;
    uint32_t blockDim, taskNumPerCore;
    uint64_t blockIndex, ubTotalSize;
};

extern "C" __global__ __aicore__ void graph_softmax_grad(GM_ADDR index, GM_ADDR softmax_output, GM_ADDR grad_output,
    GM_ADDR reduce_sum, GM_ADDR src_grad, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        GraphSoftmaxGradScattered op;
        op.Init(index, softmax_output, grad_output, reduce_sum, src_grad, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        GraphSoftmaxGrad op;
        op.Init(index, softmax_output, grad_output, reduce_sum, src_grad, &tiling_data, &pipe);
        op.Process();
    }
}
