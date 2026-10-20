/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "simt_api/device_functions.h"
using namespace AscendC;

namespace {
constexpr uint32_t THREAD_NUM = 1024;
} // namespace

// Phase 1: flatten CSR (unique_indices_offset) into workspace.
// inputIdxWsp[j - startOffset] = i, using asc_stwt for cross-SyncAll visibility.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void PrepareScatterListSimt(
    __gm__ int32_t *indicesOffsetPtr, __gm__ int32_t *inputIdxWspPtr, int32_t totalInputPoints, int32_t startOffset) {
    for (int32_t i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < totalInputPoints;
         i += Simt::GetThreadNum() * Simt::GetBlockNum()) {
        int32_t startJ = indicesOffsetPtr[i];
        int32_t endJ = indicesOffsetPtr[i + 1];
        for (int32_t j = startJ; j < endJ; j++) {
            inputIdxWspPtr[j - startOffset] = i;
        }
    }
}

// Phase 2: uniform scatter over all pairs.
// threadIdx.x = channel, threadIdx.y grid-strides over pair index k.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterFeatureSimt(__gm__ float *featuresGm,
    __gm__ float *outputGm, __gm__ int32_t *inputIdxWsp, __gm__ int32_t *sortedIndicesGm, int32_t totalPairs,
    int32_t startOffset, uint32_t inChannel) {
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yStride = blockDim.y;

    for (int32_t k = Simt::GetBlockIdx() * yStride + yId; k < totalPairs; k += Simt::GetBlockNum() * yStride) {
        int32_t inputRow = inputIdxWsp[k];
        int32_t outputRow = sortedIndicesGm[k + startOffset];
        outputGm[(int64_t)outputRow * inChannel + xId] = featuresGm[(int64_t)inputRow * inChannel + xId];
    }
}

extern "C" __global__ __aicore__ void sparse_inverse_conv3d(GM_ADDR features, GM_ADDR indices,
    GM_ADDR unique_indices_offset, GM_ADDR sorted_idx_to_former_indices, GM_ADDR output_img2col, GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.totalPairs == 0) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    GlobalTensor<float> featuresGm, outputGm;
    GlobalTensor<int32_t> indicesOffsetGm, sortedIndicesGm, inputIdxWsp;
    featuresGm.SetGlobalBuffer((__gm__ float *)features);
    indicesOffsetGm.SetGlobalBuffer((__gm__ int32_t *)unique_indices_offset);
    sortedIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sorted_idx_to_former_indices);
    outputGm.SetGlobalBuffer((__gm__ float *)output_img2col);
    inputIdxWsp.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace);

    Simt::VF_CALL<PrepareScatterListSimt>(Simt::Dim3{THREAD_NUM}, (__gm__ int32_t *)indicesOffsetGm.GetPhyAddr(),
        (__gm__ int32_t *)inputIdxWsp.GetPhyAddr(), static_cast<int32_t>(tiling_data.totalInputPoints),
        static_cast<int32_t>(tiling_data.startOffset));
    SyncAll();

    Simt::VF_CALL<ScatterFeatureSimt>(Simt::Dim3{tiling_data.inChannel, THREAD_NUM / tiling_data.inChannel},
        (__gm__ float *)featuresGm.GetPhyAddr(), (__gm__ float *)outputGm.GetPhyAddr(),
        (__gm__ int32_t *)inputIdxWsp.GetPhyAddr(), (__gm__ int32_t *)sortedIndicesGm.GetPhyAddr(),
        static_cast<int32_t>(tiling_data.totalPairs), static_cast<int32_t>(tiling_data.startOffset),
        tiling_data.inChannel);
}
