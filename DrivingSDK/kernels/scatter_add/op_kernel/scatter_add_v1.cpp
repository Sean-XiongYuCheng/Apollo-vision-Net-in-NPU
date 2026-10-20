/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "scatter_add_v1.h"

extern "C" __global__ __aicore__ void scatter_add_v1(
    GM_ADDR src,
    GM_ADDR indices,
    GM_ADDR var,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;
    
    if (TILING_KEY_IS(0)) { // TILING_KEY_NO_TAIL_FULLY_LOAD
        ScatterAddFullyLoad op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) { // TILING_KEY_NO_TAIL_MULTI_HEADS
        ScatterAddMultiHeads op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // TILING_KEY_NO_TAIL_LARGE_HEAD
        ScatterAddHeadInBatch<true> op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) { // TILING_KEY_NO_TAIL_FEW_HEADS
        ScatterAddHeadInBatch<false> op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) { // TILING_KEY_WITH_SMALL_TAIL
        ScatterAddWithTail<true> op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) { // TILING_KEY_WITH_LARGE_TAIL
        ScatterAddWithTail<false> op(src, indices, out, &tiling_data, &pipe);
        op.Process();
    }
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_add_v1_do(
    uint32_t blockDim,
    void* l2ctrl,
    void* stream,
    uint8_t* src,
    uint8_t* indices,
    uint8_t* var,
    uint8_t* out,
    uint8_t* workspace,
    uint8_t* tiling)
{
    scatter_add_v1<<<blockDim, l2ctrl, stream>>>(src, indices, var, out, workspace, tiling);
}
#endif