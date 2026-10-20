// Copyright (c) 2026 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

namespace {
void check_npu(const at::Tensor &boxes_a, const at::Tensor &boxes_b) {
    TORCH_CHECK_NPU(boxes_a);
    TORCH_CHECK_NPU(boxes_b);
}
} // namespace

/**
 * @brief calculate area iou/iof/giou of boxes
 * @param boxes_a: input boxes, 2D tensor (B, M, 4)
 * @param boxes_b: input boxes, 2D tensor (B, N, 4)
 * @return res: area iou/iof/giou of boxes
 */
at::Tensor npu_bbox_overlaps(
    const at::Tensor &boxes_a, const at::Tensor &boxes_b, int32_t mode_flag, bool aligned, double margin) {
    check_npu(boxes_a, boxes_b);

    int64_t B = boxes_a.size(0);
    int64_t M = boxes_a.size(1);
    int64_t N = boxes_b.size(1);

    at::Tensor processed_a;
    at::Tensor processed_b;
    at::Tensor res;
    auto options = boxes_a.options();

    if (aligned) {
        // (B, M, 4) -> (4, B, M) -> (4, B * M)
        processed_a = boxes_a.permute({2, 0, 1}).contiguous().reshape({4, -1});
        processed_b = boxes_b.permute({2, 0, 1}).contiguous().reshape({4, -1});
        res = at::zeros({B, M}, options).reshape({-1});
        ;
    } else {
        constexpr int64_t MAX_EXPAND_ELEMENTS = 1LL << 30; // ~10 亿元素 ≈ 4GB/tensor
        int64_t expand_elements = B * M * N * 4;
        TORCH_CHECK(expand_elements <= MAX_EXPAND_ELEMENTS,
            "npu_bbox_overlaps non-aligned path would materialize B*M*N*4 = ", expand_elements, " elements (limit ",
            MAX_EXPAND_ELEMENTS, "), which risks OOM. Consider reducing M/N or using aligned mode.");

        // boxes_a: (B, M, 4) -> (B, M, 1, 4) -> (B, M, N, 4)
        at::Tensor boxes_a_expand = boxes_a.unsqueeze(2).expand({B, M, N, 4}).contiguous();
        // boxes_b: (B, N, 4) -> (B, 1, N, 4) -> (B, M, N, 4)
        at::Tensor boxes_b_expand = boxes_b.unsqueeze(1).expand({B, M, N, 4}).contiguous();

        // (B, M, N, 4) -> (4, B, M, N) -> (4, B * M * N)
        processed_a = boxes_a_expand.permute({3, 0, 1, 2}).contiguous().reshape({4, -1});
        processed_b = boxes_b_expand.permute({3, 0, 1, 2}).contiguous().reshape({4, -1});
        res = at::zeros({B, M, N}, options).reshape({-1});
    }

    check_npu(processed_a, processed_b);
    EXEC_NPU_CMD(aclnnBboxOverlaps, processed_a, processed_b, mode_flag, aligned, margin, res);
    return res;
}
