// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor> npu_hungarian(const at::Tensor &cost_matrix) {
    int32_t n = cost_matrix.size(0);
    int32_t m = cost_matrix.size(1);
    auto options = at::TensorOptions().dtype(at::kInt).device(cost_matrix.device());

    if (n > m) {
        at::Tensor cost_matrix_transposed = cost_matrix.transpose(0, 1).contiguous();
        at::Tensor row_ind = at::empty({m}, options);
        at::Tensor col_ind = at::empty({m}, options);
        bool transposed = true;
        EXEC_NPU_CMD(aclnnHungarian, cost_matrix_transposed, transposed, row_ind, col_ind);
        return std::make_tuple(row_ind, col_ind);
    } else {
        at::Tensor row_ind = at::empty({n}, options);
        at::Tensor col_ind = at::empty({n}, options);
        bool transposed = false;
        EXEC_NPU_CMD(aclnnHungarian, cost_matrix, transposed, row_ind, col_ind);
        return std::make_tuple(row_ind, col_ind);
    }
}
