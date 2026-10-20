// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

void adamw_param_update(at::Tensor &combined_param, const at::Tensor &combined_grad, at::Tensor &exp_avg,
    at::Tensor &exp_avg_sq, at::Tensor &max_exp_avg_sq, double lr, double beta1, double beta2, double weight_decay,
    double eps, double bias_correction1, double bias_correction2, bool amsgrad) {
    TORCH_CHECK_NPU(combined_param);
    TORCH_CHECK_NPU(combined_grad);
    TORCH_CHECK_NPU(exp_avg);
    TORCH_CHECK_NPU(exp_avg_sq);

    if (amsgrad) {
        TORCH_CHECK_NPU(max_exp_avg_sq);
    }

    EXEC_NPU_CMD(aclnnAdamwParamUpdate, combined_param, combined_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1,
        beta2, weight_decay, eps, bias_correction1, bias_correction2, amsgrad);
}
