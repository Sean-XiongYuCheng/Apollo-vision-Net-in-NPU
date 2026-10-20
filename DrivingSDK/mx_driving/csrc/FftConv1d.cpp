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

#include <cmath>
#include <tuple>
#include <vector>

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

namespace {
constexpr int64_t kFftMinK = 64;
constexpr int64_t kMinNfftGm = 256;
constexpr int64_t kMinNradix = 8;
constexpr int64_t kMaxNradix = 4096;

inline int64_t NextPow2(int64_t n) {
    int64_t v = 2;
    while (v < n) {
        v <<= 1;
    }
    return v;
}

inline void SplitRadix(int64_t n, int64_t *n1, int64_t *n2) {
    int64_t e = 0;
    while ((int64_t{1} << e) < n) {
        ++e;
    }
    *n1 = int64_t{1} << ((e + 1) / 2);
    *n2 = int64_t{1} << (e / 2);
}

inline int64_t TableNumel(int64_t n1, int64_t n2) { return 4 * n1 * n1 + 2 * n1 * n2 + 8 * n2 * n2; }

inline int64_t ChooseNfft(int64_t L, int64_t K) {
    const int64_t n = std::max<int64_t>(kMinNfftGm, NextPow2(std::max<int64_t>(2, L + K - 1)));
    int64_t n1 = 0;
    int64_t n2 = 0;
    SplitRadix(n, &n1, &n2);
    return (n1 <= kMaxNradix && n2 >= kMinNradix) ? n : 0;
}

void EmitTableRow(float *re, float *im, int64_t k, int64_t n1, int64_t modulus) {
    const double step = -2.0 * M_PI / static_cast<double>(modulus);
    for (int64_t n = 0; n < n1; ++n) {
        const int64_t rem = (k * n) % modulus;
        const double theta = static_cast<double>(rem) * step;
        re[n] = static_cast<float>(std::cos(theta));
        im[n] = static_cast<float>(std::sin(theta));
    }
}

at::Tensor BuildFftTables(int64_t n1, int64_t n2) {
    const int64_t n = n1 * n2;
    at::Tensor table = at::empty({TableNumel(n1, n2)}, at::TensorOptions().dtype(at::kFloat));
    float *p = table.data_ptr<float>();

    float *d1r = p;
    float *d1i = p + n1 * n1;
    float *tr = p + 2 * n1 * n1;
    float *ti = tr + n;
    float *w2 = ti + n;
    float *w4 = w2 + 4 * n2 * n2;
    float *id1 = w4 + 4 * n2 * n2;

    for (int64_t k = 0; k < n1; ++k) {
        EmitTableRow(d1r + k * n1, d1i + k * n1, k, n1, n1);
        EmitTableRow(tr + k * n2, ti + k * n2, k, n2, n);
    }

    std::vector<float> d2r(n2);
    std::vector<float> d2i(n2);
    const int64_t w = 2 * n2;
    for (int64_t r = 0; r < n2; ++r) {
        EmitTableRow(d2r.data(), d2i.data(), r, n2, n2);
        for (int64_t c = 0; c < n2; ++c) {
            const float a = d2r[c];
            const float b = d2i[c];
            w2[r * w + c] = a;
            w2[r * w + n2 + c] = b;
            w2[(n2 + r) * w + c] = -b;
            w2[(n2 + r) * w + n2 + c] = a;
            w4[r * w + c] = a;
            w4[r * w + n2 + c] = -b;
            w4[(n2 + r) * w + c] = b;
            w4[(n2 + r) * w + n2 + c] = a;
        }
    }

    for (int64_t k = 0; k < n1; ++k) {
        for (int64_t j = 0; j < n1; ++j) {
            id1[k * 2 * n1 + j] = d1r[k * n1 + j];
            id1[k * 2 * n1 + n1 + j] = d1i[k * n1 + j];
        }
    }
    return table;
}
}

at::Tensor npu_fft_conv1d(at::Tensor &x, const at::Tensor &weight, bool allow_hf32) {
    TORCH_CHECK(x.dim() == 3, "fft_conv1d: x 必须是 3 维 [B,H,L]，实际 ", x.dim(), " 维");
    TORCH_CHECK(weight.dim() == 2, "fft_conv1d: weight 必须是 2 维 [H,K]，实际 ", weight.dim(), " 维");
    TORCH_CHECK(
        x.size(1) == weight.size(0), "fft_conv1d: 通道数不一致, x 的 H=", x.size(1), ", weight 的 H=", weight.size(0));
    TORCH_CHECK(weight.size(1) <= x.size(2), "fft_conv1d: 约束要求 K <= L, 实际 K=", weight.size(1), ", L=", x.size(2));

    const int64_t L = x.size(2);
    const int64_t K = weight.size(1);

    at::Tensor table;
    const int64_t nFft = (K >= kFftMinK) ? ChooseNfft(L, K) : 0;
    if (nFft > 0) {
        int64_t n1 = 0;
        int64_t n2 = 0;
        SplitRadix(nFft, &n1, &n2);
        table = BuildFftTables(n1, n2).to(x.options());
    } else {
        table = at::zeros({1}, x.options());
    }

    at::Tensor output = at::empty_like(x);
    EXEC_NPU_CMD(aclnnFftConv1d, x, weight, table, allow_hf32, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> npu_fft_conv1d_grad(
    const at::Tensor &grad, const at::Tensor &x, const at::Tensor &weight, bool allow_hf32) {
    TORCH_CHECK(x.dim() == 3, "fft_conv1d_grad: x 必须是 3 维 [B,H,L]，实际 ", x.dim(), " 维");
    TORCH_CHECK(weight.dim() == 2, "fft_conv1d_grad: weight 必须是 2 维 [H,K]，实际 ", weight.dim(), " 维");
    TORCH_CHECK(grad.sizes() == x.sizes(), "fft_conv1d_grad: grad 与 x 的 shape 必须一致（前向输出与输入同 shape）");
    TORCH_CHECK(x.size(1) == weight.size(0), "fft_conv1d_grad: 通道数不一致, x 的 H=", x.size(1),
        ", weight 的 H=", weight.size(0));
    TORCH_CHECK(
        weight.size(1) <= x.size(2), "fft_conv1d_grad: 约束要求 K <= L, 实际 K=", weight.size(1), ", L=", x.size(2));

    const int64_t L = x.size(2);
    const int64_t K = weight.size(1);

    at::Tensor table;
    const int64_t nFft = (K >= kFftMinK) ? ChooseNfft(L, K) : 0;
    if (nFft > 0) {
        int64_t n1 = 0;
        int64_t n2 = 0;
        SplitRadix(nFft, &n1, &n2);
        table = BuildFftTables(n1, n2).to(x.options());
    } else {
        table = at::zeros({1}, x.options());
    }

    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradWeight = at::zeros_like(weight);
    EXEC_NPU_CMD(aclnnFftConv1dGrad, grad, x, weight, table, allow_hf32, gradX, gradWeight);
    return std::make_tuple(gradX, gradWeight);
}
