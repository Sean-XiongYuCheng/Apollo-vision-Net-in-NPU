// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

void npu_roiaware_pool3d_forward(const at::Tensor &rois, const at::Tensor &pts, const at::Tensor &pts_feature,
    at::Tensor &argmax, at::Tensor &pts_idx_of_voxels, at::Tensor &pooled_features, int32_t mode) {
    at::Tensor rois_cast = rois;
    at::Tensor pts_cast = pts;
    at::Tensor pts_feature_cast = pts_feature;
    at::Tensor pooled_features_cast = pooled_features;

    auto dtype = rois.dtype();
    if (dtype == at::kHalf) {
        rois_cast = rois_cast.to(at::kFloat);
        pts_cast = pts_cast.to(at::kFloat);
        pts_feature_cast = pts_feature_cast.to(at::kFloat);
        pooled_features_cast = pooled_features_cast.to(at::kFloat);
    }

    uint32_t max_pts_each_voxel = static_cast<uint32_t>(pts_idx_of_voxels.size(4));
    uint32_t outx = static_cast<uint32_t>(pts_idx_of_voxels.size(1));
    uint32_t outy = static_cast<uint32_t>(pts_idx_of_voxels.size(2));
    uint32_t outz = static_cast<uint32_t>(pts_idx_of_voxels.size(3));

    EXEC_NPU_CMD(aclnnRoiawarePool3d, rois_cast, pts_cast, pts_feature_cast, mode, max_pts_each_voxel, outx, outy, outz,
        argmax, pts_idx_of_voxels, pooled_features_cast);

    if (dtype == at::kHalf) {
        pooled_features_cast = pooled_features_cast.to(at::kHalf);
    }

    pooled_features.copy_(pooled_features_cast);
}

at::Tensor roiaware_pool3d_grad(const at::Tensor &pts_idx_of_voxels, const at::Tensor &argmax,
    const at::Tensor &grad_out, int32_t npoints, int64_t pool_method) {
    TORCH_CHECK_NPU(pts_idx_of_voxels);
    TORCH_CHECK_NPU(argmax);
    TORCH_CHECK_NPU(grad_out);

    TORCH_CHECK(
        pts_idx_of_voxels.dim() == 5, "pts_idx_of_voxels must to be a 5D Tensor, but got: ", pts_idx_of_voxels.dim());
    TORCH_CHECK(argmax.dim() == 5, "argmax has to be a 5D Tensor, but got: ", argmax.dim());
    TORCH_CHECK(grad_out.dim() == 5, "grad_out has to be a 5D Tensor, but got: ", grad_out.dim());
    TORCH_CHECK(pts_idx_of_voxels.scalar_type() == at::kInt, "pts_idx_of_voxels must have dtype int32");
    TORCH_CHECK(argmax.scalar_type() == at::kInt, "argmax must have dtype int32");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat || grad_out.scalar_type() == at::kHalf,
        "grad_out must have dtype float32 or float16");

    int32_t boxes_num = grad_out.size(0);
    int32_t out_x = grad_out.size(1);
    int32_t out_y = grad_out.size(2);
    int32_t out_z = grad_out.size(3);
    int32_t channels = grad_out.size(4);
    int32_t max_pts_per_voxel = pts_idx_of_voxels.size(4);

    TORCH_CHECK((boxes_num > 0 && out_x > 0 && out_y > 0 && out_z > 0 && channels > 0 && npoints > 0),
        "Error, some dim equals zero!\n");
    TORCH_CHECK((channels <= 2048), "channels must less equal than 2048, but got: ", channels);
    TORCH_CHECK(max_pts_per_voxel > 0 && max_pts_per_voxel <= 2048,
        "max_pts_per_voxel must be in [1, 2048], but got: ", max_pts_per_voxel);
    TORCH_CHECK(argmax.sizes() == grad_out.sizes(), "argmax and grad_out must have the same shape");
    for (int32_t dim = 0; dim < 4; ++dim) {
        TORCH_CHECK(pts_idx_of_voxels.size(dim) == grad_out.size(dim),
            "pts_idx_of_voxels and grad_out must have the same first four dimensions");
    }

    auto dtype = grad_out.dtype();
    TORCH_CHECK(pool_method == 0 || pool_method == 1, "pool_method must be 0 (max) or 1 (avg), but got: ", pool_method);

    // Both custom kernels compute in FP32. Allocate the destination directly
    // in the compute dtype so FP16 backward does not create and then convert
    // an intermediate zero tensor.
    at::Tensor grad_out_cast = dtype == at::kHalf ? grad_out.to(at::kFloat) : grad_out;
    at::Tensor grad_in = at::zeros({npoints, channels}, grad_out_cast.options());

    if (pool_method == 0) {
        // maxpool3d
        EXEC_NPU_CMD(aclnnRoiawareMaxpool3dGrad, pts_idx_of_voxels, argmax, grad_out_cast, boxes_num, out_x, out_y,
            out_z, channels, npoints, grad_in);
    } else if (pool_method == 1) {
        // avgpool3d
        EXEC_NPU_CMD(aclnnRoiawareAvgpool3dGrad, pts_idx_of_voxels, grad_out_cast, boxes_num, out_x, out_y, out_z,
            channels, npoints, max_pts_per_voxel, grad_in);
    }

    if (dtype == at::kHalf) {
        grad_in = grad_in.to(at::kHalf);
    }
    return grad_in;
}
