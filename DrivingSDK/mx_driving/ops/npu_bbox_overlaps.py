"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""

from torch.autograd import Function

import mx_driving._C


class BboxOverlapsFunction(Function):
    @staticmethod
    def forward(ctx, bboxes1, bboxes2, mode, is_aligned, eps):
        if bboxes1.ndim < 2 or bboxes2.ndim < 2:
            raise ValueError(
                f"Input boxes must be at least 2-D (e.g., (N, 4) or (B, N, 4)), "
                f"but got bboxes1: {bboxes1.ndim}-D, bboxes2: {bboxes2.ndim}-D."
            )
        if bboxes1.size(-1) != 4 and bboxes1.size(0) != 0:
            raise ValueError(f"The last dimension of bboxes1 must be 4, but got {bboxes1.size(-1)}")
        if bboxes2.size(-1) != 4 and bboxes2.size(0) != 0:
            raise ValueError(f"The last dimension of bboxes2 must be 4, but got {bboxes2.size(-1)}")
        if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
            raise ValueError(
                f"Batch dimensions must match! bboxes1: {bboxes1.shape[:-2]}, bboxes2: {bboxes2.shape[:-2]}"
            )

        batch_shape = bboxes1.shape[:-2]
        m = bboxes1.size(-2)
        n = bboxes2.size(-2)

        if is_aligned and (m != n):
            raise ValueError(f"Error! M ({m}) must equal to N ({n}) when is_aligned is True.")

        if m * n == 0:
            if is_aligned:
                return bboxes1.new_empty(batch_shape + (m,))
            return bboxes1.new_empty(batch_shape + (m, n))

        mode_dict = {"iou": 0, "iof": 1, "giou": 2}
        mode_flag = mode_dict.get(mode, 0)

        bboxes1_reshaped = bboxes1.reshape(-1, m, 4)
        bboxes2_reshaped = bboxes2.reshape(-1, n, 4)
        res = mx_driving._C.npu_bbox_overlaps(bboxes1_reshaped, bboxes2_reshaped, mode_flag, is_aligned, eps)

        if is_aligned:
            target_shape = list(batch_shape) + [m]
        else:
            target_shape = list(batch_shape) + [m, n]
        res = res.view(target_shape)
        return res


def npu_bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    return BboxOverlapsFunction.apply(bboxes1, bboxes2, mode, is_aligned, eps)
