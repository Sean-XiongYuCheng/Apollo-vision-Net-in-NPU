"""
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""

from torch.autograd import Function

import mx_driving._C


class HungarianMatch(Function):
    @staticmethod
    def forward(ctx, cost_matrix):
        row_ind, col_ind = mx_driving._C.npu_hungarian(cost_matrix)
        return row_ind, col_ind


hungarian_match = HungarianMatch.apply
