/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_SUM_H_
#define _DYNAMIC_SCATTER_SUM_H_

#include "dynamic_scatter_base.h"

namespace DynamicScatter {
using namespace AscendC;

template <typename T> class DynamicScatterSum : public DynamicScatterBase<T> {
  public:
    __aicore__ inline DynamicScatterSum() {}
    __aicore__ inline void Init(GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR voxel_feats, DynamicScatterTilingData *tilingData, TPipe *in_pipe) {
        this->BaseInit(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, tilingData, in_pipe);
    }

    __aicore__ inline void Process() {
        Compute();
        this->ReleaseEvent();
    }

  private:
    __aicore__ inline void Compute() {
        for (uint32_t voxelIdx = 0; voxelIdx < this->voxelNum; voxelIdx++) {
            uint32_t startPoint, pointNum;
            this->GetPointNum(voxelIdx, startPoint, pointNum);
            uint32_t idx = 0;
            bool isFirst = true;
            while (idx < pointNum) {
                LocalTensor<T> pointFeatsLocal = this->pointFeatsBuf.template AllocTensor<T>();
                uint32_t ubIdx = 0;
                while (ubIdx < this->availablePointNum && idx < pointNum) {
                    this->pointIdx = this->argsortCoorGm[startPoint].GetValue(idx);
                    DataCopy(pointFeatsLocal[ubIdx * this->featsDimAligned],
                        this->pointFeatsGm[this->pointIdx * this->featsDim], this->copyFeatParams);
                    ubIdx++;
                    idx++;
                }
                this->pointFeatsBuf.EnQue(pointFeatsLocal);
                this->pointFeatsBuf.DeQue();
                pipe_barrier(PIPE_ALL);

                BinaryReducesSum<T>(pointFeatsLocal, ubIdx, this->featsDimAligned);
                this->pointFeatsBuf.EnQue(pointFeatsLocal);
                this->pointFeatsBuf.DeQue();
                if (isFirst) {
                    this->CopyFeatsOut(voxelIdx, pointFeatsLocal, AtomicOp::NONE);
                    isFirst = false;
                } else {
                    this->CopyFeatsOut(voxelIdx, pointFeatsLocal, AtomicOp::ADD);
                }
                this->pointFeatsBuf.FreeTensor(pointFeatsLocal);
            }
        }
    }
};
} // namespace DynamicScatter
#endif // _DYNAMIC_SCATTER_SUM_H_
