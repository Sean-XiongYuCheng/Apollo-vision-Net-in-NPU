/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_H_
#define _KNN_H_
#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
//
// source 与官方 TopK 临时空间可放入 UB 时，完整 source 常驻 UB 并处理 G 个 target；
// 否则使用分 tile 的手写稳定归并。两条路径都按 target group 复用 source 数据。
template <typename T, typename U> class KnnKernel {
    enum class SelectorMode : uint32_t {
        LIBRARY = 0,
        TRUNCATED = 1,
        N4096_K8 = 2,
        N4096_K32 = 3,
    };

  public:
    __aicore__ inline KnnKernel(
        GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, GM_ADDR idx, const KnnTilingData *tiling_data, TPipe *tmpPipe) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        batch = tiling_data->batch;
        nPoint = tiling_data->nPoint;
        nSource = tiling_data->nSource;
        coreNum = tiling_data->coreNum;
        isFromKnn = tiling_data->isFromKnn;
        k = tiling_data->k;
        // VEC 指令要求 UB 起始地址 32B 对齐（fp32/int32 = 8 元素）。group 聚合缓冲里每个 target
        // 的槽位按 8 元素对齐，Muls/Adds 才能写在合法起始地址（否则 k 非 8 倍数时 core 会报未对齐）。
        kAligned = ((static_cast<uint32_t>(k) + 7) / 8) * 8;

        targetGroupSize = tiling_data->targetGroupSize;
        compNum = tiling_data->sourceTileSize;
        usedCoreNum = tiling_data->usedCoreNum;
        groupsPerBatch = tiling_data->groupsPerBatch;
        totalGroups = tiling_data->totalGroups;
        groupsPerCoreBase = tiling_data->groupsPerCoreBase;
        extraCoreGroups = tiling_data->extraCoreGroups;
        useTopK = tiling_data->useTopK;
        sourceAligned = tiling_data->sourceAligned;
        topkTmpSize = tiling_data->topkTmpSize;
        topKTilingData = tiling_data->topKTilingData;
        selectorMode = tiling_data->selectorMode;
        balanceTargetGroups = tiling_data->balanceTargetGroups;

        sort32RepeatTimes = static_cast<int32_t>(compNum / sort32Elements);
        sort32MergeRepeatTimes = static_cast<uint16_t>(compNum / (sort32Elements * 4));

        coreId = GetBlockIdx();
        InitGm(xyz, center_xyz, dist, idx, tmpPipe);
        InitBuffer();
    }
    __aicore__ inline void InitGm(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, GM_ADDR idx, TPipe *tmpPipe) {
        pipe = tmpPipe;
        const uint32_t localGroupCount = groupsPerCoreBase + (coreId < extraCoreGroups ? 1 : 0);
        startGroup = coreId * groupsPerCoreBase + (coreId < extraCoreGroups ? coreId : extraCoreGroups);
        endGroup = startGroup + localGroupCount;

        sourceGm.SetGlobalBuffer((__gm__ T *)xyz, static_cast<uint64_t>(batch) * nSource * 3);
        targetGm.SetGlobalBuffer((__gm__ T *)center_xyz, static_cast<uint64_t>(batch) * nPoint * 3);
        distGm.SetGlobalBuffer((__gm__ T *)dist, static_cast<uint64_t>(batch) * nPoint * k);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, static_cast<uint64_t>(batch) * nPoint * k);
    }

    __aicore__ inline void InitBuffer() {
        if (useTopK) {
            InitTopKBuffer();
        } else {
            InitTiledBuffer();
        }
    }

    __aicore__ inline void InitTopKBuffer() {
        const uint32_t targetBytes = ((targetGroupSize * 3 * sizeof(T) + 31) / 32) * 32;
        pipe->InitBuffer(targetUb, targetBytes);
        pipe->InitBuffer(sourceUb, sourceAligned * sizeof(T) * 3);
        pipe->InitBuffer(sqUb, sourceAligned * sizeof(T) * 3);
        pipe->InitBuffer(distUb, targetGroupSize * sourceAligned * sizeof(T));
        pipe->InitBuffer(idxUb, sourceAligned * sizeof(int32_t));
        pipe->InitBuffer(groupDistUb, targetGroupSize * kAligned * sizeof(T));
        pipe->InitBuffer(groupIdxUb, targetGroupSize * kAligned * sizeof(int32_t));
        pipe->InitBuffer(topkTmpUb, topkTmpSize);
    }

    __aicore__ inline void InitTiledBuffer() {
        pipe->InitBuffer(targetUb, maxGroupSize * 3 * sizeof(T));
        pipe->InitBuffer(sourceUb, compNum * sizeof(T) * 3);
        pipe->InitBuffer(sqUb, compNum * sizeof(T) * 3);
        // distUb / sortSrcUb 按 maxGroupSize 私有化：组内每个 target 拥有独立的距离与 Sort32 输出缓冲，
        // 消除 target 间对同一缓冲的写后读/读后写依赖，使多个 target 的排序指令可在流水线中重叠发射。
        pipe->InitBuffer(distUb, maxGroupSize * compNum * sizeof(T));
        pipe->InitBuffer(idxUb, compNum * sizeof(int32_t));
        pipe->InitBuffer(constIdxUb, compNum * sizeof(int32_t));

        pipe->InitBuffer(bestDistUb, compNum * sizeof(T));
        pipe->InitBuffer(bestIdxUb, compNum * sizeof(int32_t));

        // group 级输出聚合缓冲：组内每个 target 占用 kAligned（8 元素对齐）的槽位，供 VEC 合法写入；
        // 写回 GM 时再按各 target 的真实 k 长度逐个搬出。屏障合并到 group 级（每组 2 个而非每 target 2 个）。
        pipe->InitBuffer(groupDistUb, maxGroupSize * kAligned * sizeof(T));
        pipe->InitBuffer(groupIdxUb, maxGroupSize * kAligned * sizeof(int32_t));

        pipe->InitBuffer(sortSrcUb, maxGroupSize * compNum * sizeof(T) * 2);
        pipe->InitBuffer(sortTmp1Ub, maxGroupSize * mergeLength * sizeof(T) * 4);
        pipe->InitBuffer(sortTmp2Ub, maxGroupSize * mergeLength * sizeof(T) * 4);
    }

    __aicore__ inline void Process() {
        if (useTopK) {
            ProcessTopK();
        } else {
            ProcessTiled();
        }
    }

    __aicore__ inline void ResolveTargetGroup(
        uint32_t groupInBatch, uint32_t &targetStart, uint32_t &groupCount) const {
        if (balanceTargetGroups) {
            const uint32_t baseGroupSize = nPoint / groupsPerBatch;
            const uint32_t extraTargets = nPoint % groupsPerBatch;
            targetStart = groupInBatch * baseGroupSize + (groupInBatch < extraTargets ? groupInBatch : extraTargets);
            groupCount = baseGroupSize + (groupInBatch < extraTargets ? 1 : 0);
            return;
        }
        targetStart = groupInBatch * targetGroupSize;
        groupCount = targetGroupSize;
        if (groupCount > nPoint - targetStart) {
            groupCount = nPoint - targetStart;
        }
    }

    __aicore__ inline void ProcessTiled() {
        uint32_t loopTimes = nSource / compNum;
        uint32_t tailNum = nSource % compNum;
        const bool singleSourceTile = nSource <= compNum;

        sourceLocal = sourceUb.Get<T>();
        sqLocal = sqUb.Get<T>();
        targetLocal = targetUb.Get<T>();
        distLocal = distUb.Get<T>();
        idxLocal = idxUb.Get<int32_t>();
        bestDistLocal = bestDistUb.Get<T>();
        bestIdxLocal = bestIdxUb.Get<int32_t>();
        groupDistLocal = groupDistUb.Get<T>();
        groupIdxLocal = groupIdxUb.Get<int32_t>();

        constIdxLocal = constIdxUb.Get<int32_t>();
        CreateVecIndex(constIdxLocal, static_cast<int32_t>(0), compNum);

        sortSrcLocal = sortSrcUb.Get<T>();
        sortTmp1Local = sortTmp1Ub.Get<T>();
        sortTmp2Local = sortTmp2Ub.Get<T>();

        for (uint32_t group = startGroup; group < endGroup; group++) {
            uint32_t currentBatch = group / groupsPerBatch;
            uint32_t groupInBatch = group % groupsPerBatch;
            uint32_t targetStart;
            uint32_t groupCount;
            ResolveTargetGroup(groupInBatch, targetStart, groupCount);

            uint64_t sourceOffset = static_cast<uint64_t>(currentBatch) * nSource * 3; // B 3 N
            uint64_t targetBase = (static_cast<uint64_t>(currentBatch) * nPoint + targetStart) * 3; // B M 3

            // 一次搬入组内全部 target 的坐标（groupCount * 3 个连续 float）
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyPad(targetLocal, targetGm[targetBase],
                {1, static_cast<uint32_t>(groupCount * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            // GetValue 读取前需保证坐标已搬入 UB；沿用原实现在 GetValue 前使用 MTE2_V 屏障的方式
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            // A4：组内 target 坐标一次性读入标量缓存，供本组所有 source tile 复用，
            //     避免每个 tile 都对每个 target 重复 GetValue（削减 profiling 中 16.7% 的标量开销）。
            for (uint32_t g = 0; g < groupCount; g++) {
                targetXCache[g] = targetLocal.GetValue(g * 3);
                targetYCache[g] = targetLocal.GetValue(g * 3 + 1);
                targetZCache[g] = targetLocal.GetValue(g * 3 + 2);
            }

            // 首轮归并只读取 sortTmp1 的 k 个 seed 对；每个 source tile 会覆盖
            // sort32MergeRepeatTimes 个新队列。仅 256-tile 留有一个未覆盖队列，
            // 该占位队列在两个 ping-pong buffer 中都需初始化为最小值。
            if (!singleSourceTile) {
                for (uint32_t g = 0; g < groupCount; g++) {
                    uint32_t base = g * mergeLength * 4;
                    Duplicate(sortTmp1Local[base], minFloatValue, static_cast<uint32_t>(k) * 2);
                    for (uint32_t slot = sort32MergeRepeatTimes + 1; slot < 4; slot++) {
                        Duplicate(
                            sortTmp1Local[base + slot * mergeLength], minFloatValue, static_cast<uint32_t>(k) * 2);
                        Duplicate(
                            sortTmp2Local[base + slot * mergeLength], minFloatValue, static_cast<uint32_t>(k) * 2);
                    }
                }
            }

            for (uint32_t currentLoop = 0; currentLoop < loopTimes; currentLoop++) {
                ComputeTile(currentLoop, compNum, sourceOffset, groupCount);
            }
            if (tailNum > 0) {
                ComputeTile(loopTimes, tailNum, sourceOffset, groupCount);
            }

            uint64_t groupOutBase = (static_cast<uint64_t>(currentBatch) * nPoint + targetStart) * k;
            CopyOutGroup(groupOutBase, groupCount);
        }
    }

    // 完整 source 可驻留 UB 时，每个 target group 构造 G x inner 距离矩阵并一次调用稳定 TopK。
    __aicore__ inline void ProcessTopK() {
        sourceLocal = sourceUb.Get<T>();
        sqLocal = sqUb.Get<T>();
        targetLocal = targetUb.Get<T>();
        distLocal = distUb.Get<T>();
        idxLocal = idxUb.Get<int32_t>();
        groupDistLocal = groupDistUb.Get<T>();
        groupIdxLocal = groupIdxUb.Get<int32_t>();
        topkTmpLocal = topkTmpUb.Get<uint8_t>();

        // Normal 模式所有 outter 行共享一条原始索引；padding 索引自然递增且不会影响前 k。
        CreateVecIndex(idxLocal, static_cast<int32_t>(0), sourceAligned);
        uint32_t cachedBatch = 0xFFFFFFFFU;
        for (uint32_t group = startGroup; group < endGroup; group++) {
            // 等待上一 group 的 MTE3 完成，避免覆盖 TopK 输出聚合缓冲。
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

            const uint32_t currentBatch = group / groupsPerBatch;
            const uint32_t groupInBatch = group % groupsPerBatch;
            uint32_t targetStart;
            uint32_t groupCount;
            ResolveTargetGroup(groupInBatch, targetStart, groupCount);
            if (currentBatch != cachedBatch) {
                LoadFullSource(currentBatch);
                cachedBatch = currentBatch;
            }
            LoadTargetGroup(currentBatch, targetStart, groupCount);
            ComputeTopKDistances();

            if (selectorMode == static_cast<uint32_t>(SelectorMode::N4096_K32)) {
                ProcessPartialTopK32();
            } else if (selectorMode == static_cast<uint32_t>(SelectorMode::N4096_K8)) {
                ProcessPartialTopK();
            } else if (selectorMode == static_cast<uint32_t>(SelectorMode::TRUNCATED)) {
                ProcessTruncatedTopK();
            } else {
                TopKInfo topKInfo;
                topKInfo.outter = targetGroupSize;
                topKInfo.inner = sourceAligned;
                topKInfo.n = nSource;
                LocalTensor<bool> finishLocal;
                // MERGE_SORT 的等值稳定性保证相同距离时原始位置较小者优先；false 表示取最小平方距离。
                TopK<T, true, false, false, AscendC::TopKMode::TOPK_NORMAL>(groupDistLocal, groupIdxLocal, distLocal,
                    idxLocal, finishLocal, topkTmpLocal, k, topKTilingData, topKInfo, false);
            }

            const uint64_t groupOutBase =
                (static_cast<uint64_t>(currentBatch) * nPoint + targetStart) * static_cast<uint32_t>(k);
            CopyOutTopK(groupOutBase, groupCount);
        }
    }

    // N=4096、k<=8 的低内存专用路径：每个 32 点有序段只保留前 8 个候选，再做 4 路稳定归并。
    // source chunk 和 merge queue 始终按原始索引递增排列，因此相同距离仍由较小原始索引优先。
    __aicore__ inline void ProcessPartialTopK() {
        constexpr uint32_t proposalStride = 64; // 32 proposals, each proposal is value + index.
        const uint32_t sortFloatCount = sourceAligned * 2;
        const uint32_t level1QueueCount = sourceAligned / 128;
        const uint32_t level2QueueCount = sourceAligned / 512;
        const uint32_t level3QueueCount = sourceAligned / 2048;

        LocalTensor<T> partialTmp = topkTmpUb.Get<T>();
        LocalTensor<T> sortProposal = partialTmp;
        LocalTensor<T> level1 = partialTmp[sortFloatCount];
        LocalTensor<T> level2 = level1[level1QueueCount * proposalStride];
        LocalTensor<T> level3 = level2[level2QueueCount * proposalStride];
        LocalTensor<T> finalProposal = level3[level3QueueCount * proposalStride];
        LocalTensor<T> extractDist = finalProposal[proposalStride];
        LocalTensor<int32_t> extractIdx = extractDist[32].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> sourceIdx = idxLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> extractIdxU32 = extractIdx.template ReinterpretCast<uint32_t>();
        const uint16_t fourQueueCounts[4] = {8, 8, 8, 8};
        const uint16_t twoQueueCounts[4] = {8, 8, 0, 0};

        for (uint32_t g = 0; g < targetGroupSize; g++) {
            LocalTensor<T> distG = distLocal[g * sourceAligned];
            // Sort32 is descending; negate once and restore only the selected distances after Extract.
            Muls(distG, distG, static_cast<T>(-1.0f), sourceAligned);
            Sort32(sortProposal, distG, sourceIdx, static_cast<int32_t>(sourceAligned / 32));

            for (uint32_t group = 0; group < level1QueueCount; group++) {
                const uint32_t srcBase = group * 4 * proposalStride;
                AscendC::MrgSortSrcList sourceList =
                    AscendC::MrgSortSrcList(sortProposal[srcBase], sortProposal[srcBase + proposalStride],
                        sortProposal[srcBase + proposalStride * 2], sortProposal[srcBase + proposalStride * 3]);
                MrgSort<T>(level1[group * proposalStride], sourceList, {fourQueueCounts, false, 0b1111, 1});
            }
            for (uint32_t group = 0; group < level2QueueCount; group++) {
                const uint32_t srcBase = group * 4 * proposalStride;
                AscendC::MrgSortSrcList sourceList =
                    AscendC::MrgSortSrcList(level1[srcBase], level1[srcBase + proposalStride],
                        level1[srcBase + proposalStride * 2], level1[srcBase + proposalStride * 3]);
                MrgSort<T>(level2[group * proposalStride], sourceList, {fourQueueCounts, false, 0b1111, 1});
            }
            LocalTensor<T> finalSource = level2;
            if (level3QueueCount > 0) {
                for (uint32_t group = 0; group < level3QueueCount; group++) {
                    const uint32_t srcBase = group * 4 * proposalStride;
                    AscendC::MrgSortSrcList sourceList =
                        AscendC::MrgSortSrcList(level2[srcBase], level2[srcBase + proposalStride],
                            level2[srcBase + proposalStride * 2], level2[srcBase + proposalStride * 3]);
                    MrgSort<T>(level3[group * proposalStride], sourceList, {fourQueueCounts, false, 0b1111, 1});
                }
                finalSource = level3;
            }
            AscendC::MrgSortSrcList finalList =
                AscendC::MrgSortSrcList(finalSource, finalSource[proposalStride], finalSource, finalSource);
            MrgSort<T>(finalProposal, finalList, {twoQueueCounts, false, 0b0011, 1});
            Extract(extractDist, extractIdxU32, finalProposal, 1);
            Muls(groupDistLocal[g * kAligned], extractDist, static_cast<T>(-1.0f), k);
            Adds(groupIdxLocal[g * kAligned], extractIdx, 0, k);
        }
    }

    // N<=1024 的通用稳定截断选择器。每层队列只向下一层暴露前 k 个候选，
    // 但为 MrgSort 的完整四路输出保留 4*k proposal 空间。尾队列的 counts
    // 由实际 source 数量推导，避免 k>32 时读取 Sort32 的 padding proposal。
    __aicore__ inline void ProcessTruncatedTopK() {
        constexpr uint32_t sortQueueStride = 64; // 32 proposals, each value + index.
        const uint32_t queueStride = static_cast<uint32_t>(k) * 8;
        const uint32_t extractAligned = (static_cast<uint32_t>(k) + 31) / 32 * 32;
        const int32_t extractRepeatTimes = static_cast<int32_t>(extractAligned / 32);
        const uint32_t sortFloatCount = sourceAligned * 2;

        LocalTensor<T> partialTmp = topkTmpUb.Get<T>();
        LocalTensor<T> sortProposal = partialTmp;
        LocalTensor<uint32_t> sourceIdx = idxLocal.template ReinterpretCast<uint32_t>();

        for (uint32_t g = 0; g < targetGroupSize; g++) {
            LocalTensor<T> distG = distLocal[g * sourceAligned];
            Muls(distG, distG, static_cast<T>(-1.0f), sourceAligned);
            Sort32(sortProposal, distG, sourceIdx, static_cast<int32_t>(sourceAligned / 32));

            LocalTensor<T> currentSource = sortProposal;
            uint32_t currentStride = sortQueueStride;
            uint32_t queueCount = sourceAligned / 32;
            uint32_t inputSpan = 32;
            uint32_t nextOffset = sortFloatCount;
            while (queueCount > 1) {
                const uint32_t nextQueueCount = (queueCount + 3) / 4;
                LocalTensor<T> nextSource = partialTmp[nextOffset];
                for (uint32_t group = 0; group < nextQueueCount; group++) {
                    const uint32_t firstQueue = group * 4;
                    const uint32_t validQueueCount = queueCount - firstQueue < 4 ? queueCount - firstQueue : 4;
                    const uint32_t queue0 = firstQueue;
                    const uint32_t queue1 = validQueueCount > 1 ? firstQueue + 1 : firstQueue;
                    const uint32_t queue2 = validQueueCount > 2 ? firstQueue + 2 : firstQueue;
                    const uint32_t queue3 = validQueueCount > 3 ? firstQueue + 3 : firstQueue;
                    const uint32_t sourceStart0 = queue0 * inputSpan;
                    const uint32_t sourceStart1 = queue1 * inputSpan;
                    const uint32_t sourceStart2 = queue2 * inputSpan;
                    const uint32_t sourceStart3 = queue3 * inputSpan;
                    const uint32_t valid0 = sourceStart0 < nSource
                        ? ((nSource - sourceStart0 < inputSpan) ? nSource - sourceStart0 : inputSpan)
                        : 0;
                    const uint32_t valid1 = validQueueCount > 1
                        ? ((nSource - sourceStart1 < inputSpan) ? nSource - sourceStart1 : inputSpan)
                        : 0;
                    const uint32_t valid2 = validQueueCount > 2
                        ? ((nSource - sourceStart2 < inputSpan) ? nSource - sourceStart2 : inputSpan)
                        : 0;
                    const uint32_t valid3 = validQueueCount > 3
                        ? ((nSource - sourceStart3 < inputSpan) ? nSource - sourceStart3 : inputSpan)
                        : 0;
                    const uint16_t counts[4] = {
                        static_cast<uint16_t>(valid0 < static_cast<uint32_t>(k) ? valid0 : static_cast<uint32_t>(k)),
                        static_cast<uint16_t>(valid1 < static_cast<uint32_t>(k) ? valid1 : static_cast<uint32_t>(k)),
                        static_cast<uint16_t>(valid2 < static_cast<uint32_t>(k) ? valid2 : static_cast<uint32_t>(k)),
                        static_cast<uint16_t>(valid3 < static_cast<uint32_t>(k) ? valid3 : static_cast<uint32_t>(k)),
                    };
                    AscendC::MrgSortSrcList sourceList = AscendC::MrgSortSrcList(currentSource[queue0 * currentStride],
                        currentSource[queue1 * currentStride], currentSource[queue2 * currentStride],
                        currentSource[queue3 * currentStride]);
                    MrgSort<T>(nextSource[group * queueStride], sourceList,
                        {counts, false, static_cast<uint8_t>((1U << validQueueCount) - 1U), 1});
                }
                currentSource = nextSource;
                currentStride = queueStride;
                queueCount = nextQueueCount;
                inputSpan *= 4;
                nextOffset += nextQueueCount * queueStride;
            }

            LocalTensor<T> extractDist = partialTmp[nextOffset];
            LocalTensor<int32_t> extractIdx = extractDist[extractAligned].template ReinterpretCast<int32_t>();
            LocalTensor<uint32_t> extractIdxU32 = extractIdx.template ReinterpretCast<uint32_t>();
            Extract(extractDist, extractIdxU32, currentSource, extractRepeatTimes);
            Muls(groupDistLocal[g * kAligned], extractDist, static_cast<T>(-1.0f), k);
            Adds(groupIdxLocal[g * kAligned], extractIdx, 0, k);
        }
    }

    // N=4096、8<k<=32：每个归并队列保留前 32 个候选。一级归并仅保留四个暂存槽，
    // 产生一个二级队列后立即复用，避免一次性保存 32 个大队列而超出 UB。
    __aicore__ inline void ProcessPartialTopK32() {
        constexpr uint32_t proposalStride = 64; // 32 proposals, each proposal is value + index.
        constexpr uint32_t mergeQueueStride = 256; // Up to 128 merged proposals.
        constexpr uint32_t level1ScratchCount = 4;
        const uint32_t sortFloatCount = sourceAligned * 2;
        const uint32_t level2QueueCount = sourceAligned / 512;
        const uint32_t level3QueueCount = sourceAligned / 2048;

        LocalTensor<T> partialTmp = topkTmpUb.Get<T>();
        LocalTensor<T> sortProposal = partialTmp;
        LocalTensor<T> level1Scratch = partialTmp[sortFloatCount];
        LocalTensor<T> level2 = level1Scratch[level1ScratchCount * mergeQueueStride];
        LocalTensor<T> level3 = level2[level2QueueCount * mergeQueueStride];
        LocalTensor<T> finalProposal = level3[level3QueueCount * mergeQueueStride];
        LocalTensor<T> extractDist = finalProposal[mergeQueueStride];
        LocalTensor<int32_t> extractIdx = extractDist[32].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> sourceIdx = idxLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> extractIdxU32 = extractIdx.template ReinterpretCast<uint32_t>();
        const uint16_t fourQueueCounts[4] = {32, 32, 32, 32};
        const uint16_t twoQueueCounts[4] = {32, 32, 0, 0};

        for (uint32_t g = 0; g < targetGroupSize; g++) {
            LocalTensor<T> distG = distLocal[g * sourceAligned];
            Muls(distG, distG, static_cast<T>(-1.0f), sourceAligned);
            Sort32(sortProposal, distG, sourceIdx, static_cast<int32_t>(sourceAligned / 32));

            for (uint32_t level2Group = 0; level2Group < level2QueueCount; level2Group++) {
                for (uint32_t scratch = 0; scratch < level1ScratchCount; scratch++) {
                    const uint32_t level1Group = level2Group * level1ScratchCount + scratch;
                    const uint32_t srcBase = level1Group * 4 * proposalStride;
                    AscendC::MrgSortSrcList sourceList =
                        AscendC::MrgSortSrcList(sortProposal[srcBase], sortProposal[srcBase + proposalStride],
                            sortProposal[srcBase + proposalStride * 2], sortProposal[srcBase + proposalStride * 3]);
                    MrgSort<T>(
                        level1Scratch[scratch * mergeQueueStride], sourceList, {fourQueueCounts, false, 0b1111, 1});
                }
                AscendC::MrgSortSrcList level1List =
                    AscendC::MrgSortSrcList(level1Scratch, level1Scratch[mergeQueueStride],
                        level1Scratch[mergeQueueStride * 2], level1Scratch[mergeQueueStride * 3]);
                MrgSort<T>(level2[level2Group * mergeQueueStride], level1List, {fourQueueCounts, false, 0b1111, 1});
            }

            for (uint32_t level3Group = 0; level3Group < level3QueueCount; level3Group++) {
                const uint32_t srcBase = level3Group * 4 * mergeQueueStride;
                AscendC::MrgSortSrcList sourceList =
                    AscendC::MrgSortSrcList(level2[srcBase], level2[srcBase + mergeQueueStride],
                        level2[srcBase + mergeQueueStride * 2], level2[srcBase + mergeQueueStride * 3]);
                MrgSort<T>(level3[level3Group * mergeQueueStride], sourceList, {fourQueueCounts, false, 0b1111, 1});
            }

            AscendC::MrgSortSrcList finalList =
                AscendC::MrgSortSrcList(level3, level3[mergeQueueStride], level3, level3);
            MrgSort<T>(finalProposal, finalList, {twoQueueCounts, false, 0b0011, 1});
            Extract(extractDist, extractIdxU32, finalProposal, 1);
            Muls(groupDistLocal[g * kAligned], extractDist, static_cast<T>(-1.0f), k);
            Adds(groupIdxLocal[g * kAligned], extractIdx, 0, k);
        }
    }

    __aicore__ inline void LoadFullSource(uint32_t currentBatch) {
        const uint64_t sourceOffset = static_cast<uint64_t>(currentBatch) * nSource * 3;
        const uint32_t copyBytes = nSource * sizeof(T);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopyPad(sourceLocal, sourceGm[sourceOffset], {1, copyBytes, 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(
            sourceLocal[sourceAligned], sourceGm[sourceOffset + nSource], {1, copyBytes, 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sourceLocal[sourceAligned * 2], sourceGm[sourceOffset + static_cast<uint64_t>(nSource) * 2],
            {1, copyBytes, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void LoadTargetGroup(uint32_t currentBatch, uint32_t targetStart, uint32_t groupCount) {
        const uint64_t targetBase = (static_cast<uint64_t>(currentBatch) * nPoint + targetStart) * 3;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(targetLocal, targetGm[targetBase], {1, static_cast<uint32_t>(groupCount * 3 * sizeof(T)), 0, 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t g = 0; g < groupCount; g++) {
            targetXCache[g] = targetLocal.GetValue(g * 3);
            targetYCache[g] = targetLocal.GetValue(g * 3 + 1);
            targetZCache[g] = targetLocal.GetValue(g * 3 + 2);
        }
        // TopK tiling 的 outter 固定为 G；组尾复制最后一条有效 target，结果不会写回 GM。
        for (uint32_t g = groupCount; g < targetGroupSize; g++) {
            targetXCache[g] = targetXCache[groupCount - 1];
            targetYCache[g] = targetYCache[groupCount - 1];
            targetZCache[g] = targetZCache[groupCount - 1];
        }
    }

    __aicore__ inline void ComputeTopKDistances() {
        const bool hasPadding = nSource != sourceAligned;
        for (uint32_t g = 0; g < targetGroupSize; g++) {
            LocalTensor<T> distG = distLocal[g * sourceAligned];
            // 对齐场景的每个位置都会被真实距离覆盖，无需预填最大值。
            if (hasPadding) {
                Duplicate(distG, maxFloatValue, sourceAligned);
            }
            Adds<T>(sqLocal, sourceLocal, static_cast<T>(-targetXCache[g]), nSource);
            Adds<T>(sqLocal[sourceAligned], sourceLocal[sourceAligned], static_cast<T>(-targetYCache[g]), nSource);
            Adds<T>(
                sqLocal[sourceAligned * 2], sourceLocal[sourceAligned * 2], static_cast<T>(-targetZCache[g]), nSource);
            if (hasPadding) {
                Mul<T>(sqLocal, sqLocal, sqLocal, nSource);
                Mul<T>(sqLocal[sourceAligned], sqLocal[sourceAligned], sqLocal[sourceAligned], nSource);
                Mul<T>(sqLocal[sourceAligned * 2], sqLocal[sourceAligned * 2], sqLocal[sourceAligned * 2], nSource);
            } else {
                Mul<T>(sqLocal, sqLocal, sqLocal, nSource * 3);
            }
            Add<T>(distG, sqLocal, sqLocal[sourceAligned], nSource);
            Add<T>(distG, distG, sqLocal[sourceAligned * 2], nSource);
            if (isFromKnn) {
                Mins<T>(distG, distG, static_cast<T>(1e10f), nSource);
            }
        }
    }

    __aicore__ inline void CopyOutTopK(uint64_t groupBase, uint32_t groupCount) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (kAligned == static_cast<uint32_t>(k)) {
            DataCopyPad(distGm[groupBase], groupDistLocal,
                {1, static_cast<uint32_t>(groupCount * static_cast<uint32_t>(k) * sizeof(T)), 0, 0, 0});
            DataCopyPad(idxGm[groupBase], groupIdxLocal,
                {1, static_cast<uint32_t>(groupCount * static_cast<uint32_t>(k) * sizeof(int32_t)), 0, 0, 0});
            return;
        }
        for (uint32_t g = 0; g < groupCount; g++) {
            const uint64_t off = groupBase + static_cast<uint64_t>(g) * static_cast<uint32_t>(k);
            DataCopyPad(distGm[off], groupDistLocal[g * kAligned], {1, static_cast<uint32_t>(k * sizeof(T)), 0, 0, 0});
            DataCopyPad(
                idxGm[off], groupIdxLocal[g * kAligned], {1, static_cast<uint32_t>(k * sizeof(int32_t)), 0, 0, 0});
        }
    }

    // 搬入一个 source tile（组内所有 target 复用），再对组内每个 target 计算距离并维护 top-k。
    // 排序阶段按 Sort32 / 一级归并 / 二级归并分层批量下发，隐藏排序指令延迟（详见各阶段注释）。
    __aicore__ inline void ComputeTile(
        uint32_t currentLoop, uint32_t copySize, uint64_t sourceOffset, uint32_t groupCount) {
        uint32_t copyInLength = static_cast<uint32_t>(copySize * sizeof(T));
        uint32_t loopOffset = currentLoop * compNum;
        const bool singleSourceTile = nSource <= compNum;
        const bool directSort32 = singleSourceTile && (nSource >= 15) && (nSource <= 32);
        const int32_t activeSort32RepeatTimes = directSort32 ? 1 : (nSource == 64 ? 2 : sort32RepeatTimes);
        const uint8_t firstMergeValidMask = nSource == 64 ? static_cast<uint8_t>(0b0011) : static_cast<uint8_t>(0b1111);

        // 等待上一 tile 的 Vector 计算完成后再覆盖 source buffer
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopyPad(sourceLocal, sourceGm[sourceOffset + loopOffset], {1, copyInLength, 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sourceLocal[compNum], sourceGm[sourceOffset + loopOffset + nSource], {1, copyInLength, 0, 0, 0},
            {false, 0, 0, 0});
        DataCopyPad(sourceLocal[compNum * 2], sourceGm[sourceOffset + loopOffset + static_cast<uint64_t>(nSource) * 2],
            {1, copyInLength, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        // A2：源点全局下标只依赖当前 tile（loopOffset），对组内所有 target 相同，每个 tile 只计算一次。
        Adds(idxLocal, constIdxLocal, static_cast<int32_t>(loopOffset), nSource == 64 ? copySize : compNum);

        // 阶段 1：为组内每个 target 计算距离，写入各自私有的 distLocal[g] 段。
        for (uint32_t g = 0; g < groupCount; g++) {
            ComputeDist(g, copySize);
        }

        // 阶段 2/3/4：将 Sort32、一级 4 路归并、二级归并分别对组内所有 target 批量下发。
        // 组内各 target 完全独立且使用私有缓冲，分层下发使得下一 target 的排序指令可在上一 target
        // 排序结果尚在流水线中时提前发射，从而掩盖 Sort32/MrgSort 的指令延迟（此前 78% 空闲的主因）。
        // 每个 target 的运算序列与原实现逐指令一致，top-k 结果保持 bit-exact。
        AscendC::LocalTensor<uint32_t> interpreIdxInTensor = idxLocal.ReinterpretCast<uint32_t>();
        for (uint32_t g = 0; g < groupCount; g++) {
            Sort32(sortSrcLocal[g * compNum * 2], distLocal[g * compNum], interpreIdxInTensor, activeSort32RepeatTimes);
        }
        if (directSort32) {
            return;
        }

        const uint16_t sortCountList[4] = {
            (uint16_t)sort32Elements, (uint16_t)sort32Elements, (uint16_t)sort32Elements, (uint16_t)sort32Elements};
        const uint16_t mergeCountList[4] = {(uint16_t)k, (uint16_t)k, (uint16_t)k, (uint16_t)k};
        bool even = ((currentLoop % 2) == 0);
        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t base = g * mergeLength * 4;
            uint32_t srcBase = g * compNum * 2;
            AscendC::MrgSortSrcList sortList =
                AscendC::MrgSortSrcList(sortSrcLocal[srcBase], sortSrcLocal[srcBase + sort32Offset],
                    sortSrcLocal[srcBase + sort32Offset * 2], sortSrcLocal[srcBase + sort32Offset * 3]);
            LocalTensor<T> dstMerge = singleSourceTile
                ? sortTmp1Local[base]
                : (even ? sortTmp1Local[base + mergeLength] : sortTmp2Local[base + mergeLength]);
            MrgSort<T>(dstMerge, sortList, {sortCountList, false, firstMergeValidMask, sort32MergeRepeatTimes});
        }
        for (uint32_t g = 0; g < groupCount; g++) {
            if (singleSourceTile && (sort32MergeRepeatTimes == 1)) {
                continue;
            }
            uint32_t base = g * mergeLength * 4;
            LocalTensor<T> src =
                singleSourceTile ? sortTmp1Local[base] : (even ? sortTmp1Local[base] : sortTmp2Local[base]);
            LocalTensor<T> dst =
                singleSourceTile ? sortTmp2Local[base] : (even ? sortTmp2Local[base] : sortTmp1Local[base]);
            AscendC::MrgSortSrcList mergeList =
                AscendC::MrgSortSrcList(src, src[mergeLength], src[mergeLength * 2], src[mergeLength * 3]);
            const uint8_t validMask = singleSourceTile ? static_cast<uint8_t>((1U << sort32MergeRepeatTimes) - 1U)
                                                       : static_cast<uint8_t>(0b1111);
            MrgSort<T>(dst, mergeList, {mergeCountList, false, validMask, 1});
        }
    }

    // 计算组内第 g 个 target 到当前 source tile 各点的平方距离（取负后写入私有 distLocal[g] 段）。
    // 运算序列与原实现完全一致：Adds(-t)*3 -> Mul(平方) -> Add*2 -> (isFromKnn ? Mins) -> Muls(-1)。
    __aicore__ inline void ComputeDist(uint32_t g, uint32_t copySize) {
        T targetX = targetXCache[g];
        T targetY = targetYCache[g];
        T targetZ = targetZCache[g];
        LocalTensor<T> distG = distLocal[g * compNum];
        const bool compactSort32 = (nSource >= 15) && (nSource <= 32);
        const bool compactN64 = nSource == 64;
        const uint32_t activeElementCount = compactSort32 ? 32 : (compactN64 ? copySize : compNum);

        Adds<T>(sqLocal, sourceLocal, static_cast<T>(-targetX), copySize);
        Adds<T>(sqLocal[compNum], sourceLocal[compNum], static_cast<T>(-targetY), copySize);
        Adds<T>(sqLocal[compNum * 2], sourceLocal[compNum * 2], static_cast<T>(-targetZ), copySize);
        if (compactSort32 || compactN64) {
            Mul<T>(sqLocal, sqLocal, sqLocal, copySize);
            Mul<T>(sqLocal[compNum], sqLocal[compNum], sqLocal[compNum], copySize);
            Mul<T>(sqLocal[compNum * 2], sqLocal[compNum * 2], sqLocal[compNum * 2], copySize);
        } else {
            Mul<T>(sqLocal, sqLocal, sqLocal, compNum * 3);
        }

        if (!compactN64 && (copySize < activeElementCount)) {
            Duplicate(distG, maxFloatValue, activeElementCount);
        }
        Add<T>(distG, sqLocal, sqLocal[compNum], copySize);
        Add<T>(distG, distG, sqLocal[compNum * 2], copySize);

        if (isFromKnn) {
            Mins<T>(distG, distG, static_cast<T>(1e10f), activeElementCount);
        }
        Muls(distG, distG, -1.0f, activeElementCount);
    }

    // 组级写回：组内 groupCount 个 target 的 top-k 结果在 GM 中连续（targetStart..+groupCount-1）。
    // 先对每个 target 逐个 Extract、取负并写入 8 元素对齐槽位的 group 聚合缓冲，再逐 target 搬出到 GM。
    // 关键收益：同步屏障从“每 target 2 个（排空流水）”降为“每组 2 个”，G 个小搬运在两屏障间连续下发、
    // 互不等待，摊薄此前占满约 78% 空闲的写回开销。每个 target 的 Extract/取负与原实现逐指令一致，bit-exact。
    __aicore__ inline void CopyOutGroup(uint64_t groupBase, uint32_t groupCount) {
        // 一次屏障：确保上一 group 的写回（MTE3）完成后，再用 Vector 覆盖 group 聚合缓冲
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        AscendC::LocalTensor<uint32_t> interpreIdxOutTensor = bestIdxLocal.ReinterpretCast<uint32_t>();
        bool resultInTmp2 = ((((nSource - 1) / compNum + 1) % 2) == 1);
        const bool singleSourceTile = nSource <= compNum;
        const bool directSort32 = singleSourceTile && (nSource >= 15) && (nSource <= 32);
        const bool directFirstMerge = singleSourceTile && (sort32MergeRepeatTimes == 1);
        const int32_t extractRepeatTimes = static_cast<int32_t>((static_cast<uint32_t>(k) + 31) / 32);
        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t base = g * mergeLength * 4;
            LocalTensor<T> finalBuf = directSort32
                ? sortSrcLocal[g * compNum * 2]
                : (directFirstMerge ? sortTmp1Local[base] : (resultInTmp2 ? sortTmp2Local[base] : sortTmp1Local[base]));
            Extract(bestDistLocal, interpreIdxOutTensor, finalBuf, extractRepeatTimes);
            // 取负恢复真实距离并写入 8 元素对齐的槽位（VEC 写入地址必须 32B 对齐）；idx 直接整型拷贝
            Muls(groupDistLocal[g * kAligned], bestDistLocal, -1.0f, k);
            Adds(groupIdxLocal[g * kAligned], bestIdxLocal, 0, k);
        }

        // 一次屏障：确保 Extract/Muls/Adds 完成后再写回 GM；随后 G 个小搬运连续下发（无逐 target 屏障）
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (kAligned == static_cast<uint32_t>(k)) {
            DataCopyPad(distGm[groupBase], groupDistLocal,
                {1, static_cast<uint32_t>(groupCount * static_cast<uint32_t>(k) * sizeof(T)), 0, 0, 0});
            DataCopyPad(idxGm[groupBase], groupIdxLocal,
                {1, static_cast<uint32_t>(groupCount * static_cast<uint32_t>(k) * sizeof(int32_t)), 0, 0, 0});
            return;
        }
        for (uint32_t g = 0; g < groupCount; g++) {
            uint64_t off = groupBase + static_cast<uint64_t>(g) * static_cast<uint32_t>(k);
            DataCopyPad(distGm[off], groupDistLocal[g * kAligned], {1, static_cast<uint32_t>(k * sizeof(T)), 0, 0, 0});
            DataCopyPad(
                idxGm[off], groupIdxLocal[g * kAligned], {1, static_cast<uint32_t>(k * sizeof(int32_t)), 0, 0, 0});
        }
    }

  public:
    TPipe *pipe;
    GlobalTensor<T> sourceGm, targetGm, distGm;
    GlobalTensor<int32_t> idxGm;
    TBuf<TPosition::VECCALC> sourceUb, sqUb, targetUb, distUb, idxUb, bestDistUb, bestIdxUb, constIdxUb, sortTmp1Ub,
        sortTmp2Ub, sortSrcUb, groupDistUb, groupIdxUb, topkTmpUb;
    LocalTensor<T> sourceLocal, sqLocal, targetLocal, distLocal, bestDistLocal, sortTmp1Local, sortTmp2Local,
        sortSrcLocal, groupDistLocal;
    LocalTensor<int32_t> bestIdxLocal, idxLocal, constIdxLocal, groupIdxLocal;
    LocalTensor<uint8_t> topkTmpLocal;
    uint32_t coreId;
    uint32_t startGroup, endGroup;

    // 手写稳定归并路径的组内 target 上限，决定其 sort 累加缓冲的 UB 占用。
    static constexpr uint32_t maxGroupSize = 8;
    static constexpr uint32_t maxTopKGroupSize = 16;
    // A4：组内 target 坐标标量缓存（每组读入一次，各 source tile 复用）
    T targetXCache[maxTopKGroupSize];
    T targetYCache[maxTopKGroupSize];
    T targetZCache[maxTopKGroupSize];
    uint32_t compNum = 384;
    // group 聚合缓冲里每个 target 的对齐槽位长度（= k 向上取整到 8 的倍数），保证 VEC 写入 32B 对齐
    uint32_t kAligned = 8;
    uint32_t mergeLength = 256;
    uint32_t sort32Elements = 32;
    uint32_t sort32Offset = sort32Elements * 2;
    int32_t sort32RepeatTimes = compNum / 32;
    uint16_t sort32MergeRepeatTimes = compNum / 128;

    float minFloatValue = -3.40282347E+38;
    float maxFloatValue = 3.40282347E+38;

  public:
    // tiling
    uint32_t batch;
    uint32_t nPoint;
    uint32_t nSource;
    uint32_t coreNum;
    bool isFromKnn;
    int32_t k;
    uint32_t targetGroupSize;
    uint32_t usedCoreNum;
    uint32_t groupsPerBatch;
    uint32_t totalGroups;
    uint32_t groupsPerCoreBase;
    uint32_t extraCoreGroups;
    bool useTopK;
    uint32_t sourceAligned;
    uint32_t topkTmpSize;
    uint32_t selectorMode;
    bool balanceTargetGroups;
    TopkTiling topKTilingData;
};
} // namespace AscendC

#endif // _KNN_H_
