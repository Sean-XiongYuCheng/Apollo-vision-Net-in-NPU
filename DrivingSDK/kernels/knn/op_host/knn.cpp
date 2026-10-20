/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "knn_tiling.h"
#include "common/op_host/common.h"

#include <limits>

namespace optiling {
/****************class impl*****************/
static ge::graphStatus TilingForKnn(gert::TilingContext *context) {
    enum class SelectorMode : uint32_t {
        LIBRARY = 0,
        TRUNCATED = 1,
        N4096_K8 = 2,
        N4096_K32 = 3,
    };

    uint32_t batch;
    uint32_t nPoint;
    uint32_t nSource;
    bool isFromKnn;
    uint32_t coreNum;
    int32_t k;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *xyzShape = context->GetInputShape(0);
    const gert::StorageShape *centerXyzShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();
    if ((xyzShape == nullptr) || (centerXyzShape == nullptr) || (attr == nullptr) || (platformInfoPtr == nullptr) ||
        (context->GetInputDesc(0) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if ((attr->GetAttrPointer<bool>(0) == nullptr) || (attr->GetAttrPointer<int32_t>(1) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if ((xyzShape->GetStorageShape().GetDimNum() != 3) || (centerXyzShape->GetStorageShape().GetDimNum() != 3) ||
        (xyzShape->GetStorageShape().GetDim(1) != 3) || (centerXyzShape->GetStorageShape().GetDim(2) != 3) ||
        (xyzShape->GetStorageShape().GetDim(0) != centerXyzShape->GetStorageShape().GetDim(0))) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batchDim = centerXyzShape->GetStorageShape().GetDim(0);
    const int64_t nPointDim = centerXyzShape->GetStorageShape().GetDim(1);
    const int64_t nSourceDim = xyzShape->GetStorageShape().GetDim(2);
    if ((batchDim <= 0) || (nPointDim <= 0) || (nSourceDim <= 0) || (batchDim > std::numeric_limits<uint32_t>::max()) ||
        (nPointDim > std::numeric_limits<uint32_t>::max()) || (nSourceDim > std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    batch = static_cast<uint32_t>(batchDim);
    nPoint = static_cast<uint32_t>(nPointDim);
    nSource = static_cast<uint32_t>(nSourceDim);
    isFromKnn = *attr->GetAttrPointer<bool>(0);
    k = *attr->GetAttrPointer<int32_t>(1);
    if ((k <= 0) || (k > 99) || (static_cast<uint32_t>(k) > nSource)) {
        return ge::GRAPH_FAILED;
    }
    coreNum = platformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    size_t sysWorkspaceSize = 16 * 1024 * 1024; // Alloc 16M workspace
    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = sysWorkspaceSize;

    const uint32_t topkCandidates[5] = {16, 8, 4, 2, 1};
    const uint32_t tiledCandidates[6] = {8, 6, 5, 4, 2, 1};
    const uint64_t workItems = static_cast<uint64_t>(batch) * nPoint;
    const bool stabilizeSmallSourceGroups = isFromKnn && (nSource <= 64) && (workItems <= 512);
    const bool limitTinyN1024Cores = isFromKnn && (nSource == 1024) && (k <= 8) && (workItems <= 64);
    // Generic TopK has high fixed overhead for small source sets. The tiled path can bypass merging for N=15..32,
    // and remains faster and less input-sensitive for small KNN workloads around N=200/256.
    const bool useTiledForSmallKnn = isFromKnn &&
        (((nSource >= 7) && (nSource <= 9) && (static_cast<uint32_t>(k) >= nSource - 1)) ||
            ((nSource >= 15) && (nSource <= 32)) || ((nSource >= 64) && (nSource <= 257) && (workItems <= 512)) ||
            ((nSource == 1024) && (k <= 8) && (workItems <= 192)));
    // Small single-tile workloads avoid sorting padding that cannot contribute to the result.
    const uint32_t sourceTileSize =
        useTiledForSmallKnn ? ((nSource <= 128) ? 128 : ((nSource <= 256) ? 256 : 384)) : 384;
    const bool tryTopK = (nSource <= 4096) && !useTiledForSmallKnn;
    const uint32_t sourceAligned = tryTopK ? (nSource + 31) / 32 * 32 : 0;
    const uint32_t kAligned = (static_cast<uint32_t>(k) + 7) / 8 * 8;
    SelectorMode selectorMode = SelectorMode::LIBRARY;
    if (tryTopK && isFromKnn) {
        if ((nSource == 4096) && (k <= 8)) {
            selectorMode = SelectorMode::N4096_K8;
        } else if ((nSource == 4096) && (k <= 32)) {
            selectorMode = SelectorMode::N4096_K32;
        } else if (sourceAligned <= 1024) {
            selectorMode = SelectorMode::TRUNCATED;
        }
    }
    const bool useCustomSelector = selectorMode != SelectorMode::LIBRARY;
    uint64_t partialTopKTmpSize = 0;
    if (useCustomSelector) {
        uint64_t partialTmpFloatCount;
        if (selectorMode == SelectorMode::TRUNCATED) {
            partialTmpFloatCount = static_cast<uint64_t>(sourceAligned) * 2;
            uint32_t queueCount = sourceAligned / 32;
            const uint32_t queueStride = static_cast<uint32_t>(k) * 8;
            while (queueCount > 1) {
                queueCount = (queueCount + 3) / 4;
                partialTmpFloatCount += static_cast<uint64_t>(queueCount) * queueStride;
            }
            partialTmpFloatCount += static_cast<uint64_t>((static_cast<uint32_t>(k) + 31) / 32 * 32) * 2;
            partialTopKTmpSize = partialTmpFloatCount * sizeof(float);
        } else if (selectorMode == SelectorMode::N4096_K8) {
            // Sort32 proposals plus one 4-way truncated-merge level at a time.
            // Each retained queue occupies 64 floats (8 value/index pairs from
            // each of four input queues). One final slot and one Extract slot
            // are kept after the hierarchy.
            partialTmpFloatCount = static_cast<uint64_t>(sourceAligned) * 2;
            uint32_t queueCount = sourceAligned / 32;
            while (queueCount > 1) {
                queueCount = (queueCount + 3) / 4;
                partialTmpFloatCount += static_cast<uint64_t>(queueCount) * 64;
            }
            partialTmpFloatCount += 64;
            partialTopKTmpSize = partialTmpFloatCount * sizeof(float);
        } else {
            // N=4096, k<=32: retain 32 candidates per queue. Four level-1
            // queues are reused as scratch instead of materializing all 32,
            // keeping the custom selector within UB.
            constexpr uint64_t mergeQueueStride = 256;
            partialTmpFloatCount = static_cast<uint64_t>(sourceAligned) * 2 + 4 * mergeQueueStride +
                static_cast<uint64_t>(sourceAligned / 512) * mergeQueueStride +
                static_cast<uint64_t>(sourceAligned / 2048) * mergeQueueStride + mergeQueueStride + 64;
            partialTopKTmpSize = partialTmpFloatCount * sizeof(float);
        }
    }

    KnnTilingData TilingData;
    uint32_t targetGroupSize = 1;
    uint32_t topkTmpSize = 0;
    bool topkConfigured = false;
    bool useTopK = false;

    if (tryTopK) {
        // Custom truncated TopK is configured independently from the generic
        // TopK workspace so N=4096 can fit in UB without reserving unused
        // library temporary space.
        for (uint32_t i = 0; i < 5; i++) {
            const uint32_t g = topkCandidates[i];
            const uint32_t groupsPerBatchTmp = (nPoint - 1) / g + 1;
            const uint64_t totalGroupsTmp = static_cast<uint64_t>(batch) * groupsPerBatchTmp;
            if ((g != 1) && (totalGroupsTmp < coreNum)) {
                continue;
            }
            if (stabilizeSmallSourceGroups && (g != 1) && (totalGroupsTmp < static_cast<uint64_t>(coreNum) * 3)) {
                continue;
            }
            if (g == 16) {
                // 仅扩大每核至少有 8 个 group 的 N=1024 路径，避免粗粒度 group 放大尾核负载。
                const uint64_t groupSlots = static_cast<uint64_t>(groupsPerBatchTmp) * g;
                const bool enoughGroupsPerCore = totalGroupsTmp >= static_cast<uint64_t>(coreNum) * 8;
                const bool efficientTail = static_cast<uint64_t>(nPoint) * 10 >= groupSlots * 9;
                if (nSource != 1024 || !enoughGroupsPerCore || !efficientTail) {
                    continue;
                }
            }
            const uint64_t targetBytes = (static_cast<uint64_t>(g) * 3 * sizeof(float) + 31) / 32 * 32;
            if (useCustomSelector) {
                // Keep enough UB headroom for the deeper N=4096 merge tree.
                // Targets are still processed serially inside the custom
                // selector, so G=1 only gives up group-level fixed overhead.
                if ((nSource == 4096) && (g != 1)) {
                    continue;
                }
                const uint64_t partialBuffers = static_cast<uint64_t>(sourceAligned) * sizeof(float) * 6 +
                    static_cast<uint64_t>(g) * sourceAligned * sizeof(float) +
                    static_cast<uint64_t>(sourceAligned) * sizeof(int32_t) +
                    static_cast<uint64_t>(g) * kAligned * (sizeof(float) + sizeof(int32_t)) + targetBytes +
                    partialTopKTmpSize;
                if (partialBuffers > ubSize) {
                    continue;
                }
                targetGroupSize = g;
                topkTmpSize = static_cast<uint32_t>(partialTopKTmpSize);
                topkConfigured = true;
                useTopK = true;
                break;
            }

            // Generic TopK NORMAL + MERGE_SORT: explicit indices, ascending
            // order, and no input reuse.
            uint32_t maxTmpSize = 0;
            uint32_t minTmpSize = 0;
            TopkTiling candidateTiling = {};
            const bool gotTmpSize = AscendC::GetTopKMaxMinTmpSize(platformInfo, static_cast<int32_t>(sourceAligned),
                static_cast<int32_t>(g), false, true, AscendC::TopKMode::TOPK_NORMAL, false, sizeof(float), maxTmpSize,
                minTmpSize);
            const bool gotTiling =
                AscendC::TopKTilingFunc(platformInfo, static_cast<int32_t>(sourceAligned), static_cast<int32_t>(g), k,
                    sizeof(float), true, AscendC::TopKMode::TOPK_NORMAL, false, candidateTiling);
            if (!gotTmpSize || !gotTiling || (minTmpSize > maxTmpSize)) {
                continue;
            }
            const uint64_t topkTmpAligned = (static_cast<uint64_t>(minTmpSize) + 31) / 32 * 32;
            const uint64_t topkBuffers = static_cast<uint64_t>(sourceAligned) * sizeof(float) * 6 +
                static_cast<uint64_t>(g) * sourceAligned * sizeof(float) +
                static_cast<uint64_t>(sourceAligned) * sizeof(int32_t) +
                static_cast<uint64_t>(g) * kAligned * (sizeof(float) + sizeof(int32_t)) + targetBytes + topkTmpAligned;
            if (topkBuffers > ubSize) {
                continue;
            }
            targetGroupSize = g;
            topkTmpSize = static_cast<uint32_t>(topkTmpAligned);
            topkConfigured = true;
            break;
        }
        if (topkConfigured && !useCustomSelector) {
            // Generated tiling-data objects own their field metadata and must not be copy-assigned.
            useTopK = AscendC::TopKTilingFunc(platformInfo, static_cast<int32_t>(sourceAligned),
                static_cast<int32_t>(targetGroupSize), k, sizeof(float), true, AscendC::TopKMode::TOPK_NORMAL, false,
                TilingData.topKTilingData);
        }
    }
    if (!useTopK) {
        targetGroupSize = 1;
        topkTmpSize = 0;
        // 回退路径仍优先在不损失多核并行度时复用 source tile。
        for (uint32_t i = 0; i < 6; i++) {
            const uint32_t g = tiledCandidates[i];
            const uint32_t groupsPerBatchTmp = (nPoint - 1) / g + 1;
            const uint64_t totalGroupsTmp = static_cast<uint64_t>(batch) * groupsPerBatchTmp;
            if ((g == 6) && !useTiledForSmallKnn) {
                continue;
            }
            if (stabilizeSmallSourceGroups && (g != 1) && (totalGroupsTmp < static_cast<uint64_t>(coreNum) * 3)) {
                continue;
            }
            if (useTiledForSmallKnn && (g == 8) && (totalGroupsTmp < static_cast<uint64_t>(coreNum) * 2)) {
                continue;
            }
            const bool allowN64Underfill = (nSource == 64) && (g == 4) && (totalGroupsTmp * 2 >= coreNum);
            const bool allowN1024G5Underfill =
                (nSource == 1024) && (workItems <= 192) && (g == 5) && (totalGroupsTmp + 1 >= coreNum);
            if ((totalGroupsTmp >= coreNum) || allowN64Underfill || allowN1024G5Underfill) {
                targetGroupSize = g;
                break;
            }
        }
    }

    const uint32_t groupsPerBatchUnbalanced = (nPoint - 1) / targetGroupSize + 1;
    const bool balanceTargetGroups = isFromKnn && !useTopK && (batch == 1) && (nSource == 1024) && (k <= 8) &&
        (targetGroupSize == 5) && (workItems > 64) && (workItems <= 192) && (groupsPerBatchUnbalanced + 1 == coreNum);
    const uint32_t groupsPerBatch = balanceTargetGroups ? coreNum : groupsPerBatchUnbalanced;
    // Four Sort32 queues fit one merge stage exactly. At this balanced G5/core-count boundary, a 512-point tile
    // therefore turns N1024 into two full tiles instead of 384+384+256 without changing the selection order.
    const uint32_t effectiveSourceTileSize = balanceTargetGroups ? 512 : sourceTileSize;
    const uint64_t totalGroups64 = static_cast<uint64_t>(batch) * groupsPerBatch;
    if ((totalGroups64 == 0) || (totalGroups64 > std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalGroups = static_cast<uint32_t>(totalGroups64);
    const uint32_t availableCoreNum = limitTinyN1024Cores ? ((totalGroups + 1) / 2) : coreNum;
    const bool padN1024BlockDim = isFromKnn && (nSource == 1024) && (k <= 8) && (workItems > 64) && (workItems <= 192);
    const uint32_t usedCoreNum =
        padN1024BlockDim ? coreNum : (totalGroups < availableCoreNum ? totalGroups : availableCoreNum);
    const uint32_t groupsPerCoreBase = totalGroups / usedCoreNum;
    const uint32_t extraCoreGroups = totalGroups % usedCoreNum;
    const uint32_t groupsPerCore = groupsPerCoreBase + (extraCoreGroups == 0 ? 0 : 1);

    TilingData.set_batch(batch);
    TilingData.set_nPoint(nPoint);
    TilingData.set_nSource(nSource);
    TilingData.set_isFromKnn(isFromKnn);
    TilingData.set_coreNum(coreNum);
    TilingData.set_k(k);
    TilingData.set_targetGroupSize(targetGroupSize);
    TilingData.set_sourceTileSize(effectiveSourceTileSize);
    TilingData.set_usedCoreNum(usedCoreNum);
    TilingData.set_groupsPerBatch(groupsPerBatch);
    TilingData.set_totalGroups(totalGroups);
    TilingData.set_groupsPerCore(groupsPerCore);
    TilingData.set_groupsPerCoreBase(groupsPerCoreBase);
    TilingData.set_extraCoreGroups(extraCoreGroups);
    TilingData.set_useTopK(useTopK);
    TilingData.set_sourceAligned(sourceAligned);
    TilingData.set_topkTmpSize(topkTmpSize);
    TilingData.set_selectorMode(static_cast<uint32_t>(selectorMode));
    TilingData.set_balanceTargetGroups(balanceTargetGroups);
    context->SetBlockDim(usedCoreNum);
    auto *rawTilingData = context->GetRawTilingData();
    if ((rawTilingData == nullptr) || (rawTilingData->GetCapacity() < TilingData.GetDataSize())) {
        return ge::GRAPH_FAILED;
    }
    TilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForKnn(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *xyzShape = context->GetInputShape(0);
    const gert::Shape *centerXyzShape = context->GetInputShape(1);
    gert::Shape *distShape = context->GetOutputShape(0);
    gert::Shape *idxShape = context->GetOutputShape(1);
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    if ((xyzShape == nullptr) || (centerXyzShape == nullptr) || (distShape == nullptr) || (idxShape == nullptr) ||
        (attr == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if ((xyzShape->GetDimNum() != 3) || (centerXyzShape->GetDimNum() != 3)) {
        return ge::GRAPH_FAILED;
    }
    const int32_t *kPtr = attr->GetAttrPointer<int32_t>(1);
    if (kPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = centerXyzShape->GetDim(0);
    const int64_t nPoint = centerXyzShape->GetDim(1);
    const int64_t nSource = xyzShape->GetDim(2);
    const int64_t xyzBatch = xyzShape->GetDim(0);
    const int64_t xyzCoordinateDim = xyzShape->GetDim(1);
    const int64_t centerCoordinateDim = centerXyzShape->GetDim(2);
    const int32_t k = *kPtr;
    // InferShape 可能收到 -1 等未知维；仅拒绝已知且非法的值，运行时具体值由 TilingForKnn 严格校验。
    if ((batch == 0) || (nPoint == 0) || (nSource == 0) || (k <= 0) || (k > 99) ||
        ((xyzBatch >= 0) && (batch >= 0) && (xyzBatch != batch)) ||
        ((xyzCoordinateDim >= 0) && (xyzCoordinateDim != 3)) ||
        ((centerCoordinateDim >= 0) && (centerCoordinateDim != 3)) || (nSource > std::numeric_limits<int32_t>::max()) ||
        ((nSource > 0) && (k > nSource))) {
        return ge::GRAPH_FAILED;
    }

    distShape->SetDimNum(3);
    distShape->SetDim(0, batch);
    distShape->SetDim(1, nPoint);
    distShape->SetDim(2, k);

    idxShape->SetDimNum(3);
    idxShape->SetDim(0, batch);
    idxShape->SetDim(1, nPoint);
    idxShape->SetDim(2, k);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForKnn(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Knn : public OpDef {
  public:
    explicit Knn(const char *name) : OpDef(name) {
        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("center_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("is_from_knn").AttrType(REQUIRED).Bool();
        this->Attr("k").AttrType(REQUIRED).Int();
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InfershapeForKnn).SetInferDataType(ge::InferDataTypeForKnn);
        this->AICore().SetTiling(optiling::TilingForKnn);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(Knn);
}
