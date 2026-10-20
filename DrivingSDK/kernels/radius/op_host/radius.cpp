/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#include "radius_tiling.h"
#include "common/op_host/common.h"
#include "ge/utils.h"

namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Y_INDEX = 1;
constexpr uint32_t PTR_X_INDEX = 2;
constexpr uint32_t PTR_Y_INDEX = 3;
constexpr uint32_t OUT_FINAL_INDEX = 0;
constexpr uint32_t NUM_NEIGHBORS_INDEX = 1;
constexpr uint32_t GROUP_NUM = 2; //Inputs only contain group X and group Y
constexpr uint32_t BLOCK_BYTES = 32; // Single Block requires 32B alignment.
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t ALIGN_NUM_64 = 64;
constexpr uint32_t FLOAT32_BYTES = 4;
constexpr uint32_t INT32_BYTES = 4;
constexpr uint32_t RESERVED_UB_SIZE = 16 * 1024;
constexpr uint32_t MAX_POINTS_PER_BATCH = 1024;
}

namespace optiling {
/****************class impl*****************/
static ge::graphStatus TilingForRadius(gert::TilingContext *context) {
    CHECK_NULLPTR(context);
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX); // [coordinateDim, num_points_x]
    const gert::StorageShape *yShape = context->GetInputShape(Y_INDEX); // [coordinateDim, num_points_y]
    const gert::StorageShape *ptrXShape = context->GetInputShape(PTR_X_INDEX); // [batch_size + 1]
    const gert::StorageShape *ptrYShape = context->GetInputShape(PTR_Y_INDEX); // [batch_size + 1]
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    auto platformInfoPtr = context->GetPlatformInfo();

    CHECK_NULLPTR(xShape);
    CHECK_NULLPTR(yShape);
    CHECK_NULLPTR(ptrXShape);
    CHECK_NULLPTR(ptrYShape);
    CHECK_NULLPTR(platformInfoPtr);
    CHECK_NULLPTR(attr);
    const int32_t *maxNumNeighborsAttr = attr->GetAttrPointer<int32_t>(1);
    CHECK_NULLPTR(maxNumNeighborsAttr);

    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t batchSize = ptrXShape->GetStorageShape().GetDim(0) - 1; // batchSize Num
    uint32_t coordinateDim = xShape->GetStorageShape().GetDim(0);
    uint32_t numPointsX = xShape->GetStorageShape().GetDim(1);
    uint32_t numPointsY = yShape->GetStorageShape().GetDim(1);
    float r = *(attr->GetAttrPointer<float>(0));
    float rSquare = r * r;
    uint32_t maxNumNeighbors = static_cast<uint32_t>(*maxNumNeighborsAttr);
    uint32_t coreOutputOffsetSlots = *(attr->GetAttrPointer<int32_t>(2));

    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || batchSize == 0 || coordinateDim == 0 || coreOutputOffsetSlots == 0 ||
        ubSize <= RESERVED_UB_SIZE) {
        return ge::GRAPH_FAILED;
    }
    uint32_t usedCoreNum = batchSize < coreNum ? batchSize : coreNum;
    if (usedCoreNum + 1 > coreOutputOffsetSlots) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchPerCore = DivCeil(batchSize, usedCoreNum);
    uint32_t batchPerCoreTail = batchPerCore - 1;
    uint32_t headCoreNum = batchSize - batchPerCoreTail * usedCoreNum;

    uint32_t coreIndexSize = CeilAlign((batchPerCore + 1) * INT32_BYTES, BLOCK_BYTES);
    uint32_t coreIndexCount = coreIndexSize / sizeof(uint32_t);
    uint32_t batchPointsCount = CeilAlign(MAX_POINTS_PER_BATCH, ALIGN_NUM_64);
    uint32_t batchPointsSize = CeilAlign(coordinateDim * batchPointsCount * FLOAT32_BYTES, BLOCK_BYTES);

    uint64_t requiredUbSize = static_cast<uint64_t>(coreIndexSize) * 2 + static_cast<uint64_t>(batchPointsSize) * 2 +
        static_cast<uint64_t>(batchPointsCount) * FLOAT32_BYTES * 6 + static_cast<uint64_t>(batchPointsCount);
    requiredUbSize += CeilAlign(ALIGN_NUM * INT32_BYTES, BLOCK_BYTES);
    requiredUbSize += CeilAlign((usedCoreNum + 1) * ALIGN_NUM * INT32_BYTES, BLOCK_BYTES);
    if (ubSize <= RESERVED_UB_SIZE + requiredUbSize) {
        return ge::GRAPH_FAILED;
    }

    auto systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    uint64_t outTempWorkspaceSize =
        CeilAlign(static_cast<uint64_t>(GROUP_NUM) * numPointsY * maxNumNeighbors * INT32_BYTES,
            static_cast<uint64_t>(BLOCK_BYTES));
    auto currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize + outTempWorkspaceSize;

    RadiusTilingData TilingData;
    TilingData.set_coordinateDim(coordinateDim);
    TilingData.set_batchSize(batchSize);
    TilingData.set_numPointsX(numPointsX);
    TilingData.set_numPointsY(numPointsY);
    TilingData.set_maxNumNeighbors(maxNumNeighbors);
    TilingData.set_coreOutputOffsetSlots(coreOutputOffsetSlots);
    TilingData.set_usedCoreNum(usedCoreNum);
    TilingData.set_headCoreNum(headCoreNum);
    TilingData.set_batchPerCore(batchPerCore);
    TilingData.set_batchPerCoreTail(batchPerCoreTail);
    TilingData.set_coreIndexSize(coreIndexSize);
    TilingData.set_batchPointsSize(batchPointsSize);
    TilingData.set_coreIndexCount(coreIndexCount);
    TilingData.set_batchPointsCount(batchPointsCount);

    TilingData.set_rSquare(rSquare);

    context->SetBlockDim(usedCoreNum);
    CHECK_NULLPTR(context->GetRawTilingData());
    TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InfershapeForRadius(gert::InferShapeContext *context) {
    const gert::Shape *yShape = context->GetInputShape(Y_INDEX); // [coordinateDim, num_points_y]

    gert::Shape *outFinalShape = context->GetOutputShape(OUT_FINAL_INDEX); // [2, num_points_y * max_num_neighbors]
    gert::Shape *numNeighborsShape = context->GetOutputShape(NUM_NEIGHBORS_INDEX);
    const gert::RuntimeAttrs *attr = context->GetAttrs();

    CHECK_NULLPTR(yShape);
    CHECK_NULLPTR(outFinalShape);
    CHECK_NULLPTR(numNeighborsShape);
    CHECK_NULLPTR(attr);

    uint32_t numPointsY = yShape->GetDim(1);
    const int32_t maxNumNeighbors = *attr->GetAttrPointer<int32_t>(1);
    const int32_t coreOutputOffsetSlots = *attr->GetAttrPointer<int32_t>(2);

    outFinalShape->SetDimNum(GROUP_NUM);
    outFinalShape->SetDim(0, GROUP_NUM);
    outFinalShape->SetDim(1, static_cast<int64_t>(numPointsY) * maxNumNeighbors);

    numNeighborsShape->SetDimNum(1);
    numNeighborsShape->SetDim(0, coreOutputOffsetSlots * ALIGN_NUM);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRadius(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(OUT_FINAL_INDEX, ge::DT_INT32);
    context->SetOutputDataType(NUM_NEIGHBORS_INDEX, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Radius : public OpDef {
  public:
    explicit Radius(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ptr_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ptr_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_final")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("actual_num_neighbors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("r").AttrType(REQUIRED).Float();
        this->Attr("max_num_neighbors").AttrType(REQUIRED).Int();
        this->Attr("core_output_offset_slots").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InfershapeForRadius).SetInferDataType(ge::InferDataTypeForRadius);
        this->AICore().SetTiling(optiling::TilingForRadius);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Radius);
}
