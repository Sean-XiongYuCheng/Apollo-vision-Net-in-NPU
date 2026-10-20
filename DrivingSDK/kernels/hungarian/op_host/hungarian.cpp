/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "ge/utils.h"
#include "hungarian_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    CHECK_NULLPTR(context);
    uint32_t coreNum = 1;
    context->SetBlockDim(coreNum);

    auto inputTensorPtr = context->GetInputTensor(0);
    auto inputShape = inputTensorPtr->GetStorageShape();
    uint32_t numRows = inputShape.GetDim(0);
    uint32_t numCols = inputShape.GetDim(1);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    bool transposed = *(attrsPtr->GetAttrPointer<bool>(0));

    HungarianTilingData tiling;
    tiling.set_numRows(numRows);
    tiling.set_numCols(numCols);
    tiling.set_coreNum(coreNum);
    tiling.set_transposed(transposed);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const auto inputShape = context->GetInputShape(0);
    auto n = inputShape->GetDim(0);

    auto rowIndShape = context->GetOutputShape(0);
    auto colIndShape = context->GetOutputShape(1);

    rowIndShape->SetDimNum(1);
    rowIndShape->SetDim(0, n);
    colIndShape->SetDimNum(1);
    colIndShape->SetDim(0, n);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Hungarian : public OpDef {
  public:
    explicit Hungarian(const char *name) : OpDef(name) {
        this->Input("cost_matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("row_ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("col_ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("transposed").AttrType(REQUIRED).Bool();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(Hungarian);
}
