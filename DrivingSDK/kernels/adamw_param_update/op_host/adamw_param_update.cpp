/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "adamw_param_update_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {

static ge::graphStatus SetTilingKeyForAdamw(gert::TilingContext *context) {
    if (context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(0);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(1);
    } else if (dtype == ge::DT_BF16) {
        context->SetTilingKey(2);
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncForAdamwParamUpdate(gert::TilingContext *context) {
    CHECK_NULLPTR(context);
    auto platform = context->GetPlatformInfo();
    CHECK_NULLPTR(platform);
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    AdamwParamUpdateTilingData tiling;
    auto combinedParamPtr = context->GetInputTensor(0);
    CHECK_NULLPTR(combinedParamPtr);
    auto combinedParamShape = combinedParamPtr->GetStorageShape();
    uint32_t totalCoreTaskNum = combinedParamShape.GetShapeSize();

    auto attrsPtr = context->GetAttrs();
    CHECK_NULLPTR(attrsPtr);
    float lr = *(attrsPtr->GetAttrPointer<float>(0));
    float beta1 = *(attrsPtr->GetAttrPointer<float>(1));
    float beta2 = *(attrsPtr->GetAttrPointer<float>(2));
    float decay = *(attrsPtr->GetAttrPointer<float>(3));
    float eps = *(attrsPtr->GetAttrPointer<float>(4));
    float biasCorrection1 = *(attrsPtr->GetAttrPointer<float>(5));
    float biasCorrection2 = *(attrsPtr->GetAttrPointer<float>(6));
    bool amsgrad = *(attrsPtr->GetAttrPointer<bool>(7));

    SetTilingKeyForAdamw(context);

    context->SetBlockDim(coreNum);
    tiling.set_usedCoreNum(coreNum);
    tiling.set_totalCoreTaskNum(totalCoreTaskNum);
    tiling.set_lr(lr);
    tiling.set_beta1(beta1);
    tiling.set_beta2(beta2);
    tiling.set_decay(decay);
    tiling.set_eps(eps);
    tiling.set_biasCorrection1(biasCorrection1);
    tiling.set_biasCorrection2(biasCorrection2);
    tiling.set_amsgrad(amsgrad);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class AdamwParamUpdate : public OpDef {
  public:
    explicit AdamwParamUpdate(const char *name) : OpDef(name) {
        this->Input("combined_param")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("combined_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("exp_avg")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("exp_avg_sq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("max_exp_avg_sq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("lr").AttrType(REQUIRED).Float();
        this->Attr("beta1").AttrType(REQUIRED).Float();
        this->Attr("beta2").AttrType(REQUIRED).Float();
        this->Attr("weight_decay").AttrType(REQUIRED).Float();
        this->Attr("eps").AttrType(REQUIRED).Float();
        this->Attr("bias_correction1").AttrType(REQUIRED).Float();
        this->Attr("bias_correction2").AttrType(REQUIRED).Float();
        this->Attr("amsgrad").AttrType(REQUIRED).Bool();

        this->Output("combined_param")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("exp_avg")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("exp_avg_sq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("max_exp_avg_sq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFuncForAdamwParamUpdate);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};
OP_ADD(AdamwParamUpdate);
} // namespace ops
