
#include "ge/utils.h"
#include "fft_conv1d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
inline uint32_t NextPow2(uint32_t n) {
    uint32_t v = 2;
    while (v < n && v <= (1U << 30)) {
        v <<= 1;
    }
    return v;
}

inline void SplitRadix(uint32_t n, uint32_t &n1, uint32_t &n2) {
    uint32_t e = 0;
    while ((1U << e) < n) {
        ++e;
    }
    n1 = 1U << ((e + 1) / 2);
    n2 = 1U << (e / 2);
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    CHECK_NULLPTR(context);
    auto xTensor = context->GetInputTensor(0);
    auto kTensor = context->GetInputTensor(1);
    if (xTensor == nullptr || kTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto xShape = xTensor->GetOriginShape();
    auto kShape = kTensor->GetOriginShape();

    if (xShape.GetDimNum() != 3 || kShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t L = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t Hk = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(kShape.GetDim(1));

    if (B < 1 || H < 1 || L < 1 || K < 1) {
        return ge::GRAPH_FAILED;
    }
    if (H != Hk) {
        return ge::GRAPH_FAILED;
    }
    if (K > L) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t need = (L + K - 1) < 2 ? 2 : (L + K - 1);
    const uint32_t nMono = NextPow2(need);

    uint32_t algo = (K >= FFT_CONV1D_FFT_MIN_K) ? FFT_CONV1D_ALGO_FFT_GM : FFT_CONV1D_ALGO_DIRECT;

    const uint32_t nFft = (nMono < FFT_CONV1D_MIN_NFFT_GM) ? FFT_CONV1D_MIN_NFFT_GM : nMono;
    uint32_t nRadix = 0;
    uint32_t nRadix2 = 0;
    SplitRadix(nFft, nRadix, nRadix2);
    if (algo == FFT_CONV1D_ALGO_FFT_GM && (nRadix > FFT_CONV1D_VEC_CHUNK_MAX || nRadix2 < FFT_CONV1D_MIN_NRADIX)) {
        algo = FFT_CONV1D_ALGO_DIRECT;
    }
    const uint32_t vecChunk = (nFft < FFT_CONV1D_VEC_CHUNK_MAX) ? nFft : FFT_CONV1D_VEC_CHUNK_MAX;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    FftConv1dTilingData tilingData;
    if (algo != FFT_CONV1D_ALGO_DIRECT) {
        constexpr auto posAB = matmul_tiling::TPosition::GM;
        constexpr auto posC = matmul_tiling::TPosition::GM;
        const int32_t s1 = static_cast<int32_t>(nRadix);
        const int32_t s2 = static_cast<int32_t>(nRadix2);

        auto solve = [&](int32_t m, int32_t n, int32_t k, TCubeTiling &out, const char *tag) {
            matmul_tiling::MatmulApiTiling t(ascendcPlatform);
            t.SetAType(posAB, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            t.SetBType(posAB, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            t.SetCType(posC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            t.SetBiasType(
                matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            t.SetShape(m, n, k);
            t.SetOrgShape(m, n, k);
            t.SetBias(false);
            t.SetBufferSpace(-1, -1, -1);
            if (t.GetTiling(out) == -1) {
                return false;
            }
            return true;
        };
        if (!solve(2 * s1, s2, s1, tilingData.cubeTiling, "A") || !solve(s1, s2, 2 * s1, tilingData.cubeTilingB, "B") ||
            !solve(s1, 2 * s2, 2 * s2, tilingData.cubeTilingC, "C")) {
            algo = FFT_CONV1D_ALGO_DIRECT;
        }
    }

    const uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t aivGroupNum = aivCoreNum / 2;
    uint32_t coreNum = (aicCoreNum < aivGroupNum) ? aicCoreNum : aivGroupNum;
    const uint32_t totalRows = B * H;
    const uint32_t splitUnit = (algo != FFT_CONV1D_ALGO_DIRECT) ? H : totalRows;
    const uint32_t neededGroupNum = CeilDiv(splitUnit, 2);
    uint32_t usedCoreNum = (neededGroupNum < coreNum) ? neededGroupNum : coreNum;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    const uint32_t vectorWorkerNum = usedCoreNum * 2;
    const uint32_t rowsPerCore = CeilDiv(totalRows, vectorWorkerNum);

    uint32_t tileLen = (L < FFT_CONV1D_DIRECT_TILE) ? L : FFT_CONV1D_DIRECT_TILE;
    if (tileLen < K) {
        tileLen = K;
    }

    tilingData.set_batch(B);
    tilingData.set_channel(H);
    tilingData.set_seqLen(L);
    tilingData.set_kernelLen(K);
    tilingData.set_totalRows(totalRows);
    tilingData.set_algo(algo);
    tilingData.set_nFft(nFft);
    tilingData.set_nRadix(nRadix);
    tilingData.set_nRadix2(nRadix2);
    tilingData.set_vecChunk(vecChunk);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_tileLen(tileLen);
    tilingData.set_rowsPerCore(rowsPerCore);
    auto attrsPtr = context->GetAttrs();
    const bool *allowHf32Ptr = (attrsPtr == nullptr) ? nullptr : attrsPtr->GetBool(0);
    tilingData.set_allowHf32(allowHf32Ptr != nullptr && *allowHf32Ptr);

    context->SetBlockDim(usedCoreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t userWorkspace = 0;
    if (algo == FFT_CONV1D_ALGO_FFT_GM) {
        userWorkspace = static_cast<size_t>(vectorWorkerNum) * FFT_CONV1D_GM_BUFFER_COUNT * static_cast<size_t>(nFft) *
            sizeof(float);
    } else {
        const size_t rowLen = ((static_cast<size_t>(K) - 1 + L + 7) / 8) * 8;
        userWorkspace = static_cast<size_t>(vectorWorkerNum) * rowLen * sizeof(float);
    }
    const size_t sysWorkspace = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspace + sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FftConv1d : public OpDef {
  public:
    explicit FftConv1d(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kernel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("allow_hf32").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
#if __DRIVING_HOST_AICORE__ == 310
        this->AICore().AddConfig("ascend950");
#endif
    }
};

OP_ADD(FftConv1d);
}
