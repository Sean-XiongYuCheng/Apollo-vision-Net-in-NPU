
#ifndef FFT_CONV1D_TILING_H
#define FFT_CONV1D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t FFT_CONV1D_ALGO_DIRECT = 0;
constexpr uint32_t FFT_CONV1D_ALGO_FFT_GM = 2;

constexpr uint32_t FFT_CONV1D_FFT_MIN_K = 64;
constexpr uint32_t FFT_CONV1D_DIRECT_TILE = 4096;

constexpr uint32_t FFT_CONV1D_VEC_CHUNK_MAX = 4096;

constexpr uint32_t FFT_CONV1D_MIN_NFFT_GM = 256U;

constexpr uint32_t FFT_CONV1D_MIN_NRADIX = 8U;

constexpr uint32_t FFT_CONV1D_GM_BUFFER_COUNT = 8U;

BEGIN_TILING_DATA_DEF(FftConv1dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, channel);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, kernelLen);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, algo);
TILING_DATA_FIELD_DEF(uint32_t, nFft);
TILING_DATA_FIELD_DEF(uint32_t, nRadix);
TILING_DATA_FIELD_DEF(uint32_t, nRadix2);
TILING_DATA_FIELD_DEF(uint32_t, vecChunk);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(bool, allowHf32);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingB);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingC);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FftConv1d, FftConv1dTilingData)

}

#endif
