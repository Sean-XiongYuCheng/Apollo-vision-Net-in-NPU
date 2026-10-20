
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr int32_t ALGO_DIRECT = 0;
constexpr int32_t ALGO_FFT_GM = 2;
constexpr int32_t GRAD_GM_BUFFER_COUNT = 12;
constexpr int32_t VEC_CHUNK_MAX = 4096;

__aicore__ inline int32_t VectorWorkerIdx() { return static_cast<int32_t>(GetBlockIdx()); }

__aicore__ inline int32_t AlignUp8(int32_t x) { return (x + 7U) & ~7U; }

__aicore__ inline int32_t MinU(int32_t a, int32_t b) { return a < b ? a : b; }
}

class FftConv1dGradDirect {
  public:
    __aicore__ inline FftConv1dGradDirect() {}

    template <class TILING>
    __aicore__ inline void Init(
        TPipe *pipe, GM_ADDR grad, GM_ADDR x, GM_ADDR w, GM_ADDR gx, GM_ADDR gw, GM_ADDR workspace, const TILING &t) {
        B_ = t.batch;
        H_ = t.channel;
        L_ = t.seqLen;
        K_ = t.kernelLen;
        tile_ = t.tileLen;
        workers_ = static_cast<int32_t>(t.usedCoreNum) * 2;
        split_ = static_cast<int32_t>(t.batchSplit);
        rowLen_ = AlignUp8(L_ + K_ - 1);

        gGm_.SetGlobalBuffer((__gm__ float *)grad);
        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        gxGm_.SetGlobalBuffer((__gm__ float *)gx);
        gwGm_.SetGlobalBuffer((__gm__ float *)gw);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        pipe->InitBuffer(bufIn_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufIn2_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufOut_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufWork_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufK_, AlignUp8(K_) * sizeof(float));
        pipe->InitBuffer(bufAcc_, AlignUp8(K_) * sizeof(float));
        pipe->InitBuffer(bufRed_, 32);
        pipe->InitBuffer(bufZero_, AlignUp8(K_ > 8 ? K_ : 8) * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> zeros = bufZero_.Get<float>();
        Duplicate(zeros, 0.0f, AlignUp8(K_ > 8 ? K_ : 8));

        const int32_t workerIdx = VectorWorkerIdx();
        const int32_t base = workerIdx * rowLen_;

        const int32_t tasks = H_ * split_;
        const int32_t bPerChunk = (B_ + split_ - 1) / split_;
        for (int32_t t = workerIdx; t < tasks; t += workers_) {
            const int32_t h = t / split_;
            const int32_t b0 = (t % split_) * bPerChunk;
            const int32_t b1 = MinU(b0 + bPerChunk, B_);
            if (b0 >= b1) {
                continue;
            }

            LoadKernelRow(h);
            LocalTensor<float> acc = bufAcc_.Get<float>();
            Duplicate(acc, 0.0f, AlignUp8(K_));

            for (int32_t b = b0; b < b1; ++b) {
                const int32_t row = b * H_ + h;
                BuildPaddedGradRow(row, base);
                for (int32_t t0 = 0; t0 < L_; t0 += tile_) {
                    GxTile(row, base, t0, MinU(tile_, L_ - t0));
                }
                GkAccumulate(row, base, acc);
            }
            StoreGradKernel(h, acc);
        }
    }

  private:
    __aicore__ inline void LoadKernelRow(int32_t h) {
        LocalTensor<float> kb = bufK_.Get<float>();
        DataCopyExtParams cp{1, K_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(kb, wGm_[static_cast<uint64_t>(h) * K_], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void BuildPaddedGradRow(int32_t row, int32_t base) {
        LocalTensor<float> a = bufIn_.Get<float>();
        LocalTensor<float> zeros = bufZero_.Get<float>();

        for (int32_t o = 0; o < L_; o += tile_) {
            const int32_t len = MinU(tile_, L_ - o);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(a, gGm_[static_cast<uint64_t>(row) * L_ + o], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyExtParams op{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
            DataCopyPad(wsGm_[base + o], a, op);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if (K_ > 1) {
            DataCopyExtParams zp{1, static_cast<uint32_t>(K_ - 1) * 4U, 0, 0, 0};
            DataCopyPad(wsGm_[base + L_], zeros, zp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void GxTile(int32_t row, int32_t base, int32_t t0, int32_t len) {
        LocalTensor<float> kb = bufK_.Get<float>();
        LocalTensor<float> out = bufOut_.Get<float>();
        LocalTensor<float> gin = bufIn_.Get<float>();

        Duplicate(out, 0.0f, AlignUp8(len));

        for (int32_t j = 0; j < K_; ++j) {
            const float kv = kb.GetValue(j);
            DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(gin, wsGm_[base + t0 + j], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Axpy(out, gin, kv, len);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams op{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
        DataCopyPad(gxGm_[static_cast<uint64_t>(row) * L_ + t0], out, op);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void GkAccumulate(int32_t row, int32_t base, const LocalTensor<float> &acc) {
        LocalTensor<float> xa = bufIn2_.Get<float>();
        LocalTensor<float> ga = bufIn_.Get<float>();
        LocalTensor<float> mul = bufOut_.Get<float>();
        LocalTensor<float> work = bufWork_.Get<float>();
        LocalTensor<float> red = bufRed_.Get<float>();

        for (int32_t j = 0; j < K_; ++j) {
            float sum = 0.0f;
            for (int32_t t0 = 0; t0 < L_; t0 += tile_) {
                const int32_t len = MinU(tile_, L_ - t0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                DataCopyPad(xa, xGm_[static_cast<uint64_t>(row) * L_ + t0], cp, pad);
                DataCopyPad(ga, wsGm_[base + t0 + j], cp, pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Mul(mul, xa, ga, len);
                ReduceSum<float>(red, mul, work, len);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                sum += red.GetValue(0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            acc.SetValue(j, acc.GetValue(j) + sum);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void StoreGradKernel(int32_t h, const LocalTensor<float> &acc) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        SetAtomicAdd<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(K_) * 4U, 0, 0, 0};
        DataCopyPad(gwGm_[static_cast<uint64_t>(h) * K_], acc, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetAtomicNone();
    }

    GlobalTensor<float> gGm_, xGm_, wGm_, gxGm_, gwGm_, wsGm_;
    TBuf<TPosition::VECCALC> bufIn_, bufIn2_, bufOut_, bufWork_, bufK_, bufAcc_, bufRed_, bufZero_;
    int32_t B_, H_, L_, K_, tile_, workers_, split_, rowLen_;
};

class FftConv1dGradFftGm {
  public:
    using MmGm =
        Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>,
            MatmulType<TPosition::GM, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>>;
    MmGm mmA_;
    MmGm mmB_;
    MmGm mmC_;

    __aicore__ inline FftConv1dGradFftGm() {}

    template <class TILING>
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR grad, GM_ADDR x, GM_ADDR w, GM_ADDR table, GM_ADDR gx, GM_ADDR gw,
        GM_ADDR workspace, const TILING &t) {
        B_ = t.batch;
        H_ = t.channel;
        L_ = t.seqLen;
        K_ = t.kernelLen;
        N_ = t.nFft;
        N1_ = t.nRadix;
        N2_ = t.nRadix2;
        chunk_ = static_cast<int32_t>(t.vecChunk);
        workers_ = static_cast<int32_t>(t.usedCoreNum) * 2;
        split_ = static_cast<int32_t>(t.batchSplit);
        workerIdx_ = VectorWorkerIdx();
        allowHf32_ = t.allowHf32;

        gGm_.SetGlobalBuffer((__gm__ float *)grad);
        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        gxGm_.SetGlobalBuffer((__gm__ float *)gx);
        gwGm_.SetGlobalBuffer((__gm__ float *)gw);
        tabGm_.SetGlobalBuffer((__gm__ float *)table, TabNumel());

        const uint64_t workspaceElements =
            static_cast<uint64_t>(workers_) * GRAD_GM_BUFFER_COUNT * static_cast<uint64_t>(N_);
        wsGm_.SetGlobalBuffer((__gm__ float *)GetUserWorkspace(workspace), workspaceElements);
        base_ = static_cast<uint64_t>(workerIdx_) * GRAD_GM_BUFFER_COUNT * N_;

        if ASCEND_IS_AIV {
            constexpr int32_t kChunkBytes = VEC_CHUNK_MAX * sizeof(float);
            pipe->InitBuffer(b0_, kChunkBytes);
            pipe->InitBuffer(b1_, kChunkBytes);
            pipe->InitBuffer(b2_, kChunkBytes);
            pipe->InitBuffer(b3_, kChunkBytes);
            pipe->InitBuffer(b4_, kChunkBytes);
            pipe->InitBuffer(b5_, kChunkBytes);
            pipe->InitBuffer(b6_, kChunkBytes);
            pipe->InitBuffer(b7_, kChunkBytes);
        }
    }

    __aicore__ inline void Process() {
        const int32_t tasks = H_ * split_;
        const int32_t bPerChunk = (B_ + split_ - 1) / split_;
        for (int32_t t = workerIdx_; t < tasks; t += workers_) {
            const int32_t h = t / split_;
            const int32_t b0 = (t % split_) * bPerChunk;
            const int32_t b1 = MinU(b0 + bPerChunk, B_);
            if (b0 >= b1) {
                continue;
            }

            LoadRowG(Buf(XT), wGm_, static_cast<uint64_t>(h) * K_, K_);
            Forward(Buf(KS));

            for (int32_t b = b0; b < b1; ++b) {
                const int32_t row = b * H_ + h;
                const uint64_t rowBase = static_cast<uint64_t>(row) * L_;

                LoadRowG(Buf(XT), gGm_, rowBase, L_);
                Forward(Buf(GS));

                CMulG(Buf(YC), true, Buf(GS), 0, true, Buf(KS), 0, true, false, true);
                Inverse(Buf(YC));
                StoreCropG(gxGm_, rowBase, L_, false);

                LoadRowG(Buf(XT), xGm_, rowBase, L_);
                Forward(Buf(XS));
                CMulAccG(Buf(ACC), Buf(XS), Buf(GS), b == b0);
            }

            Inverse(Buf(ACC));
            StoreCropG(gwGm_, static_cast<uint64_t>(h) * K_, K_, true);
        }
    }

  private:
    enum : int32_t { XS = 0, KS = 2, YS = 4, YC = 6, XT = 6, GS = 8, ACC = 10 };

    __aicore__ inline uint64_t Buf(int32_t i) const { return base_ + static_cast<uint64_t>(i) * N_; }

    __aicore__ inline uint64_t Sq1() const { return static_cast<uint64_t>(N1_) * static_cast<uint64_t>(N1_); }
    __aicore__ inline uint64_t Sq2() const { return static_cast<uint64_t>(N2_) * static_cast<uint64_t>(N2_); }
    __aicore__ inline uint64_t TabD1() const { return 0; }
    __aicore__ inline uint64_t TabTr() const { return 2ULL * Sq1(); }
    __aicore__ inline uint64_t TabTi() const { return TabTr() + static_cast<uint64_t>(N_); }
    __aicore__ inline uint64_t TabW2() const { return TabTi() + static_cast<uint64_t>(N_); }
    __aicore__ inline uint64_t TabW4() const { return TabW2() + 4ULL * Sq2(); }
    __aicore__ inline uint64_t TabID1() const { return TabW4() + 4ULL * Sq2(); }
    __aicore__ inline uint64_t TabNumel() const { return TabID1() + 2ULL * Sq1(); }

    __aicore__ inline void LoadGRaw(
        const LocalTensor<float> &d, const GlobalTensor<float> &src, uint64_t off, int32_t len) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d, src[off], cp, pad);
    }

    __aicore__ inline void LoadG(
        const LocalTensor<float> &d, const GlobalTensor<float> &src, uint64_t off, int32_t len) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadGRaw(d, src, off, len);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void StoreGRaw(uint64_t off, const LocalTensor<float> &s, int32_t len) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
        DataCopyPad(wsGm_[off], s, cp);
    }

    __aicore__ inline void StoreG(uint64_t off, const LocalTensor<float> &s, int32_t len) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        StoreGRaw(off, s, len);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void StoreRowHalfGRaw(
        uint64_t dstBase, int32_t elemOff, const LocalTensor<float> &s, int32_t len, int32_t half) {
        DataCopyExtParams cp{
            static_cast<uint16_t>(len / N2_), static_cast<uint32_t>(N2_) * 4U, 0, static_cast<uint32_t>(N2_) * 4U, 0};
        DataCopyPad(wsGm_[RowHalfOff(dstBase, elemOff, half)], s, cp);
    }

    __aicore__ inline void LoadRowHalfGRaw(const LocalTensor<float> &d, const GlobalTensor<float> &src,
        uint64_t srcBase, int32_t elemOff, int32_t len, int32_t half) {
        DataCopyExtParams cp{
            static_cast<uint16_t>(len / N2_), static_cast<uint32_t>(N2_) * 4U, static_cast<uint32_t>(N2_) * 4U, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d, src[RowHalfOff(srcBase, elemOff, half)], cp, pad);
    }

    __aicore__ inline uint64_t RowHalfOff(uint64_t base, int32_t elemOff, int32_t half) const {
        return base + static_cast<uint64_t>(elemOff / N2_) * 2ULL * static_cast<uint64_t>(N2_) +
            static_cast<uint64_t>(half) * static_cast<uint64_t>(N2_);
    }

    __aicore__ inline void LoadPairG(const LocalTensor<float> &re, const LocalTensor<float> &im,
        const GlobalTensor<float> &src, uint64_t base, uint64_t baseIm, bool inter, int32_t o, int32_t len) {
        if (inter) {
            LoadRowHalfGRaw(re, src, base, o, len, 0);
            LoadRowHalfGRaw(im, src, base, o, len, 1);
        } else {
            LoadGRaw(re, src, base + o, len);
            LoadGRaw(im, src, baseIm + o, len);
        }
    }

    __aicore__ inline void StorePairG(
        uint64_t base, bool inter, int32_t o, const LocalTensor<float> &re, const LocalTensor<float> &im, int32_t len) {
        if (inter) {
            StoreRowHalfGRaw(base, o, re, len, 0);
            StoreRowHalfGRaw(base, o, im, len, 1);
        } else {
            StoreGRaw(base + o, re, len);
            StoreGRaw(base + static_cast<uint64_t>(N_) + o, im, len);
        }
    }

    __aicore__ inline void GemmMx2(uint64_t c, uint64_t tabA, uint64_t b) {
#if __CCE_AICORE__ == 310
        if (allowHf32_) {
            mmA_.SetHF32(true, 1);
        }
#endif
        mmA_.SetTensorA(tabGm_[tabA]);
        mmA_.SetTensorB(wsGm_[b]);
        mmA_.IterateAll(wsGm_[c]);
        mmA_.End();
    }

    __aicore__ inline void GemmKx2T(uint64_t c, uint64_t tabA, uint64_t b) {
#if __CCE_AICORE__ == 310
        if (allowHf32_) {
            mmB_.SetHF32(true, 1);
        }
#endif
        mmB_.SetTensorA(tabGm_[tabA]);
        mmB_.SetTensorB(wsGm_[b]);
        mmB_.IterateAll(wsGm_[c]);
        mmB_.End();
    }

    __aicore__ inline void GemmWide(uint64_t c, uint64_t a, uint64_t tabB) {
#if __CCE_AICORE__ == 310
        if (allowHf32_) {
            mmC_.SetHF32(true, 1);
        }
#endif
        mmC_.SetTensorA(wsGm_[a]);
        mmC_.SetTensorB(tabGm_[tabB]);
        mmC_.IterateAll(wsGm_[c]);
        mmC_.End();
    }

    __aicore__ inline void CMulG(uint64_t dst, bool outInter, uint64_t oar, uint64_t oai, bool aInter, uint64_t obr,
        uint64_t obi, bool bInter, bool bFromTab, bool conjB) {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<float> ar = b0_.Get<float>();
        LocalTensor<float> ai = b1_.Get<float>();
        LocalTensor<float> br = b2_.Get<float>();
        LocalTensor<float> bi = b3_.Get<float>();
        LocalTensor<float> t0 = b4_.Get<float>();
        LocalTensor<float> t1 = b5_.Get<float>();
        LocalTensor<float> t2 = b6_.Get<float>();
        LocalTensor<float> t3 = b7_.Get<float>();
        for (int32_t o = 0; o < N_; o += chunk_) {
            const int32_t len = MinU(chunk_, N_ - o);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            LoadPairG(ar, ai, wsGm_, oar, oai, aInter, o, len);
            if (bFromTab) {
                LoadPairG(br, bi, tabGm_, obr, obi, bInter, o, len);
            } else {
                LoadPairG(br, bi, wsGm_, obr, obi, bInter, o, len);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(t0, ar, br, len);
            Mul(t1, ai, bi, len);
            Mul(t2, ar, bi, len);
            Mul(t3, ai, br, len);
            if (conjB) {
                Add(t0, t0, t1, len);
                Sub(t2, t3, t2, len);
            } else {
                Add(t2, t2, t3, len);
                Sub(t0, t0, t1, len);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            StorePairG(dst, outInter, o, t0, t2, len);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void CMulAccG(uint64_t acc, uint64_t oa, uint64_t ob, bool first) {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<float> ar = b0_.Get<float>();
        LocalTensor<float> ai = b1_.Get<float>();
        LocalTensor<float> br = b2_.Get<float>();
        LocalTensor<float> bi = b3_.Get<float>();
        LocalTensor<float> t0 = b4_.Get<float>();
        LocalTensor<float> t1 = b5_.Get<float>();
        LocalTensor<float> t2 = b6_.Get<float>();
        LocalTensor<float> t3 = b7_.Get<float>();
        for (int32_t o = 0; o < N_; o += chunk_) {
            const int32_t len = MinU(chunk_, N_ - o);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            LoadPairG(ar, ai, wsGm_, oa, 0, true, o, len);
            LoadPairG(br, bi, wsGm_, ob, 0, true, o, len);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(t0, ar, br, len);
            Mul(t1, ai, bi, len);
            Mul(t2, ar, bi, len);
            Mul(t3, ai, br, len);
            Add(t0, t0, t1, len);
            Sub(t2, t2, t3, len);
            if (!first) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                LoadPairG(ar, ai, wsGm_, acc, 0, true, o, len);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Add(t0, t0, ar, len);
                Add(t2, t2, ai, len);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            StorePairG(acc, true, o, t0, t2, len);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void Forward(uint64_t out) {
        GemmMx2(Buf(YS), TabD1(), Buf(XT));
        CMulG(Buf(YC), true, Buf(YS), Buf(YS) + static_cast<uint64_t>(N_), false, TabTr(), TabTi(), false, true, false);
        GemmWide(out, Buf(YC), TabW2());
    }

    __aicore__ inline void Inverse(uint64_t src) {
        GemmWide(Buf(XS), src, TabW4());
        CMulG(Buf(YS), false, Buf(XS), 0, true, TabTr(), TabTi(), false, true, true);
        GemmKx2T(Buf(XT), TabID1(), Buf(YS));
    }

    __aicore__ inline void ZeroRangeG(uint64_t dst, const LocalTensor<float> &a, int32_t begin, int32_t count) {
        if (count <= 0) {
            return;
        }
        Duplicate(a, 0.0f, MinU(chunk_, count));
        for (int32_t o = 0; o < count; o += chunk_) {
            StoreG(dst + static_cast<uint64_t>(begin) + o, a, MinU(chunk_, count - o));
        }
    }

    __aicore__ inline void LoadRowG(uint64_t dst, const GlobalTensor<float> &src, uint64_t srcOff, int32_t cnt) {
        if ASCEND_IS_AIC {
            return;
        }
        LocalTensor<float> a = b0_.Get<float>();
        ZeroRangeG(dst, a, cnt, N_ - cnt);
        for (int32_t o = 0; o < cnt; o += chunk_) {
            const int32_t len = MinU(chunk_, cnt - o);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyExtParams cp{1, static_cast<uint32_t>(len) * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(a, src[srcOff + o], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            StoreG(dst + o, a, len);
        }
    }

    __aicore__ inline void StoreCropG(const GlobalTensor<float> &dstGm, uint64_t dstOff, int32_t cnt, bool atomic) {
        if ASCEND_IS_AIC {
            return;
        }
        const float scale = 1.0f / static_cast<float>(N_);
        LocalTensor<float> a = b0_.Get<float>();
        for (int32_t o = 0; o < cnt; o += chunk_) {
            const int32_t n = MinU(chunk_, cnt - o);
            LoadG(a, wsGm_, Buf(XT) + o, n);
            Muls(a, a, scale, n);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (atomic) {
                SetAtomicAdd<float>();
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(n) * 4U, 0, 0, 0};
            DataCopyPad(dstGm[dstOff + o], a, cp);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (atomic) {
                SetAtomicNone();
            }
        }
    }

    GlobalTensor<float> gGm_, xGm_, wGm_, gxGm_, gwGm_, wsGm_, tabGm_;
    TBuf<TPosition::VECCALC> b0_, b1_, b2_, b3_, b4_, b5_, b6_, b7_;
    uint64_t base_;
    int32_t B_, H_, L_, K_, N_, N1_, N2_, chunk_, workers_, split_, workerIdx_;
    bool allowHf32_;
};

extern "C" __global__ __aicore__ void fft_conv1d_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR kernel, GM_ADDR table,
    GM_ADDR grad_x, GM_ADDR grad_kernel, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    if (tilingData.algo == ALGO_DIRECT) {
        if ASCEND_IS_AIC {
            return;
        }
        FftConv1dGradDirect op;
        op.Init(&pipe, grad, x, kernel, grad_x, grad_kernel, workspace, tilingData);
        op.Process();
    } else {
        FftConv1dGradFftGm op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mmA_, &tilingData.cubeTiling, op.mmB_,
            &tilingData.cubeTilingB, op.mmC_, &tilingData.cubeTilingC);
        if ASCEND_IS_AIV {
            op.Init(&pipe, grad, x, kernel, table, grad_x, grad_kernel, workspace, tilingData);
            op.Process();
        }
    }
}
