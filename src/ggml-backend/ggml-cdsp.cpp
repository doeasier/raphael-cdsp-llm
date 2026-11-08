//============================================================================
// ggml-cdsp.cpp - a ggml backend plugin that offloads GGML_OP_MUL_MAT to the
// Redmi K20 Pro cDSP (Hexagon 690 / V66) via our custom FastRPC int8 GEMM
// kernel (libgemm_skel.so, built from gemm.idl / gemm_dsp.c).
//
// Design notes (modeled on the upstream ggml-hexagon registration pattern but
// deliberately simplified for a research proof-of-concept):
//   * We only claim to support GGML_OP_MUL_MAT with an F32 activation and an
//     {F32, Q8_0, Q4_K} weight, so the ggml scheduler keeps every other op on
//     the CPU backend and only hands us MUL_MAT nodes. Our graph_compute
//     therefore only ever sees MUL_MAT and never has to delegate other ops.
//   * IMPORTANT: ggml_mul_mat(src0=weight [K,N], src1=activation [K,M]) ->
//     dst [N,M]. This is the REAL llama.cpp convention (see ggml.h); the
//     column-major data layout maps directly to Bf[n*K+k] / Ab[m*K+k] /
//     Cb[m*N+n], so no extra transpose of the host tensors is needed.
//   * Tensors live in the ordinary CPU host buffer (ggml_backend_cpu_buffer_type)
//     - FastRPC copies them to the DSP and back - so we need no custom buffer
//     implementation.
//   * For each MUL_MAT we: dequantize the (Q8_0 / Q4_K) weight to F32 then
//     quantize its columns to int8 (per-column scale), transpose, upload+prepack
//     ONCE on the DSP (weight cache), quantize the activation rows to int8
//     (per-row scale), run the cached DSP int8 GEMM with only A shipped, then
//     apply the scales on the host to recover F32 output.
//
// Build (on-device, aarch64 Linux): see build_device.sh
// Registered as backend "cdsp".
//============================================================================

extern "C" {
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // shipped copy: exact ABI for ggml 0.9.0
}

#include "cdsp_gemm_host.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// quant / dequant helpers (symmetric per-vector int8, scale = maxabs/127)
// ---------------------------------------------------------------------------
static void quant_vec(const float * x, int n, int8_t * q, float * scale) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = (x[i] < 0.f) ? -x[i] : x[i];
        if (a > mx) mx = a;
    }
    float s = (mx == 0.0f) ? 1.0f : (mx / 127.0f);
    *scale = s;
    for (int i = 0; i < n; i++) {
        int v = (int)std::lroundf(x[i] / s);
        if (v >  127) v =  127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

// IEEE 754 half -> float. ggml stores the Q8_0 / Q4_K block scales as fp16.
// Used by both dequant_q8_0() and dequant_q4_k().
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            uint32_t e = 0;
            while (!(mant & 0x400)) { mant <<= 1; e++; }
            mant &= 0x3FF;
            f = sign | ((127 - 15 - e + 1) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | (0xFFu << 23) | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

// Q8_0 layout in ggml 0.9.0: block = [fp16 d][int8 qs[32]] = 34 bytes /
// 32 weights. (Older ggml used an f32 scale / 36-byte block - do NOT use that
// here, it silently corrupts every offloaded layer.)
struct block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
};

static void dequant_q8_0(const uint8_t * w, int K, int N, float * out) {
    const int bpk    = K / 32;          // blocks along K
    const int stride = sizeof(block_q8_0); // 34
    for (int n = 0; n < N; n++) {
        for (int bk = 0; bk < bpk; bk++) {
            const block_q8_0 * blk =
                (const block_q8_0 *)(w + ((size_t)n * bpk + bk) * stride);
            const float s = fp16_to_fp32(blk->d);
            for (int j = 0; j < 32; j++) {
                out[(size_t)n * K + (size_t)bk * 32 + j] = s * (float)blk->qs[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K (K-quant, 4-bit) dequant — copied verbatim from ggml 0.9.0
// (ggml-quants.c: dequantize_row_q4_K / get_scale_min_k4). We must support this
// because real models (e.g. Qwen3.5-4B-Q4_K_M) ship Q4_K weights; without it
// the MUL_MAT never gets routed to the cDSP. Block = 144 B (d fp16, dmin fp16,
// scales[K_SCALE_SIZE=12], qs[QK_K/2=128]) holding QK_K=256 weights.
// NOTE: K_SCALE_SIZE is 12 in ggml 0.9.0 (ggml-common.h), NOT 8 - using 8
// silently shifts every subsequent block and produces garbage outputs.
// ---------------------------------------------------------------------------
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// fp16_to_fp32() is defined above (shared with dequant_q8_0).
static void dequant_q4_k(const uint8_t * w, int K, int N, float * out) {
    const int nb = K / 256;   // QK_K = 256
    const block_q4_K * x = (const block_q4_K *)w;
    float * y = out;
    for (int n = 0; n < N; n++) {
        const block_q4_K * xb = x + (size_t)n * nb;
        for (int i = 0; i < nb; i++) {
            const uint8_t * q = xb[i].qs;
            const float d   = fp16_to_fp32(xb[i].d);
            const float min = fp16_to_fp32(xb[i].dmin);
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, xb[i].scales, &sc, &m);
                const float d1 = d * sc; const float m1 = min * m;
                get_scale_min_k4(is + 1, xb[i].scales, &sc, &m);
                const float d2 = d * sc; const float m2 = min * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4)   - m2;
                q += 32; is += 2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WEIGHT CACHE (Stage 5)
//   A 4B model's weight matrices are tens of MB each. Shipping them over
//   FastRPC on every token is what made the naive path unusable. Instead we
//   upload + prepack each unique weight ONCE on the DSP and keep a handle;
//   every subsequent MUL_MAT only ships the small activation A.
//
//   The cache is keyed by the weight tensor's data pointer + its in-tensor
//   batch index. It is an LRU with a byte capacity (default 256 MB, tunable via
//   GGML_CDSP_CACHE_MB) so we never try to hold the entire model on the cDSP
//   (which cannot fit a 4B model as int8). On eviction we release the DSP
//   handle (freeing its prepacked buffer).
// ---------------------------------------------------------------------------
struct CachedWeight {
    int32_t            handle;
    int                N, K;
    std::vector<float> sB;          // per-column weight scales (N)
    size_t             packed_bytes; // DSP-side prepacked size (capacity acct)
    long long          last;         // LRU timestamp
};

struct WKey {
    const void * p;
    int64_t      b1;
    bool operator==(const WKey & o) const { return p == o.p && b1 == o.b1; }
};
struct WKeyHash {
    size_t operator()(const WKey & k) const {
        return std::hash<const void*>()(k.p) ^ (std::hash<int64_t>()(k.b1) << 1);
    }
};

static std::unordered_map<WKey, CachedWeight, WKeyHash> g_wcache;
static size_t   g_wcache_bytes = 0;
static long long g_wcache_tick = 0;
static size_t   g_wcache_cap   = 256ULL * 1024 * 1024;  // tunable via env
static int      g_wcache_hits  = 0;
static int      g_wcache_misses = 0;

static void wcache_init_cap(void) {
    static bool done = false;
    if (done) return;
    done = true;
    const char * e = getenv("GGML_CDSP_CACHE_MB");
    if (e) { long long mb = atoll(e); if (mb > 0) g_wcache_cap = (size_t)mb * 1024 * 1024; }
}

static void wcache_evict_to_fit(size_t incoming) {
    // Evict least-recently-used entries until it fits. The entry we are about
    // to use always has the newest `last`, so it is never chosen.
    while (!g_wcache.empty() &&
           g_wcache_bytes + incoming > g_wcache_cap &&
           g_wcache.size() > 1) {
        auto mit = g_wcache.begin();
        for (auto it = g_wcache.begin(); it != g_wcache.end(); ++it)
            if (it->second.last < mit->second.last) mit = it;
        cdsp_gemm_release(mit->second.handle);
        g_wcache_bytes -= mit->second.packed_bytes;
        g_wcache.erase(mit);
    }
}

static void wcache_release_all(void) {
    for (auto & kv : g_wcache) cdsp_gemm_release(kv.second.handle);
    g_wcache.clear();
    g_wcache_bytes = 0;
    g_wcache_hits = g_wcache_misses = 0;
}

// ---------------------------------------------------------------------------
// Capability predicate. supports_op() and cdsp_mul_mat() MUST use the SAME
// predicate: if the scheduler hands us an op that compute then rejects,
// ggml_backend_sched_graph_compute_async() aborts the whole graph with -1
// (observed on a real Qwen3.5-4B run where an extra batched MUL_MAT slipped
// through). Keep every guard here and nowhere else.
//
// GGML_CDSP_MAX_ELEMS (optional): skip matrices whose weight element count
// N*K exceeds this value, leaving them on the CPU. Useful for large models
// where shipping every weight to the cDSP thrashes the cache (see §1.1 of
// docs/next-steps.md). Unset / 0 = offload everything (full-offload mode).
// ---------------------------------------------------------------------------
static bool cdsp_can_mul_mat(const struct ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT) return false;
    const struct ggml_tensor * w = op->src[0]; // weight     [K, N]
    const struct ggml_tensor * a = op->src[1]; // activation [K, M]
    if (!w || !a) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (a->type != GGML_TYPE_F32) return false;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_Q8_0
        && w->type != GGML_TYPE_Q4_K) return false;

    const int64_t K = w->ne[0], N = w->ne[1], M = a->ne[1];
    if (K <= 0 || M <= 0 || N <= 0) return false;
    if (a->ne[0] != K) return false;
    const int64_t B0 = a->ne[2] * a->ne[3];
    const int64_t B1 = w->ne[2] * w->ne[3];
    if (B1 != 1 && B1 != B0) return false;
    if (w->type == GGML_TYPE_Q8_0 && (K % 32)  != 0) return false;
    if (w->type == GGML_TYPE_Q4_K && (K % 256) != 0) return false;

    const char * e = getenv("GGML_CDSP_MAX_ELEMS");
    if (e && *e) {
        long long lim = atoll(e);
        if (lim > 0 && (long long)(N * K) > lim) return false;
    }
    // GGML_CDSP_MIN_ELEMS: leave tiny matmuls on the CPU. A real Qwen3.5-4B
    // decode issues ~37 MUL_MATs of shape 1x32x2560 per token; each costs a
    // FastRPC round-trip (~0.25 ms) for only 82k MACs, so offloading them is
    // pure overhead. Skipping them cuts ~75% of the per-token RPC calls.
    const char * emn = getenv("GGML_CDSP_MIN_ELEMS");
    if (emn && *emn) {
        long long lim = atoll(emn);
        if (lim > 0 && (long long)(N * K) < lim) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// the actual MUL_MAT compute, dispatched to the cDSP
// ---------------------------------------------------------------------------
static enum ggml_status cdsp_mul_mat(struct ggml_tensor * dst) {
    // ggml MUL_MAT semantics (ggml.h / ggml_mul_mat):
    //   src0 = weight     [K, N]  (F32 / Q8_0 / Q4_K)  -> result->ne[0] = N
    //   src1 = activation [K, M]  (F32)                -> result->ne[1] = M
    //   dst  [N, M], column-major: dst[n + m*N] = sum_k w[k + n*K] * a[k + m*K]
    // The column-major layout matches our Bf[n*K+k] / Ab[m*K+k] / Cb[m*N+n].
    if (!cdsp_can_mul_mat(dst)) return GGML_STATUS_FAILED;
    struct ggml_tensor * w = dst->src[0]; // weight
    struct ggml_tensor * a = dst->src[1]; // activation

    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    const int64_t M = a->ne[1];
    const int64_t B0 = a->ne[2] * a->ne[3];          // activation batches
    const int64_t B1 = w->ne[2] * w->ne[3];          // weight batches

    if (getenv("GGML_CDSP_DEBUG"))
        fprintf(stderr, "[ggml-cdsp] offloaded MUL_MAT M=%lld N=%lld K=%lld\n",
                (long long)M, (long long)N, (long long)K);

    if (getenv("GGML_CDSP_TRACE"))
        fprintf(stderr,
            "[ggml-cdsp] TRACE w(type=%d ne=[%lld,%lld,%lld,%lld] "
            "nb=[%lld,%lld,%lld,%lld] contig=%d nbytes=%zu expect=%zu) "
            "a(type=%d ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d) "
            "dst(ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d)\n",
            (int)w->type, (long long)w->ne[0], (long long)w->ne[1],
            (long long)w->ne[2], (long long)w->ne[3],
            (long long)w->nb[0], (long long)w->nb[1],
            (long long)w->nb[2], (long long)w->nb[3],
            (int)ggml_is_contiguous(w), ggml_nbytes(w),
            (w->type == GGML_TYPE_Q4_K)
                ? (size_t)N * (size_t)(K / 256) * sizeof(block_q4_K)
                : (w->type == GGML_TYPE_Q8_0)
                    ? (size_t)N * (size_t)(K / 32) * sizeof(block_q8_0)
                    : (size_t)N * (size_t)K * sizeof(float),
            (int)a->type, (long long)a->ne[0], (long long)a->ne[1],
            (long long)a->ne[2], (long long)a->ne[3],
            (long long)a->nb[0], (long long)a->nb[1],
            (long long)a->nb[2], (long long)a->nb[3],
            (int)ggml_is_contiguous(a),
            (long long)dst->ne[0], (long long)dst->ne[1],
            (long long)dst->ne[2], (long long)dst->ne[3],
            (long long)dst->nb[0], (long long)dst->nb[1],
            (long long)dst->nb[2], (long long)dst->nb[3],
            (int)ggml_is_contiguous(dst));

    const float   * A = (const float   *)a->data;
    const uint8_t * W = (const uint8_t *)w->data;
    float         * C = (float         *)dst->data;

    const size_t wbatch = ggml_nbytes(w) / (size_t)(B1 > 0 ? B1 : 1);

    std::vector<int8_t>  Aq((size_t)M * K);
    std::vector<float>   sA((size_t)M);

    wcache_init_cap();
    const bool no_cache = getenv("GGML_CDSP_NO_CACHE") != nullptr;
    const bool host_f32 = getenv("GGML_CDSP_HOST_F32") != nullptr;

    // ---- diagnostic mode: exact F32 GEMM on the host, no int8 / no DSP ----
    // Uses the SAME dequant helpers as the real path. If model output is correct
    // here but not in the normal path, the dequant/layout code is fine and the
    // fault lies in int8 quantization, the DSP kernel, or the weight cache.
    if (host_f32) {
        for (int64_t b = 0; b < B0; b++) {
            const int64_t   b1 = (B1 == 1) ? 0 : b;
            const float   * Ab = A  + b * M * K;
            const uint8_t * Wb = W  + b1 * wbatch;
            float         * Cb = C  + b * M * N;
            std::vector<float> Bf((size_t)N * K);
            if (w->type == GGML_TYPE_F32)
                memcpy(Bf.data(), Wb, (size_t)N * K * sizeof(float));
            else if (w->type == GGML_TYPE_Q8_0)
                dequant_q8_0(Wb, (int)K, (int)N, Bf.data());
            else
                dequant_q4_k(Wb, (int)K, (int)N, Bf.data());
            for (int64_t m = 0; m < M; m++)
                for (int64_t n = 0; n < N; n++) {
                    float acc = 0.f;
                    for (int64_t k = 0; k < K; k++)
                        acc += Ab[(size_t)m * K + k] * Bf[(size_t)n * K + k];
                    Cb[(size_t)m * N + n] = acc;
                }
        }
        return GGML_STATUS_SUCCESS;
    }

    for (int64_t b = 0; b < B0; b++) {
        int64_t b1 = (B1 == 1) ? 0 : b;
        const float  * Ab  = A  + b * M * K;
        const uint8_t * Wb  = W  + b1 * wbatch;
        float         * Cb  = C  + b * M * N;

        // ---- weight cache lookup (keyed by pointer + in-tensor batch) ----
        // GGML_CDSP_NO_CACHE bypasses the cache entirely: upload every time,
        // release immediately. Useful to detect STALE cache entries, i.e. a
        // dynamic tensor that reuses the same buffer but changes its contents
        // between forwards (e.g. quantized KV-cache slices).
        WKey key{ (const void *)Wb, b1 };
        auto it = no_cache ? g_wcache.end() : g_wcache.find(key);
        CachedWeight local;
        CachedWeight * pcw = nullptr;
        if (it == g_wcache.end()) {
            g_wcache_misses++;
            // 1. weight Wb (N x K) -> F32 (dequant Q8_0 / Q4_K if needed)
            std::vector<float> Bf((size_t)N * K);
            if (w->type == GGML_TYPE_F32) {
                memcpy(Bf.data(), Wb, (size_t)N * K * sizeof(float));
            } else if (w->type == GGML_TYPE_Q8_0) {
                dequant_q8_0(Wb, (int)K, (int)N, Bf.data());
            } else {
                dequant_q4_k(Wb, (int)K, (int)N, Bf.data());
            }
            // 2. quantize weight per column -> Bq (N x K) + sB (N scales)
            std::vector<int8_t>  Bq((size_t)N * K);
            std::vector<float>   sB((size_t)N);
            for (int64_t n = 0; n < N; n++)
                quant_vec(Bf.data() + n * K, (int)K, Bq.data() + n * K, &sB[n]);
            // 3. transpose Bq (N x K) -> Bt (K x N): Bt[k][n] = Bq[n][k]
            std::vector<int8_t> Bt((size_t)K * N);
            for (int64_t n = 0; n < N; n++)
                for (int64_t k = 0; k < K; k++)
                    Bt[(size_t)k * N + n] = Bq[(size_t)n * K + k];
            // 4. upload + prepack on DSP -> handle (sent over FastRPC ONCE)
            int32_t handle;
            int rc = cdsp_gemm_upload(Bt.data(), (int)K, (int)N, &handle);
            if (rc != 0) {
                fprintf(stderr, "ggml-cdsp: DSP weight upload failed rc=%d\n", rc);
                return GGML_STATUS_FAILED;
            }
            const int Kblk = (K + 3) / 4, nT = (N + 31) / 32;
            size_t packed = (size_t)nT * Kblk * 128;
            local = CachedWeight{ handle, (int)N, (int)K, std::move(sB), packed,
                                  ++g_wcache_tick };
            if (no_cache) {
                pcw = &local;
            } else {
                wcache_evict_to_fit(packed);
                g_wcache[key] = std::move(local);
                g_wcache_bytes += packed;
                it = g_wcache.find(key);
                pcw = &it->second;
            }
        } else {
            g_wcache_hits++;
            pcw = &it->second;
        }
        pcw->last = ++g_wcache_tick;
        const CachedWeight & cw = *pcw;

        // 5. quantize activation A (M x K) per row
        for (int64_t m = 0; m < M; m++)
            quant_vec(Ab + m * K, (int)K, Aq.data() + m * K, &sA[m]);

        // 6. DSP int8 GEMM via cached weight: C32 = Aq @ cached B (only A sent)
        std::vector<int32_t> C32((size_t)M * N);
        int rc = cdsp_gemm_compute_cached(Aq.data(), (int)M, (int)K, (int)N,
                                          cw.handle, C32.data());
        if (rc != 0) {
            fprintf(stderr, "ggml-cdsp: DSP cached GEMM failed rc=%d\n", rc);
            return GGML_STATUS_FAILED;
        }

        // 7. recover F32: C[m][n] = sA[m] * sB[n] * C32[m][n]
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++)
                Cb[(size_t)m * N + n] =
                    sA[m] * cw.sB[n] * (float)C32[(size_t)m * N + n];

        // In diagnostic no-cache mode `local` is a stack entry not owned by
        // g_wcache, so release its DSP handle explicitly here (otherwise every
        // offloaded MUL_MAT leaks one prepacked weight buffer on the cDSP).
        if (no_cache && pcw == &local) cdsp_gemm_release(local.handle);
    }

    if (getenv("GGML_CDSP_DEBUG"))
        fprintf(stderr, "[ggml-cdsp] MUL_MAT M=%lld N=%lld K=%lld "
                "cache hits=%d misses=%d bytes=%zu\n",
                (long long)M, (long long)N, (long long)K,
                g_wcache_hits, g_wcache_misses, g_wcache_bytes);

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Split statistics (Stage 7 feasibility probe for batched RPC)
//
// ggml_backend_sched hands the ACCEL backend a whole "split" through
// graph_compute(). If a split contains several independent MUL_MAT nodes,
// they can be shipped to the DSP in one FastRPC call. If every split has
// exactly one node, batching at this level cannot help and the RPC cost must
// instead be attacked by fusing host-side work.
//
// Enable with GGML_CDSP_SPLIT_STATS=1. The histogram is printed at .so unload
// so the normal log stays clean.
// ---------------------------------------------------------------------------
static const bool g_split_stats_on = getenv("GGML_CDSP_SPLIT_STATS") != nullptr;
static uint64_t g_split_calls = 0;
// Stage 7 batch counters, dumped together with the split histogram below.
static uint64_t g_batch_splits = 0;
static uint64_t g_batch_ops    = 0;
static uint64_t g_batch_saved  = 0;
static uint64_t g_split_nodes = 0;
static uint64_t g_split_mm    = 0;
static uint64_t g_split_multi = 0;
static uint64_t g_split_hist[9] = {};   // index 8 means "8 or more"

__attribute__((destructor))
static void cdsp_split_stats_dump(void) {
    if (!g_split_stats_on && g_batch_ops == 0) return;
    if (g_split_stats_on && g_split_calls > 0) {
        fprintf(stderr,
                "[cdsp-split] calls=%llu nodes=%llu mul_mat=%llu "
                "splits_with_>1_mm=%llu\n",
                (unsigned long long)g_split_calls,
                (unsigned long long)g_split_nodes,
                (unsigned long long)g_split_mm,
                (unsigned long long)g_split_multi);
        fprintf(stderr, "[cdsp-split] mm-per-split histogram:");
        for (int i = 0; i < 9; i++)
            fprintf(stderr, " %d%s=%llu", i, i == 8 ? "+" : "",
                    (unsigned long long)g_split_hist[i]);
        fprintf(stderr, "\n");
    }
    if (g_batch_ops > 0)
        fprintf(stderr,
                "[cdsp-batch] splits=%llu ops=%llu rpcs_saved=%llu\n",
                (unsigned long long)g_batch_splits,
                (unsigned long long)g_batch_ops,
                (unsigned long long)g_batch_saved);
}

// ---------------------------------------------------------------------------
// Stage 7: batched RPC for the MUL_MAT nodes of ONE scheduler split
//
// ggml_backend_sched hands the ACCEL backend a whole "split" through
// graph_compute(). Measured on Qwen3.5-4B (24 tokens): 208 splits, 286
// MUL_MAT nodes, 78 splits contain 2 nodes. Those 78 round-trips can be
// eliminated by shipping both ops in a single FastRPC call.
//
// Encoding (no IDL change!): cdsp_gemm_compute_cached with
//   dims = { -count, M0,N0,K0,h0, M1,N1,K1,h1, ... }
//   A    = concat(A_i), C = concat(C_i)
// See src/dsp/gemm_dsp.c :: cdsp_gemm_gemm_compute_cached().
//
// Enable with GGML_CDSP_BATCH=1. When disabled (default) every op uses the
// original scalar cdsp_mul_mat() path, so this is a clean A/B switch.
// ---------------------------------------------------------------------------
#define CDSP_BATCH_MAX 8

static const bool g_batch_on = getenv("GGML_CDSP_BATCH") != nullptr;

struct BatchOp {
    int M = 0, N = 0, K = 0;
    int32_t handle       = 0;
    int32_t local_handle = 0;  // no-cache diagnostic: release after submit
    std::vector<int8_t> Aq;
    std::vector<float>  sA, sB;
    float   * Cb  = nullptr;
    int32_t * C32 = nullptr;   // view into the shared output buffer
};

// Prepare one MUL_MAT WITHOUT issuing its RPC. Returns false when the op is
// not batchable; the caller then falls back to scalar cdsp_mul_mat().
static bool cdsp_prepare_op(struct ggml_tensor * dst, BatchOp & op) {
    if (!cdsp_can_mul_mat(dst)) return false;
    struct ggml_tensor * w = dst->src[0];
    struct ggml_tensor * a = dst->src[1];
    const int64_t K = w->ne[0], N = w->ne[1], M = a->ne[1];
    const int64_t B0 = a->ne[2] * a->ne[3];
    const int64_t B1 = w->ne[2] * w->ne[3];
    if (B0 != 1 || B1 > 1) return false;        // only single-batch shapes
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (M > INT_MAX || N > INT_MAX || K > INT_MAX) return false;
    if (getenv("GGML_CDSP_HOST_F32")) return false;  // diagnostics: scalar path

    if (getenv("GGML_CDSP_DEBUG"))
        fprintf(stderr, "[ggml-cdsp] offloaded MUL_MAT M=%lld N=%lld K=%lld\n",
                (long long)M, (long long)N, (long long)K);

    const float   * A = (const float   *)a->data;
    const uint8_t * W = (const uint8_t *)w->data;

    wcache_init_cap();
    const bool no_cache = getenv("GGML_CDSP_NO_CACHE") != nullptr;

    WKey key{ (const void *)W, 0 };
    auto it = no_cache ? g_wcache.end() : g_wcache.find(key);
    CachedWeight local;
    CachedWeight * pcw = nullptr;
    if (it == g_wcache.end()) {
        g_wcache_misses++;
        std::vector<float> Bf((size_t)N * K);
        if (w->type == GGML_TYPE_F32)
            memcpy(Bf.data(), W, (size_t)N * K * sizeof(float));
        else if (w->type == GGML_TYPE_Q8_0)
            dequant_q8_0(W, (int)K, (int)N, Bf.data());
        else
            dequant_q4_k(W, (int)K, (int)N, Bf.data());
        std::vector<int8_t> Bq((size_t)N * K);
        std::vector<float>  sB((size_t)N);
        for (int64_t n = 0; n < N; n++)
            quant_vec(Bf.data() + n * K, (int)K, Bq.data() + n * K, &sB[n]);
        std::vector<int8_t> Bt((size_t)K * N);
        for (int64_t n = 0; n < N; n++)
            for (int64_t k = 0; k < K; k++)
                Bt[(size_t)k * N + n] = Bq[(size_t)n * K + k];
        int32_t handle;
        int rc = cdsp_gemm_upload(Bt.data(), (int)K, (int)N, &handle);
        if (rc != 0) {
            fprintf(stderr, "ggml-cdsp: DSP weight upload failed rc=%d\n", rc);
            return false;
        }
        const int Kblk = (K + 3) / 4, nT = (N + 31) / 32;
        const size_t packed = (size_t)nT * Kblk * 128;
        local = CachedWeight{ handle, (int)N, (int)K, std::move(sB), packed,
                              ++g_wcache_tick };
        if (no_cache) {
            pcw = &local;
        } else {
            wcache_evict_to_fit(packed);
            g_wcache[key] = std::move(local);
            g_wcache_bytes += packed;
            it = g_wcache.find(key);
            pcw = &it->second;
        }
    } else {
        g_wcache_hits++;
        pcw = &it->second;
    }
    pcw->last = ++g_wcache_tick;

    op.M = (int)M; op.N = (int)N; op.K = (int)K;
    op.handle = pcw->handle;
    op.sB = pcw->sB;        // copy: g_wcache may rehash for the next op
    op.Cb = (float *)dst->data;
    if (no_cache && pcw == &local) op.local_handle = local.handle;

    op.Aq.resize((size_t)M * K);
    op.sA.resize((size_t)M);
    for (int64_t m = 0; m < M; m++)
        quant_vec(A + m * K, (int)K, op.Aq.data() + m * K, &op.sA[m]);
    return true;
}

static void cdsp_finish_op(BatchOp & op) {
    for (int m = 0; m < op.M; m++)
        for (int n = 0; n < op.N; n++)
            op.Cb[(size_t)m * op.N + n] =
                op.sA[m] * op.sB[n] * (float)op.C32[(size_t)m * op.N + n];
    if (op.local_handle) {
        cdsp_gemm_release(op.local_handle);
        op.local_handle = 0;
    }
}

// ---------------------------------------------------------------------------
// backend iface
// ---------------------------------------------------------------------------
static const char * cdsp_backend_get_name(ggml_backend_t) { return "cdsp"; }

static void cdsp_backend_free(ggml_backend_t backend) {
    wcache_release_all();   // release all cached DSP weight handles
    delete backend;
}

static void cdsp_backend_synchronize(ggml_backend_t) { /* compute is synchronous */ }

static enum ggml_status cdsp_graph_compute(ggml_backend_t backend,
                                           struct ggml_cgraph * graph) {
    (void)backend;
    // ggml_cgraph is opaque in the public headers, so use the graph accessor API
    const int n_nodes = ggml_graph_n_nodes(graph);
    if (g_split_stats_on) {
        int mm = 0;
        for (int i = 0; i < n_nodes; i++)
            if (ggml_graph_node(graph, i)->op == GGML_OP_MUL_MAT) mm++;
        g_split_calls++;
        g_split_nodes += (uint64_t)n_nodes;
        g_split_mm    += (uint64_t)mm;
        if (mm > 1) g_split_multi++;
        g_split_hist[mm < 8 ? mm : 8]++;
    }

    // ---- Stage 7: try to ship all INDEPENDENT MUL_MATs of this split at once
    if (g_batch_on && n_nodes >= 2) {
        std::vector<ggml_tensor *> mm;
        mm.reserve(n_nodes);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_MUL_MAT) mm.push_back(t);
        }
        // A node that consumes another MUL_MAT of this split cannot be merged
        // (its activation would not exist yet). Views are resolved to their
        // parent tensor before the check.
        bool indep = mm.size() >= 2 && mm.size() <= CDSP_BATCH_MAX;
        for (size_t i = 0; indep && i < mm.size(); i++) {
            struct ggml_tensor * src[2] = { mm[i]->src[0], mm[i]->src[1] };
            for (int k = 0; k < 2 && indep; k++) {
                struct ggml_tensor * p = src[k];
                if (!p) continue;
                if (p->view_src) p = p->view_src;
                for (size_t j = 0; j < mm.size(); j++)
                    if (p == mm[j]) { indep = false; break; }
            }
        }

        if (indep) {
            std::vector<BatchOp> ops(mm.size());
            bool ok = true;
            for (size_t i = 0; i < mm.size(); i++)
                if (!cdsp_prepare_op(mm[i], ops[i])) { ok = false; break; }
            if (ok) {
                size_t aLen = 0, cLen = 0;
                for (const BatchOp & o : ops) {
                    aLen += (size_t)o.M * o.K;
                    cLen += (size_t)o.M * o.N;
                }
                std::vector<int8_t>  Abuf(aLen);
                std::vector<int32_t> Cbuf(cLen, 0);
                std::vector<int32_t> dims;
                dims.reserve(1 + 4 * ops.size());
                dims.push_back(-(int32_t)ops.size());
                size_t ao = 0, co = 0;
                for (BatchOp & o : ops) {
                    memcpy(Abuf.data() + ao, o.Aq.data(), o.Aq.size());
                    ao += o.Aq.size();
                    dims.push_back(o.M); dims.push_back(o.N);
                    dims.push_back(o.K); dims.push_back(o.handle);
                    o.C32 = Cbuf.data() + co;
                    co += (size_t)o.M * o.N;
                }
                int rc = cdsp_gemm_compute_multi(Abuf.data(), (int)ao,
                                                 dims.data(), (int)dims.size(),
                                                 Cbuf.data(), (int)co);
                if (rc == 0) {
                    for (BatchOp & o : ops) cdsp_finish_op(o);
                    g_batch_splits++;
                    g_batch_ops += ops.size();
                    g_batch_saved += ops.size() - 1;
                    return GGML_STATUS_SUCCESS;
                }
                // e.g. a cached handle was evicted between prepares. Re-run
                // every node through the scalar path (which re-uploads).
                fprintf(stderr,
                        "[ggml-cdsp] batched RPC rc=%d, falling back to scalar\n",
                        rc);
            }
            for (BatchOp & o : ops)
                if (o.local_handle) {
                    cdsp_gemm_release(o.local_handle);
                    o.local_handle = 0;
                }
        }
    }

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * n = ggml_graph_node(graph, i);
        if (n->op == GGML_OP_MUL_MAT) {
            enum ggml_status st = cdsp_mul_mat(n);
            if (st != GGML_STATUS_SUCCESS) return st;
        }
        // any other op type is handled by the CPU backend (scheduler does not
        // assign it to us because supports_op() returns false for it)
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i cdsp_backend_i = {
    /*  0 get_name              */ cdsp_backend_get_name,
    /*  1 free                  */ cdsp_backend_free,
    /*  2 set_tensor_async      */ nullptr,
    /*  3 get_tensor_async      */ nullptr,
    /*  4 set_tensor_2d_async   */ nullptr,
    /*  5 get_tensor_2d_async   */ nullptr,
    /*  6 cpy_tensor_async      */ nullptr,
    /*  7 synchronize           */ cdsp_backend_synchronize,
    /*  8 graph_plan_create     */ nullptr,
    /*  9 graph_plan_free       */ nullptr,
    /* 10 graph_plan_update     */ nullptr,
    /* 11 graph_plan_compute    */ nullptr,
    /* 12 graph_compute         */ cdsp_graph_compute,
    /* 13 event_record          */ nullptr,
    /* 14 event_wait            */ nullptr,
    /* 15 graph_optimize        */ nullptr,
};

static ggml_guid_t cdsp_guid(void) {
    static ggml_guid guid = { 0xCD, 0x53, 0x50, 0x44, 0x00, 0x01, 0x02, 0x03,
                              0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B };
    return &guid;
}

// ---------------------------------------------------------------------------
// device iface
// ---------------------------------------------------------------------------
static const char * cdsp_dev_get_name(ggml_backend_dev_t)     { return "cdsp"; }
static const char * cdsp_dev_get_description(ggml_backend_dev_t) { return "Redmi K20 Pro cDSP (Hexagon 690)"; }

static void cdsp_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0; *total = 0; // no dedicated memory to report
}

static enum ggml_backend_dev_type cdsp_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL; // accelerator used with the CPU
}

static void cdsp_dev_get_props(ggml_backend_dev_t dev,
                               struct ggml_backend_dev_props * props) {
    props->name        = cdsp_dev_get_name(dev);
    props->description = cdsp_dev_get_description(dev);
    props->type        = cdsp_dev_get_type(dev);
    props->device_id   = nullptr;
    cdsp_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_buffer_type_t cdsp_dev_get_buffer_type(ggml_backend_dev_t) {
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t cdsp_dev_get_host_buffer_type(ggml_backend_dev_t) {
    return ggml_backend_cpu_buffer_type();
}

static bool cdsp_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    // Single source of truth: must agree with cdsp_mul_mat() exactly, or the
    // scheduler assigns us an op we reject and the whole graph fails.
    return cdsp_can_mul_mat(op);
}

static bool cdsp_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft); // we use host (CPU) buffers
}

static ggml_backend_t cdsp_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)params;
    return new ggml_backend{
        /* .guid     = */ cdsp_guid(),
        /* .iface    = */ cdsp_backend_i,
        /* .device   = */ dev,
        /* .context  = */ dev->context,
    };
}

static const ggml_backend_device_i cdsp_device_i = {
    /*  0 get_name             */ cdsp_dev_get_name,
    /*  1 get_description      */ cdsp_dev_get_description,
    /*  2 get_memory           */ cdsp_dev_get_memory,
    /*  3 get_type             = */ cdsp_dev_get_type,
    /*  4 get_props            */ cdsp_dev_get_props,
    /*  5 init_backend         */ cdsp_dev_init_backend,
    /*  6 get_buffer_type      */ cdsp_dev_get_buffer_type,
    /*  7 get_host_buffer_type */ cdsp_dev_get_host_buffer_type,
    /*  8 buffer_from_host_ptr */ nullptr,
    /*  9 supports_op          */ cdsp_dev_supports_op,
    /* 10 supports_buft        */ cdsp_dev_supports_buft,
    /* 11 offload_op           */ nullptr,
    /* 12 event_new            */ nullptr,
    /* 13 event_free           */ nullptr,
    /* 14 event_synchronize    */ nullptr,
};

// single static device
static struct cdsp_dev_ctx { /* placeholder for future per-device state */ } g_cdsp_dev_ctx;

static ggml_backend_device g_cdsp_device = {
    /* .iface    = */ cdsp_device_i,
    /* .reg      = */ nullptr,
    /* .context  = */ &g_cdsp_dev_ctx,
};

// ---------------------------------------------------------------------------
// reg iface
// ---------------------------------------------------------------------------
static const char * cdsp_reg_get_name(ggml_backend_reg_t) { return "cdsp"; }

static size_t cdsp_reg_get_device_count(ggml_backend_reg_t) { return 1; }

static ggml_backend_dev_t cdsp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) return nullptr;
    g_cdsp_device.reg = reg;
    return &g_cdsp_device;
}

static const ggml_backend_reg_i cdsp_reg_i = {
    /* .get_name         = */ cdsp_reg_get_name,
    /* .get_device_count = */ cdsp_reg_get_device_count,
    /* .get_device       = */ cdsp_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg cdsp_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ cdsp_reg_i,
    /* .context     = */ nullptr,
};

extern "C" ggml_backend_reg_t ggml_backend_cdsp_reg(void) {
    return &cdsp_reg;
}

// When built with -DGGML_BACKEND_DL this defines the ggml_backend_init()
// symbol that ggml_backend_load() looks up in the .so.
GGML_BACKEND_DL_IMPL(ggml_backend_cdsp_reg)
