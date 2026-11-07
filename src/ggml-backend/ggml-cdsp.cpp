//============================================================================
// ggml-cdsp.cpp - a ggml backend plugin that offloads GGML_OP_MUL_MAT to the
// Redmi K20 Pro cDSP (Hexagon 690 / V66) via our custom FastRPC int8 GEMM
// kernel (libgemm_skel.so, built from gemm.idl / gemm_dsp.c).
//
// Design notes (modeled on the upstream ggml-hexagon registration pattern but
// deliberately simplified for a research proof-of-concept):
//   * We only claim to support GGML_OP_MUL_MAT (F32 act x {F32,Q8_0} weight),
//     so the ggml scheduler keeps every other op on the CPU backend and only
//     hands us MUL_MAT nodes. Our graph_compute therefore only ever sees
//     MUL_MAT and never has to delegate other ops.
//   * Tensors live in the ordinary CPU host buffer (ggml_backend_cpu_buffer_type)
//     - FastRPC copies them to the DSP and back - so we need no custom buffer
//     implementation.
//   * For each MUL_MAT we: quantize the activation rows to int8 (per-row
//     scale), dequantize the (possibly Q8_0) weight to F32 then quantize its
//     columns to int8 (per-column scale), transpose the weight, run the DSP
//     int8 GEMM, then apply the scales on the host to recover F32 output.
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

// Q8_0 layout per (K,N) weight block: [f32 scale][int8 x32] repeated over K/32.
static void dequant_q8_0(const uint8_t * w, int K, int N, float * out) {
    const int bpk    = K / 32;          // blocks along K
    const int stride = 36;              // 4 (scale) + 32 (int8)
    for (int n = 0; n < N; n++) {
        for (int bk = 0; bk < bpk; bk++) {
            const uint8_t * blk = w + ((size_t)n * bpk + bk) * stride;
            float s = *(const float *)blk;
            const int8_t * q = (const int8_t *)(blk + 4);
            for (int j = 0; j < 32; j++) {
                out[(size_t)n * K + (size_t)bk * 32 + j] = s * (float)q[j];
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
// the actual MUL_MAT compute, dispatched to the cDSP
// ---------------------------------------------------------------------------
static enum ggml_status cdsp_mul_mat(struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0]; // activation (F32)
    struct ggml_tensor * src1 = dst->src[1]; // weight     (F32 or Q8_0)

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t B0 = src0->ne[2] * src0->ne[3];          // activation batches
    const int64_t B1 = src1->ne[2] * src1->ne[3];          // weight batches

    if (K <= 0 || M <= 0 || N <= 0)            return GGML_STATUS_FAILED;
    if (src0->type != GGML_TYPE_F32)           return GGML_STATUS_FAILED;
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_0)
                                                return GGML_STATUS_FAILED;
    if (B1 != 1 && B1 != B0)                   return GGML_STATUS_FAILED;

    if (getenv("GGML_CDSP_DEBUG"))
        fprintf(stderr, "[ggml-cdsp] offloaded MUL_MAT M=%lld N=%lld K=%lld\n",
                (long long)M, (long long)N, (long long)K);

    const float  * A = (const float  *)src0->data;
    const uint8_t * W = (const uint8_t *)src1->data;
    float         * C = (float         *)dst->data;

    const size_t wbatch = ggml_nbytes(src1) / (size_t)(B1 > 0 ? B1 : 1);

    std::vector<int8_t>  Aq((size_t)M * K);
    std::vector<float>   sA((size_t)M);

    wcache_init_cap();

    for (int64_t b = 0; b < B0; b++) {
        int64_t b1 = (B1 == 1) ? 0 : b;
        const float  * Ab  = A  + b * M * K;
        const uint8_t * Wb  = W  + b1 * wbatch;
        float         * Cb  = C  + b * M * N;

        // ---- weight cache lookup (keyed by pointer + in-tensor batch) ----
        WKey key{ (const void *)Wb, b1 };
        auto it = g_wcache.find(key);
        if (it == g_wcache.end()) {
            g_wcache_misses++;
            // 1. weight Wb (N x K) -> F32 (dequant Q8_0 if needed)
            std::vector<float> Bf((size_t)N * K);
            if (src1->type == GGML_TYPE_F32) {
                memcpy(Bf.data(), Wb, (size_t)N * K * sizeof(float));
            } else {
                dequant_q8_0(Wb, (int)K, (int)N, Bf.data());
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
            CachedWeight entry{ handle, (int)N, (int)K, std::move(sB), packed, ++g_wcache_tick };
            wcache_evict_to_fit(packed);
            g_wcache[key] = std::move(entry);
            g_wcache_bytes += packed;
            it = g_wcache.find(key);
        } else {
            g_wcache_hits++;
        }
        it->second.last = ++g_wcache_tick;
        const CachedWeight & cw = it->second;

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
    }

    if (getenv("GGML_CDSP_DEBUG"))
        fprintf(stderr, "[ggml-cdsp] MUL_MAT M=%lld N=%lld K=%lld "
                "cache hits=%d misses=%d bytes=%zu\n",
                (long long)M, (long long)N, (long long)K,
                g_wcache_hits, g_wcache_misses, g_wcache_bytes);

    return GGML_STATUS_SUCCESS;
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
    if (op->op != GGML_OP_MUL_MAT) return false;
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];
    if (a == nullptr || b == nullptr) return false;
    if (a->type != GGML_TYPE_F32) return false;
    if (b->type != GGML_TYPE_F32 && b->type != GGML_TYPE_Q8_0) return false;
    if (a->ne[0] != b->ne[0]) return false; // contraction dim must match
    return true;
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
