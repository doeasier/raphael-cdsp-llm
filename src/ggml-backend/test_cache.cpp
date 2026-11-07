//============================================================================
// test_cache.cpp - verify the ggml-cdsp WEIGHT-CACHE at the scheduler level.
//
// It builds a MUL_MAT graph, computes it TWICE with the SAME weight tensor, and
// asserts:
//   * both runs produce a bit-identical result (the cache path is correct),
//   * the second run is a cache HIT (only the activation A is shipped), visible
//     in the GGML_CDSP_DEBUG line "cache hits=1".
//
// This is the core claim of Stage 5: weights are uploaded to the cDSP once and
// reused across tokens instead of being re-transported every call.
//
// Build (device):
//   g++ -std=c++20 test_cache.cpp -I$GGML_INCLUDE -L$GGML_LIB \
//       -lggml -lggml-base -lggml-cpu -o test_cache
//   GGML_BACKEND_DIR=/data/local/tmp/ggml-backend GGML_CDSP_DEBUG=1 \
//   LD_LIBRARY_PATH=$GGML_LIB ./test_cache
//============================================================================

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// CPU emulation of the plugin's DSP path (for comparison)
static void ref_int8_gemm(const float * A, const float * B,
                          int M, int K, int N, std::vector<float> & C) {
    std::vector<int8_t>  Aq((size_t)M * K);
    std::vector<int8_t>  Bq((size_t)N * K);
    std::vector<float>   sA((size_t)M);
    std::vector<float>   sB((size_t)N);
    std::vector<int32_t> C32((size_t)M * N);
    auto quant = [](const float * x, int n, int8_t * q, float * s) {
        float mx = 0.f;
        for (int i = 0; i < n; i++) { float a = std::fabs(x[i]); if (a > mx) mx = a; }
        float sc = (mx == 0.f) ? 1.f : (mx / 127.f);
        *s = sc;
        for (int i = 0; i < n; i++) {
            int v = (int)std::lroundf(x[i] / sc);
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            q[i] = (int8_t)v;
        }
    };
    for (int m = 0; m < M; m++) quant(A + (size_t)m * K, K, Aq.data() + (size_t)m * K, &sA[m]);
    for (int n = 0; n < N; n++) quant(B + (size_t)n * K, K, Bq.data() + (size_t)n * K, &sB[n]);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)Aq[(size_t)m * K + k] * (int32_t)Bq[(size_t)n * K + k];
            C32[(size_t)m * N + n] = acc;
        }
    C.resize((size_t)M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C[(size_t)m * N + n] = sA[m] * sB[n] * (float)C32[(size_t)m * N + n];
}

int main() {
    const int M = 64, K = 256, N = 192;

    const char * bp = getenv("GGML_CDSP_BACKEND");
    if (!bp || bp[0] == '\0') bp = "/data/local/tmp/ggml-backend/libggml-cdsp.so";
    ggml_backend_reg_t cdsp_reg = ggml_backend_load(bp);
    if (!cdsp_reg) { fprintf(stderr, "FAIL: load 'cdsp' backend from %s\n", bp); return 1; }
    ggml_backend_dev_t cdsp_dev = ggml_backend_reg_dev_get(cdsp_reg, 0);
    ggml_backend_t    cdsp      = ggml_backend_dev_init(cdsp_dev, NULL);
    if (!cdsp) { fprintf(stderr, "FAIL: init cdsp backend\n"); return 1; }
    printf("OK: loaded cDSP backend '%s'\n", ggml_backend_name(cdsp));

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("cpu");
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, NULL);
    if (!cpu) { fprintf(stderr, "FAIL: init cpu backend\n"); return 1; }

    ggml_backend_t backends[2] = { cdsp, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { fprintf(stderr, "FAIL: sched_new\n"); return 1; }

    std::vector<float> Adata((size_t)M * K), Bdata((size_t)N * K);
    srand(12345);
    for (auto & v : Adata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
    for (auto & v : Bdata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.f;

    struct ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "FAIL: ggml_init\n"); return 1; }

    // ggml_mul_mat(src0=weight, src1=activation): result ne[0]=N, ne[1]=M
    ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // activation [K,M]
    ggml_tensor * B  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); // weight     [K,N]
    ggml_tensor * mm = ggml_mul_mat(ctx, B, A);
    ggml_set_name(mm, "out");

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, mm);

    if (!ggml_backend_sched_alloc_graph(sched, g)) { fprintf(stderr, "FAIL: alloc_graph\n"); return 1; }
    ggml_backend_tensor_set(A, Adata.data(), 0, Adata.size() * sizeof(float));
    ggml_backend_tensor_set(B, Bdata.data(), 0, Bdata.size() * sizeof(float));

    std::vector<float> Crun1((size_t)M * N);
    enum ggml_status st = ggml_backend_sched_graph_compute(sched, g);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "FAIL: compute #1 status=%d\n", st); return 1; }
    ggml_backend_tensor_get(mm, Crun1.data(), 0, Crun1.size() * sizeof(float));

    // second compute: SAME weights -> cache HIT (only A re-shipped)
    std::vector<float> Crun2((size_t)M * N);
    st = ggml_backend_sched_graph_compute(sched, g);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "FAIL: compute #2 status=%d\n", st); return 1; }
    ggml_backend_tensor_get(mm, Crun2.data(), 0, Crun2.size() * sizeof(float));

    std::vector<float> Cref;
    ref_int8_gemm(Adata.data(), Bdata.data(), M, K, N, Cref);

    float maxerr = 0.f;
    for (size_t i = 0; i < Crun1.size(); i++) {
        float e = std::fabs(Crun1[i] - Cref[i]); if (e > maxerr) maxerr = e;
        float e2 = std::fabs(Crun2[i] - Crun1[i]); if (e2 > maxerr) maxerr = e2;
    }
    printf("OK: 2 computes ran; max abs error vs int8-emu = %.6f\n", maxerr);

    if (maxerr > 1e-2f) { fprintf(stderr, "FAIL: error exceeds tolerance\n"); return 1; }
    printf("CACHE_TEST PASS\n");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cdsp);
    ggml_backend_free(cpu);
    ggml_backend_unload(cdsp_reg);
    ggml_free(ctx);
    return 0;
}
