//============================================================================
// test_cdsp.cpp - end-to-end correctness check for the ggml-cdsp backend.
//
// It builds a single F32 MUL_MAT  (C = A @ B,  C(M,N) = A(M,K) @ B(K,N))
// and asks the cDSP backend to compute it. The result is compared against a
// CPU emulation of the *exact same* int8 quantization+GEMM+rescale path the
// plugin uses (per-row scale for A, per-column scale for B, int8 GEMM on the
// DSP, then C = sA*sB*C32). This isolates DSP compute correctness from the
// (expected, lossy) int8 quantization, so the tolerance can be tiny.
//
// Build (device):
//   g++ -std=c++20 test_cdsp.cpp -I$GGML_INCLUDE -L$GGML_LIB -lggml \
//       -o test_cdsp
//   GGML_BACKEND_DIR=/data/local/tmp/ggml-backend LD_LIBRARY_PATH=$GGML_LIB \
//       ./test_cdsp
// Exits 0 on PASS.
//============================================================================

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// ---- mirror of the plugin's int8 quantization (per-vector, scale=maxabs/127)
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

// ---- CPU emulation of the plugin's DSP path (for comparison)
static void ref_int8_gemm(const float * A, const float * B,
                          int M, int K, int N, std::vector<float> & C) {
    std::vector<int8_t>  Aq((size_t)M * K);
    std::vector<int8_t>  Bq((size_t)N * K);
    std::vector<float>   sA((size_t)M);
    std::vector<float>   sB((size_t)N);
    std::vector<int32_t> C32((size_t)M * N);

    for (int m = 0; m < M; m++)
        quant_vec(A + (size_t)m * K, K, Aq.data() + (size_t)m * K, &sA[m]);
    for (int n = 0; n < N; n++)
        quant_vec(B + (size_t)n * K, K, Bq.data() + (size_t)n * K, &sB[n]);

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
    const int M = 64, K = 128, N = 96;

    // ggml_backend_load() dlopen's the .so directly (it treats its argument as
    // a library *path*, not a backend name), registers the "cdsp" backend and
    // returns the reg handle. We then grab device 0 and init a backend from it.
    // The path can be overridden with GGML_CDSP_BACKEND; default to the location
    // deploy_ggml_cdsp.py installs the plugin into.
    const char * backend_path = getenv("GGML_CDSP_BACKEND");
    if (!backend_path || backend_path[0] == '\0')
        backend_path = "/data/local/tmp/ggml-backend/libggml-cdsp.so";
    ggml_backend_reg_t reg = ggml_backend_load(backend_path);
    if (!reg) {
        fprintf(stderr, "FAIL: could not load 'cdsp' backend "
                        "(is libggml-cdsp.so in GGML_BACKEND_DIR?)\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) {
        fprintf(stderr, "FAIL: cdsp backend reports no device\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        fprintf(stderr, "FAIL: could not init cdsp backend\n");
        return 1;
    }
    printf("OK: loaded backend '%s'\n", ggml_backend_name(backend));

    std::vector<float> Adata((size_t)M * K), Bdata((size_t)N * K);
    srand(12345);
    for (auto & v : Adata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (auto & v : Bdata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    struct ggml_init_params ip = { /*mem_size*/ 256u * 1024u * 1024u,
                                   /*mem_buffer*/ nullptr,
                                   /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "FAIL: ggml_init\n"); return 1; }

    // ggml_mul_mat(src0=weight, src1=activation): result ne[0]=N, ne[1]=M
    ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // activation [K,M]
    ggml_tensor * B  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); // weight     [K,N]
    ggml_tensor * mm = ggml_mul_mat(ctx, B, A);
    ggml_set_name(mm, "mulmat");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "FAIL: alloc tensors on cDSP backend\n"); return 1; }

    ggml_backend_tensor_set(A, Adata.data(), 0, Adata.size() * sizeof(float));
    ggml_backend_tensor_set(B, Bdata.data(), 0, Bdata.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, mm);

    enum ggml_status st = ggml_backend_graph_compute(backend, g);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: graph_compute status=%d\n", st);
        return 1;
    }

    std::vector<float> Cgot((size_t)M * N);
    ggml_backend_tensor_get(mm, Cgot.data(), 0, Cgot.size() * sizeof(float));

    std::vector<float> Cref;
    ref_int8_gemm(Adata.data(), Bdata.data(), M, K, N, Cref);

    float maxerr = 0.0f;
    for (size_t i = 0; i < Cgot.size(); i++) {
        float e = Cgot[i] - Cref[i]; if (e < 0.f) e = -e;
        if (e > maxerr) maxerr = e;
    }
    printf("OK: MUL_MAT computed on cDSP; max abs error vs int8-emulation = %.6f\n",
           maxerr);

    if (maxerr > 1e-2f) {
        fprintf(stderr, "FAIL: error exceeds tolerance (DSP compute mismatch?)\n");
        return 1;
    }
    printf("PASS\n");

    ggml_backend_free(backend);
    ggml_backend_unload(reg);
    ggml_free(ctx);
    return 0;
}
