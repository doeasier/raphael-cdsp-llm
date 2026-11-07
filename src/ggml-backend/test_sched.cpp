//============================================================================
// test_sched.cpp - multi-backend scheduler integration test for the ggml-cdsp
// backend.
//
// This exercises the EXACT integration path that llama.cpp uses: a
// ggml_backend_sched with two backends (cDSP first, CPU second) and
// op_offload=true. The scheduler must:
//   * assign GGML_OP_MUL_MAT to the cDSP backend (low index + offload),
//   * assign every other op (here: relu) to the CPU backend,
//   * allocate tensors across both backends' buffers and copy between them.
//
// The MUL_MAT output is computed by the cDSP via FastRPC int8 GEMM; the relu
// runs on the CPU. We compare the final result against a CPU int8-emulation of
// the MUL_MAT path (relu applied after). A successful run prints
// "SCHED_TEST PASS" and (with GGML_CDSP_DEBUG=1) a "[ggml-cdsp] offloaded
// MUL_MAT ..." line proving the offload actually happened.
//
// Build (device):
//   g++ -std=c++20 test_sched.cpp -I$GGML_INCLUDE -L$GGML_LIB \
//       -lggml -lggml-base -lggml-cpu -o test_sched
//   GGML_BACKEND_DIR=/data/local/tmp/ggml-backend GGML_CDSP_DEBUG=1 \
//   LD_LIBRARY_PATH=$GGML_LIB ./test_sched
//============================================================================

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static void quant_vec(const float * x, int n, int8_t * q, float * scale) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = (x[i] < 0.f) ? -x[i] : x[i];
        if (a > mx) mx = a;
    }
    float s = (mx == 0.f) ? 1.f : (mx / 127.f);
    *scale = s;
    for (int i = 0; i < n; i++) {
        int v = (int)lroundf(x[i] / s);
        if (v >  127) v =  127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

// CPU emulation of the plugin's DSP path (for comparison)
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

    // --- cDSP backend (loaded from the explicit .so path) ---
    const char * bp = getenv("GGML_CDSP_BACKEND");
    if (!bp || bp[0] == '\0')
        bp = "/data/local/tmp/ggml-backend/libggml-cdsp.so";
    ggml_backend_reg_t cdsp_reg = ggml_backend_load(bp);
    if (!cdsp_reg) {
        fprintf(stderr, "FAIL: could not load 'cdsp' backend from %s\n", bp);
        return 1;
    }
    ggml_backend_dev_t cdsp_dev = ggml_backend_reg_dev_get(cdsp_reg, 0);
    ggml_backend_t    cdsp      = ggml_backend_dev_init(cdsp_dev, NULL);
    if (!cdsp) { fprintf(stderr, "FAIL: init cdsp backend\n"); return 1; }
    printf("OK: loaded cDSP backend '%s'\n", ggml_backend_name(cdsp));

    // --- CPU backend (built-in) ---
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("cpu");
    if (!cpu_dev) { fprintf(stderr, "FAIL: no cpu backend device\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, NULL);
    if (!cpu) { fprintf(stderr, "FAIL: init cpu backend\n"); return 1; }
    printf("OK: loaded CPU backend '%s'\n", ggml_backend_name(cpu));

    // Low index = priority; offload sends MUL_MAT to cDSP, rest to CPU.
    ggml_backend_t backends[2] = { cdsp, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!sched) { fprintf(stderr, "FAIL: ggml_backend_sched_new\n"); return 1; }

    std::vector<float> Adata((size_t)M * K), Bdata((size_t)N * K);
    srand(12345);
    for (auto & v : Adata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (auto & v : Bdata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    struct ggml_init_params ip = { /*mem_size*/ 256u * 1024u * 1024u,
                                    /*mem_buffer*/ nullptr,
                                    /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "FAIL: ggml_init\n"); return 1; }

    // ggml_mul_mat(src0=weight, src1=activation): ne[0]=K contraction
    ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // activation [K,M]
    ggml_tensor * B  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); // weight     [K,N]
    ggml_tensor * mm = ggml_mul_mat(ctx, B, A);   // -> cDSP
    ggml_tensor * out= ggml_relu(ctx, mm);          // -> CPU
    ggml_set_name(out, "out");

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);

    if (!ggml_backend_sched_alloc_graph(sched, g)) {
        fprintf(stderr, "FAIL: ggml_backend_sched_alloc_graph\n");
        return 1;
    }
    ggml_backend_tensor_set(A, Adata.data(), 0, Adata.size() * sizeof(float));
    ggml_backend_tensor_set(B, Bdata.data(), 0, Bdata.size() * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(sched, g);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: sched graph_compute status=%d\n", st);
        return 1;
    }

    std::vector<float> Cgot((size_t)M * N);
    ggml_backend_tensor_get(out, Cgot.data(), 0, Cgot.size() * sizeof(float));

    std::vector<float> Cref;
    ref_int8_gemm(Adata.data(), Bdata.data(), M, K, N, Cref);

    float maxerr = 0.0f;
    for (size_t i = 0; i < Cgot.size(); i++) {
        float ref = (Cref[i] > 0.f) ? Cref[i] : 0.0f;  // relu
        float e = Cgot[i] - ref; if (e < 0.f) e = -e;
        if (e > maxerr) maxerr = e;
    }
    printf("OK: scheduler ran MUL_MAT(cDSP)+relu(CPU); max abs error vs int8-emu = %.6f\n",
           maxerr);

    if (maxerr > 1e-2f) {
        fprintf(stderr, "FAIL: error exceeds tolerance\n");
        return 1;
    }
    printf("SCHED_TEST PASS\n");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cdsp);
    ggml_backend_free(cpu);
    ggml_backend_unload(cdsp_reg);
    ggml_free(ctx);
    return 0;
}
