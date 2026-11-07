//============================================================================
// test_q4k.cpp - validate the QUANTIZED dequant paths in the cDSP backend
// against ggml's own CPU kernels. Covers Q4_K and Q8_0.
//
// WHY: real GGUF models (e.g. Qwen3.5-4B-Q4_K_M) store weights as Q4_K, and
// llama.cpp additionally produces Q8_0 tensors for some attention/KV paths.
// Both formats changed across ggml versions:
//   * Q4_K block = 144 B  (d fp16, dmin fp16, scales[12], qs[128]) - NOT 140.
//   * Q8_0 block =  34 B  (d fp16, qs[32])                        - NOT 36/f32.
// Getting either wrong silently corrupts every offloaded layer while the
// tiny hand-built F32 tests still pass. This test builds REAL quantized
// tensors with ggml_quantize_chunk(), runs the same graph on the CPU backend
// and on the cDSP backend, and compares.
//
// Expected: relative error a few percent (symmetric per-vector int8 is the
// only extra approximation). A large error means a dequant/layout bug.
//
// Build (device):
//   g++ -std=c++20 test_q4k.cpp -I$GGML_INCLUDE -L$GGML_LIB \
//       -lggml -lggml-base -lggml-cpu -o test_q4k
//   GGML_CDSP_BACKEND=/data/local/tmp/ggml-backend/libggml-cdsp.so \
//   LD_LIBRARY_PATH=$GGML_LIB:/usr/local/lib ./test_q4k
//============================================================================
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static bool compute(ggml_type wtype, ggml_backend_t backend,
                    const std::vector<uint8_t> & Wq,
                    const std::vector<float>   & Adata,
                    int M, int K, int N,
                    std::vector<float> & Cout) {
    struct ggml_init_params ip = { 256u * 1024u * 1024u, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // ggml_mul_mat(src0=weight [K,N], src1=activation [K,M]) -> [N,M]
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * mm = ggml_mul_mat(ctx, w, a);

    memcpy(w->data, Wq.data(), Wq.size());
    memcpy(a->data, Adata.data(), Adata.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, mm);

    enum ggml_status st = ggml_backend_graph_compute(backend, g);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "backend compute failed: %d\n", (int)st);
        ggml_free(ctx);
        return false;
    }
    Cout.assign((size_t)M * N, 0.f);
    memcpy(Cout.data(), mm->data, Cout.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

static bool check_type(ggml_type wtype, const char * name,
                       ggml_backend_t cpu, ggml_backend_t cdsp) {
    // 512 is a multiple of both 256 (QK_K) and 32 (QK8_0).
    const int M = 3, K = 512, N = 64;

    std::vector<float> Wf((size_t)N * K), Adata((size_t)M * K);
    srand(20261107);
    for (auto & v : Wf)    v = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
    for (auto & v : Adata) v = ((float)rand() / RAND_MAX - 0.5f) * 2.f;

    const size_t row_bytes = ggml_row_size(wtype, K);
    std::vector<uint8_t> Wq(row_bytes * (size_t)N);
    size_t qbytes = ggml_quantize_chunk(wtype, Wf.data(), Wq.data(),
                                        0, N, K, nullptr);
    printf("%-5s: %zu bytes (expected %zu), row = %zu bytes\n",
           name, qbytes, Wq.size(), row_bytes);
    if (qbytes != Wq.size()) { fprintf(stderr, "FAIL: unexpected %s size\n", name); return false; }

    std::vector<float> Cref, Cdsp;
    if (!compute(wtype, cpu,  Wq, Adata, M, K, N, Cref)) return false;
    if (!compute(wtype, cdsp, Wq, Adata, M, K, N, Cdsp)) return false;

    double maxabs = 0.0, maxerr = 0.0, sumerr = 0.0;
    for (size_t i = 0; i < Cref.size(); i++) {
        double r = Cref[i];
        if (std::fabs(r) > maxabs) maxabs = std::fabs(r);
        double e = std::fabs((double)Cdsp[i] - r);
        if (e > maxerr) maxerr = e;
        sumerr += e;
    }
    double rel = maxabs > 0.0 ? maxerr / maxabs : 0.0;
    printf("  CPU range=%.4f  max abs err=%.6f  mean=%.6f  "
           "rel max err=%.4f (%.2f%%)\n",
           maxabs, maxerr, sumerr / Cref.size(), rel, rel * 100.0);

    if (rel > 0.15) {
        fprintf(stderr, "FAIL: %s dequant path diverges from CPU (rel %.4f)\n",
                name, rel);
        return false;
    }
    printf("  %s_TEST PASS\n", name);
    return true;
}

int main() {
    const char * bp = getenv("GGML_CDSP_BACKEND");
    if (!bp || !bp[0]) bp = "/data/local/tmp/ggml-backend/libggml-cdsp.so";
    ggml_backend_reg_t cdsp_reg = ggml_backend_load(bp);
    if (!cdsp_reg) { fprintf(stderr, "FAIL: load cDSP backend\n"); return 1; }
    ggml_backend_dev_t cdsp_dev = ggml_backend_reg_dev_get(cdsp_reg, 0);
    ggml_backend_t cdsp = ggml_backend_dev_init(cdsp_dev, nullptr);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    if (!cdsp || !cpu) { fprintf(stderr, "FAIL: init backends\n"); return 1; }

    bool ok = true;
    ok &= check_type(GGML_TYPE_Q4_K, "Q4_K", cpu, cdsp);
    ok &= check_type(GGML_TYPE_Q8_0, "Q8_0", cpu, cdsp);

    ggml_backend_free(cdsp);
    ggml_backend_free(cpu);
    ggml_backend_unload(cdsp_reg);
    ggml_quantize_free();

    if (!ok) return 1;
    printf("Q4K_TEST PASS\n");   // kept for deploy_ggml_cdsp.py
    return 0;
}
