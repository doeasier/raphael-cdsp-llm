//============================================================================
// bench_cached.c - AP-side microbenchmark for the cDSP weight-cache GEMM.
//
// Separates two costs that a single "tokens/s" number cannot:
//   * FastRPC round-trip latency (tiny shape, e.g. 1x32x4)
//   * DSP kernel throughput       (decode / prompt / square shapes)
//
// It uploads the weight ONCE (gemm_upload), then calls gemm_compute_cached
// repeatedly with the same activation. Reports average ms/call and effective
// GMAC/s = 2*M*N*K / time. The first few calls are warmups (cold cache,
// unsigned-PD / HVX lock, DSP frequency ramp) and are excluded.
//
// Build on device (aarch64 Linux):
//   gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
//       bench_cached.c gen/cdsp_gemm_stub.c -o bench_cached \
//       -L/usr/local/lib -ladsprpc -lpthread -ldl
// Usage: ./bench_cached M N K [iters] [warmups]
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "gemm.h"
#include <remote.h>

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void enable_unsigned_pd(void) {
    struct remote_rpc_control_unsigned_module data;
    memset(&data, 0, sizeof(data));
    data.enable = 1;
    data.domain = CDSP_DOMAIN_ID;
    (void)remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                 (void *)&data, (uint32_t)sizeof(data));
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 2560;
    int iters = (argc > 4) ? atoi(argv[4]) : 50;
    int warm  = (argc > 5) ? atoi(argv[5]) : 5;
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0) return -1;

    const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    int8_t   * A = (int8_t *)  malloc(szA);
    int8_t   * B = (int8_t *)  malloc(szB);
    int32_t  * C = (int32_t *) malloc(szC * sizeof(int32_t));
    if (!A || !B || !C) { fprintf(stderr, "malloc fail\n"); return -1; }

    srand(12345);
    for (size_t i = 0; i < szA; i++) A[i] = (int8_t)((rand() & 0x1F) - 8);
    for (size_t i = 0; i < szB; i++) B[i] = (int8_t)((rand() & 0x1F) - 8);

    enable_unsigned_pd();
    remote_handle64 h = 0;
    int rc = cdsp_gemm_open(cdsp_gemm_URI CDSP_DOMAIN, &h);
    if (rc != 0) { fprintf(stderr, "open failed rc=%d\n", rc); return -2; }
    cdsp_gemm_rpc_init(h);

    int32_t dims_up[3] = { 0, N, K };
    int32_t wh = -1;
    double t0 = now_ms();
    rc = cdsp_gemm_gemm_upload(h, B, (int)szB, dims_up, 3, &wh);
    double upload_ms = now_ms() - t0;
    if (rc != 0) { fprintf(stderr, "upload failed rc=%d\n", rc); return -3; }

    int32_t dims_c[3] = { M, N, K };
    for (int i = 0; i < warm; i++) {
        rc = cdsp_gemm_gemm_compute_cached(h, A, (int)szA, dims_c, 3, wh,
                                           C, (int)szC);
        if (rc != 0) { fprintf(stderr, "warmup compute failed rc=%d\n", rc); return -4; }
    }

    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        rc = cdsp_gemm_gemm_compute_cached(h, A, (int)szA, dims_c, 3, wh,
                                           C, (int)szC);
        if (rc != 0) { fprintf(stderr, "compute failed rc=%d\n", rc); return -4; }
    }
    double total_ms = now_ms() - t0;
    double per_ms   = total_ms / iters;
    double macs     = 2.0 * (double)M * (double)N * (double)K;
    double gmac_s   = macs / (per_ms / 1000.0) / 1e9;

    // spot-check correctness of the LAST result (one element, scalar ref)
    int32_t ref = 0;
    for (int kk = 0; kk < K; kk++)
        ref += (int32_t)A[kk] * (int32_t)B[(size_t)kk * N + (N - 1)];
    int ok = (C[N - 1] == ref);

    printf("BENCH M=%d N=%d K=%d iters=%d | upload=%.2f ms | "
           "compute=%.4f ms/call | %.2f GMAC/s | verify[0,%d]=%s\n",
           M, N, K, iters, upload_ms, per_ms, gmac_s, N - 1,
           ok ? "PASS" : "FAIL");
    if (!ok) fprintf(stderr, "  got=%d ref=%d\n", C[N - 1], ref);

    cdsp_gemm_gemm_release(h, wh);
    cdsp_gemm_close(h);
    free(A); free(B); free(C);
    return ok ? 0 : -5;
}
