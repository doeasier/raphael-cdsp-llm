//============================================================================
// bench_regbuf.c - does a persistent FastRPC buffer mapping cut RPC latency?
//
// Stage 8 investigation. The Stage 8 per-stage profile on SmolLM2-135M showed
// `gemm_compute_cached` costing ~1.11 ms/call while a tiny ping shape costs
// only ~0.25 ms. One suspect is that every call maps/unmaps (and flushes the
// caches of) the activation A and result C buffers, which are freshly
// malloc'd std::vectors in the ggml backend.
//
// This microbenchmark uploads one weight and then times the SAME compute with
// three different A/C buffer allocation strategies:
//
//   mode 0 HEAP    : plain malloc                         (current backend)
//   mode 1 RPCMEM  : rpcmem_alloc_def (DMA-BUF)          (zero-copy candidate)
//   mode 2 MAPPED  : RPCMEM + fastrpc_mmap(STATIC)       (persistent mapping)
//
// Build on device (aarch64 Linux):
//   gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
//       bench_regbuf.c gen/cdsp_gemm_stub.c -o bench_regbuf \
//       -L/usr/local/lib -ladsprpc -lpthread -ldl
// Usage: ./bench_regbuf [M N K iters]   (default 1 576 1536 100)
//
// Results are printed as REGBUF mode=<n> ... so deploy scripts can grep them.
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "gemm.h"
#include <remote.h>
#include <rpcmem.h>

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif
#ifndef FASTRPC_MAP_STATIC
#define FASTRPC_MAP_STATIC 0
#endif

#define PAGE 4096
static size_t page_align(size_t n) { return (n + PAGE - 1) & ~(size_t)(PAGE - 1); }

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
    int M     = (argc > 1) ? atoi(argv[1]) : 1;
    int N     = (argc > 2) ? atoi(argv[2]) : 576;
    int K     = (argc > 3) ? atoi(argv[3]) : 1536;
    int iters = (argc > 4) ? atoi(argv[4]) : 100;
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0) return -1;

    const size_t szA = (size_t)M * K;
    const size_t szB = (size_t)K * N;
    const size_t szC = (size_t)M * N;
    const size_t lenA = page_align(szA), lenC = page_align(szC * sizeof(int32_t));

    int8_t  * Aref = (int8_t *)  malloc(szA);
    int8_t  * B    = (int8_t *)  malloc(szB);
    int32_t * Cref = (int32_t *) malloc(szC * sizeof(int32_t));
    if (!Aref || !B || !Cref) { fprintf(stderr, "malloc fail\n"); return -1; }

    srand(12345);
    for (size_t i = 0; i < szA; i++) Aref[i] = (int8_t)((rand() & 0x1F) - 8);
    for (size_t i = 0; i < szB; i++) B[i]    = (int8_t)((rand() & 0x1F) - 8);

    enable_unsigned_pd();
    remote_handle64 h = 0;
    int rc = cdsp_gemm_open(cdsp_gemm_URI CDSP_DOMAIN, &h);
    if (rc != 0) { fprintf(stderr, "open failed rc=%d\n", rc); return -2; }
    cdsp_gemm_rpc_init(h);

    int32_t dims_up[3] = { 0, N, K };
    int32_t wh = -1;
    rc = cdsp_gemm_gemm_upload(h, B, (int)szB, dims_up, 3, &wh);
    if (rc != 0) { fprintf(stderr, "upload failed rc=%d\n", rc); return -3; }

    // scalar reference for one output element, valid for every mode
    int32_t ref = 0;
    for (int kk = 0; kk < K; kk++)
        ref += (int32_t)Aref[kk] * (int32_t)B[(size_t)kk * N + (N - 1)];

    const int32_t dims_c[3] = { M, N, K };
    static const char * names[] = { "HEAP", "RPCMEM", "MAPPED" };
    double baseline = 0.0;
    int any_fail = 0;

    for (int mode = 0; mode < 3; mode++) {
        int8_t  * A = NULL;
        int32_t * C = NULL;
        int fdA = -1, fdC = -1, mappedA = 0, mappedC = 0;

        if (mode == 0) {
            A = (int8_t *)  malloc(szA);
            C = (int32_t *) malloc(szC * sizeof(int32_t));
        } else {
            A = (int8_t *)  rpcmem_alloc_def((int)lenA);
            C = (int32_t *) rpcmem_alloc_def((int)lenC);
            if (A) fdA = rpcmem_to_fd(A);
            if (C) fdC = rpcmem_to_fd(C);
        }
        if (!A || !C) {
            printf("REGBUF mode=%d %-6s | ALLOC_FAIL\n", mode, names[mode]);
            any_fail = 1;
            goto cleanup;
        }
        memcpy(A, Aref, szA);
        memset(C, 0, szC * sizeof(int32_t));

        if (mode == 2) {
            int ra = fastrpc_mmap(CDSP_DOMAIN_ID, fdA, A, 0, lenA,
                                  (enum fastrpc_map_flags)FASTRPC_MAP_STATIC);
            int rc2 = fastrpc_mmap(CDSP_DOMAIN_ID, fdC, C, 0, lenC,
                                   (enum fastrpc_map_flags)FASTRPC_MAP_STATIC);
            mappedA = (ra == 0);
            mappedC = (rc2 == 0);
            if (!mappedA || !mappedC) {
                printf("REGBUF mode=2 MAPPED  | MMAP_UNSUPPORTED "
                       "rcA=%d rcC=%d (fdA=%d fdC=%d)\n", ra, rc2, fdA, fdC);
                goto cleanup;
            }
        }

        for (int i = 0; i < 5; i++) {
            rc = cdsp_gemm_gemm_compute_cached(h, A, (int)szA, dims_c, 3, wh,
                                               C, (int)szC);
            if (rc != 0) { printf("REGBUF mode=%d | warmup rc=%d\n", mode, rc);
                           any_fail = 1; goto cleanup; }
        }
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            rc = cdsp_gemm_gemm_compute_cached(h, A, (int)szA, dims_c, 3, wh,
                                               C, (int)szC);
            if (rc != 0) { printf("REGBUF mode=%d | compute rc=%d\n", mode, rc);
                           any_fail = 1; goto cleanup; }
        }
        double per_ms = (now_ms() - t0) / iters;
        if (mode == 0) baseline = per_ms;

        int ok = (C[N - 1] == ref);
        if (!ok) any_fail = 1;
        printf("REGBUF mode=%d %-6s | %8.4f ms/call | %+6.1f%% vs HEAP | "
               "verify=%s\n",
               mode, names[mode], per_ms,
               baseline > 0 ? (per_ms / baseline - 1.0) * 100.0 : 0.0,
               ok ? "PASS" : "FAIL");
        if (!ok) printf("  got=%d ref=%d\n", C[N - 1], ref);

cleanup:
        if (mappedA) fastrpc_munmap(CDSP_DOMAIN_ID, fdA, A, lenA);
        if (mappedC) fastrpc_munmap(CDSP_DOMAIN_ID, fdC, C, lenC);
        if (mode == 0) { free(A); free(C); }
        else {
            if (A) rpcmem_free(A);
            if (C) rpcmem_free(C);
        }
    }

    cdsp_gemm_gemm_release(h, wh);
    cdsp_gemm_close(h);
    free(Aref); free(B); free(Cref);
    return any_fail ? -5 : 0;
}
