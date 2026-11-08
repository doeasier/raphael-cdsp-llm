//============================================================================
// host_test_batch.c - AP (Linux on K20 Pro) correctness + latency test for the
// Stage 7 BATCHED FastRPC encoding of gemm_compute_cached.
//
// It uploads two DIFFERENT weights (different shapes, different data), runs
// them (a) as two scalar compute_cached calls and (b) as ONE batched call:
//   dims = { -2, M0,N0,K0,h0, M1,N1,K1,h1 }
//   A    = concat(A0, A1),  C = concat(C0, C1)
// and verifies every element against an int32 scalar GEMM reference. It also
// reports the measured round-trip latency of "2 RPCs" vs "1 RPC".
//
// Compile on-device (aarch64) against the QAIC-generated gemm.h + libfastrpc:
//   gcc -O2 -I. -Igen -I/usr/local/include/fastrpc host_test_batch.c \
//       gen/cdsp_gemm_stub.c -o host_test_batch \
//       -L/usr/local/lib -ladsprpc -lpthread -ldl
// Usage: ./host_test_batch [iterations]
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "gemm.h"               // QAIC-generated header
#include <remote.h>             // remote_session_control, unsigned-PD control

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

#define NG 2                    /* number of batched groups */

static void enable_unsigned_pd(void) {
    struct remote_rpc_control_unsigned_module data;
    memset(&data, 0, sizeof(data));
    data.enable = 1;
    data.domain = CDSP_DOMAIN_ID;
    int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                    (void*)&data, (uint32_t)sizeof(data));
    if (rc != 0)
        fprintf(stderr, "host: unsigned-PD control rc=%d (non-fatal)\n", rc);
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int32_t ref_gemm(const int8_t * A, const int8_t * B,
                        int m, int n, int N, int K) {
    int32_t s = 0;
    for (int k = 0; k < K; k++)
        s += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)k * N + n];
    return s;
}

static int verify(const char * tag, const int32_t * got,
                  const int32_t * ref, size_t n) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            if (bad < 8)
                fprintf(stderr, "host: %s mismatch[%zu] dsp=%d ref=%d\n",
                        tag, i, got[i], ref[i]);
            bad++;
        }
    }
    return bad;
}

int main(int argc, char ** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 30;
    if (iters < 1) iters = 1;

    /* Deliberately DIFFERENT shapes so the concat offsets are exercised. */
    const int M[NG] = { 1, 1 };
    const int N[NG] = { 1024, 32 };
    const int K[NG] = { 2560, 2560 };

    int8_t  * A[NG], * B[NG];
    int32_t * R[NG], * Cs[NG];
    size_t szA[NG], szB[NG], szC[NG];
    int32_t wh[NG] = { -1, -1 };

    for (int g = 0; g < NG; g++) {
        szA[g] = (size_t)M[g] * K[g];
        szB[g] = (size_t)K[g] * N[g];
        szC[g] = (size_t)M[g] * N[g];
        A[g] = malloc(szA[g]);  B[g] = malloc(szB[g]);
        R[g] = malloc(szC[g] * 4); Cs[g] = malloc(szC[g] * 4);
        if (!A[g] || !B[g] || !R[g] || !Cs[g]) {
            fprintf(stderr, "host: malloc fail\n"); return -1;
        }
        srand(1000 + g * 7919);
        for (size_t i = 0; i < szA[g]; i++) A[g][i] = (int8_t)(rand() & 0x1F) - 8;
        for (size_t i = 0; i < szB[g]; i++) B[g][i] = (int8_t)(rand() & 0x1F) - 8;
        for (int m = 0; m < M[g]; m++)
            for (int n = 0; n < N[g]; n++)
                R[g][(size_t)m * N[g] + n] =
                    ref_gemm(A[g], B[g], m, n, N[g], K[g]);
    }

    size_t totalA = 0, totalC = 0;
    for (int g = 0; g < NG; g++) { totalA += szA[g]; totalC += szC[g]; }
    int8_t  * Abuf = malloc(totalA);
    int32_t * Cbuf = malloc(totalC * 4);
    if (!Abuf || !Cbuf) { fprintf(stderr, "host: malloc fail\n"); return -1; }

    int32_t dims[1 + 4 * NG];
    dims[0] = -NG;
    size_t ao = 0;
    for (int g = 0; g < NG; g++) {
        memcpy(Abuf + ao, A[g], szA[g]);
        ao += szA[g];
        dims[1 + 4 * g + 0] = M[g];
        dims[1 + 4 * g + 1] = N[g];
        dims[1 + 4 * g + 2] = K[g];
        dims[1 + 4 * g + 3] = 0;   /* filled in after upload */
    }

    enable_unsigned_pd();
    remote_handle64 h = 0;
    int rc = cdsp_gemm_open(cdsp_gemm_URI CDSP_DOMAIN, &h);
    if (rc != 0) { fprintf(stderr, "host: open failed rc=%d\n", rc); return -2; }
    cdsp_gemm_rpc_init(h);

    /* ---- upload both weights ---- */
    for (int g = 0; g < NG; g++) {
        int32_t du[3] = { 0, N[g], K[g] };
        rc = cdsp_gemm_gemm_upload(h, B[g], (int)szB[g], du, 3, &wh[g]);
        if (rc != 0) {
            fprintf(stderr, "host: upload[%d] failed rc=%d\n", g, rc);
            cdsp_gemm_close(h); return -3;
        }
        dims[1 + 4 * g + 3] = wh[g];
    }
    printf("host: uploaded %d weights -> handles %d / %d\n", NG, wh[0], wh[1]);

    int bad = 0;

    /* ---- (a) two scalar calls: correctness + latency ---- */
    for (int g = 0; g < NG; g++) {
        int32_t d[3] = { M[g], N[g], K[g] };
        rc = cdsp_gemm_gemm_compute_cached(h, A[g], (int)szA[g], d, 3,
                                           wh[g], Cs[g], (int)szC[g]);
        if (rc != 0) {
            fprintf(stderr, "host: scalar compute[%d] failed rc=%d\n", g, rc);
            cdsp_gemm_close(h); return -4;
        }
        bad += verify("scalar", Cs[g], R[g], szC[g]);
    }
    for (int i = 0; i < 5; i++)
        for (int g = 0; g < NG; g++) {
            int32_t d[3] = { M[g], N[g], K[g] };
            cdsp_gemm_gemm_compute_cached(h, A[g], (int)szA[g], d, 3,
                                          wh[g], Cs[g], (int)szC[g]);
        }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        for (int g = 0; g < NG; g++) {
            int32_t d[3] = { M[g], N[g], K[g] };
            rc = cdsp_gemm_gemm_compute_cached(h, A[g], (int)szA[g], d, 3,
                                               wh[g], Cs[g], (int)szC[g]);
            if (rc != 0) { fprintf(stderr, "host: scalar rc=%d\n", rc); return -4; }
        }
    double scalar_ms = (now_ms() - t0) / iters;

    /* ---- (b) ONE batched call: correctness + latency ---- */
    memset(Cbuf, 0, totalC * 4);
    rc = cdsp_gemm_gemm_compute_cached(h, Abuf, (int)totalA,
                                        dims, 1 + 4 * NG, 0,
                                        Cbuf, (int)totalC);
    if (rc != 0) {
        fprintf(stderr, "host: BATCHED compute failed rc=%d\n", rc);
        cdsp_gemm_close(h); return -5;
    }
    size_t co = 0;
    for (int g = 0; g < NG; g++) {
        bad += verify("batch", Cbuf + co, R[g], szC[g]);
        co += szC[g];
    }
    for (int i = 0; i < 5; i++)
        cdsp_gemm_gemm_compute_cached(h, Abuf, (int)totalA, dims,
                                      1 + 4 * NG, 0, Cbuf, (int)totalC);
    t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        rc = cdsp_gemm_gemm_compute_cached(h, Abuf, (int)totalA, dims,
                                           1 + 4 * NG, 0, Cbuf, (int)totalC);
        if (rc != 0) { fprintf(stderr, "host: batch rc=%d\n", rc); return -5; }
    }
    double batch_ms = (now_ms() - t0) / iters;

    printf("host: scalar  2 RPCs = %.3f ms/pair\n", scalar_ms);
    printf("host: batched 1 RPC  = %.3f ms/pair  (%.1f%% less)\n",
           batch_ms, 100.0 * (scalar_ms - batch_ms) / scalar_ms);
    if (bad == 0)
        printf("host: BATCH_TEST PASS - %zu elements verified correct.\n", totalC);
    else
        fprintf(stderr, "host: BATCH_TEST FAIL - %d mismatches.\n", bad);

    for (int g = 0; g < NG; g++) cdsp_gemm_gemm_release(h, wh[g]);
    cdsp_gemm_close(h);
    for (int g = 0; g < NG; g++) {
        free(A[g]); free(B[g]); free(R[g]); free(Cs[g]);
    }
    free(Abuf); free(Cbuf);
    return (bad == 0) ? 0 : -6;
}
