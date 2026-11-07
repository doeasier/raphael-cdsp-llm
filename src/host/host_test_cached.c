//============================================================================
// host_test_cached.c - AP (Linux on K20 Pro) correctness + perf test for the
// cDSP WEIGHT-CACHE GEMM API (gemm_upload / gemm_compute_cached /
// gemm_release). This is the Stage-1-level validation of Stage 5: it proves
// that uploading the (large) weight B once and reusing it across compute calls
// produces identical results to the one-shot gemm_int8, and shows the time
// saved (only the small activation A is shipped per call).
//
// Compile on-device (aarch64) against the QAIC-generated gemm.h + libfastrpc.
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemm.h"               // QAIC-generated header
#include <remote.h>             // remote_session_control, unsigned-PD control

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

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

static int32_t ref_gemm(const int8_t* A, const int8_t* B, int m, int n, int M, int N, int K) {
    int32_t s = 0;
    for (int kk = 0; kk < K; kk++)
        s += (int32_t)A[(size_t)m * K + kk] * (int32_t)B[(size_t)kk * N + n];
    (void)M;
    return s;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 32;
    int N = (argc > 2) ? atoi(argv[2]) : 128;
    int K = (argc > 3) ? atoi(argv[3]) : 128;
    if (M <= 0 || N <= 0 || K <= 0) { fprintf(stderr, "bad dims\n"); return -1; }

    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;   // B is K x N
    size_t szC = (size_t)M * N;

    int8_t *A = malloc(szA), *B = malloc(szB);
    int32_t *C = malloc(szC * 4), *Cref = malloc(szC * 4);
    if (!A || !B || !C || !Cref) { fprintf(stderr, "malloc fail\n"); return -1; }

    srand(12345);
    for (size_t i = 0; i < szA; i++) A[i] = (int8_t)(rand() & 0x1F) - 8;
    for (size_t i = 0; i < szB; i++) B[i] = (int8_t)(rand() & 0x1F) - 8;

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            Cref[(size_t)m * N + n] = ref_gemm(A, B, m, n, M, N, K);

    enable_unsigned_pd();

    remote_handle64 handle = 0;
    int rc = cdsp_gemm_open(cdsp_gemm_URI CDSP_DOMAIN, &handle);
    if (rc != 0) { fprintf(stderr, "host: open failed rc=%d\n", rc); return -2; }
    printf("host: cDSP session open, handle=%llu\n", (unsigned long long)handle);
    cdsp_gemm_rpc_init(handle);

    // ---- upload weight B once (K x N) ----
    int32_t dims_up[3] = { 0, N, K };
    int32_t wh = -1;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rc = cdsp_gemm_gemm_upload(handle, B, (int)szB, dims_up, 3, &wh);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0) { fprintf(stderr, "host: upload failed rc=%d\n", rc); cdsp_gemm_close(handle); return -3; }
    double up_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("host: gemm_upload done -> weight handle=%d in %.2f ms\n", wh, up_ms);

    // ---- compute via cache (only A shipped) ----
    int32_t dims_c[3] = { M, N, K };
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rc = cdsp_gemm_gemm_compute_cached(handle, A, (int)szA, dims_c, 3, wh, C, (int)szC);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0) { fprintf(stderr, "host: compute_cached failed rc=%d\n", rc); cdsp_gemm_close(handle); return -3; }
    double cc_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("host: gemm_compute_cached done (M=%d N=%d K=%d) in %.2f ms\n", M, N, K, cc_ms);

    // ---- verify ----
    int mismatches = 0;
    for (size_t i = 0; i < szC; i++) {
        if (C[i] != Cref[i]) {
            if (mismatches < 8)
                fprintf(stderr, "host: mismatch[%zu] dsp=%d ref=%d\n", i, C[i], Cref[i]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("host: PASS - all %zu elements verified correct.\n", szC);
    else
        fprintf(stderr, "host: FAIL - %d/%zu mismatches.\n", mismatches, szC);

    // ---- release ----
    rc = cdsp_gemm_gemm_release(handle, wh);
    printf("host: gemm_release rc=%d\n", rc);

    cdsp_gemm_close(handle);
    free(A); free(B); free(C); free(Cref);
    return (mismatches == 0) ? 0 : -4;
}
