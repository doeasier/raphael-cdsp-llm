//============================================================================
// host_test.c - AP (Linux on K20 Pro) side correctness + perf test for the
// cDSP int8 GEMM kernel. Generates random int8 A(MxK) and B(KxN), dispatches
// the GEMM to the cDSP via FastRPC, and verifies C == reference(A*B) on the
// host. Also reports wall-clock DSP compute time.
//
// Compile on-device (aarch64) against the QAIC-generated cdsp_gemm.h and
// libfastrpc. See Makefile (DEVICE build target).
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemm.h"               // QAIC-generated header (named after the .idl file)
#include <remote.h>             // remote_session_control, unsigned-PD control

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

/* Enable loading our unsigned skel .so (must run before the first FastRPC). */
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
    /* single element reference: C[m][n] = sum_k A[m][k]*B[k][n] */
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
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    int8_t *A = malloc(szA), *B = malloc(szB);
    int32_t *C = malloc(szC * 4), *Cref = malloc(szC * 4);
    if (!A || !B || !C || !Cref) { fprintf(stderr, "malloc fail\n"); return -1; }

    srand(12345);
    for (size_t i = 0; i < szA; i++) A[i] = (int8_t)(rand() & 0x1F) - 8; /* small range */
    for (size_t i = 0; i < szB; i++) B[i] = (int8_t)(rand() & 0x1F) - 8;

    /* host reference (for verification only) */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            Cref[(size_t)m * N + n] = ref_gemm(A, B, m, n, M, N, K);

    enable_unsigned_pd();

    int32_t dims[3] = { M, N, K };

    remote_handle64 handle = 0;
    int rc = cdsp_gemm_open(cdsp_gemm_URI CDSP_DOMAIN, &handle);
    if (rc != 0) {
        fprintf(stderr, "host: cdsp_gemm_open failed rc=%d\n", rc);
        return -2;
    }
    printf("host: cDSP session open, handle=%llu\n", (unsigned long long)handle);

    rc = cdsp_gemm_rpc_init(handle);
    if (rc != 0) fprintf(stderr, "host: rpc_init rc=%d (non-fatal)\n", rc);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rc = cdsp_gemm_gemm_int8(handle, A, (int)szA, B, (int)szB, dims, 3, C, (int)szC);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0) {
        fprintf(stderr, "host: gemm_int8 failed rc=%d\n", rc);
        cdsp_gemm_close(handle);
        return -3;
    }
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("host: gemm_int8 done (M=%d N=%d K=%d) in %.2f ms\n", M, N, K, ms);

    /* DEBUG: for small matrices, dump A, B, full C vs Cref */
    if (szC <= 64) {
        printf("host: --- A (int8) ---\n");
        for (size_t i = 0; i < szA; i++) printf("%d ", (int)(int8_t)A[i]);
        printf("\nhost: --- B (int8) ---\n");
        for (size_t i = 0; i < szB; i++) printf("%d ", (int)(int8_t)B[i]);
        printf("\nhost: --- C (DSP) ---\n");
        for (size_t i = 0; i < szC; i++) printf("%d ", C[i]);
        printf("\nhost: --- Cref (host) ---\n");
        for (size_t i = 0; i < szC; i++) printf("%d ", Cref[i]);
        printf("\n");
    }

    /* verify */
    int mismatches = 0;
    for (size_t i = 0; i < szC; i++) {
        if (C[i] != Cref[i]) {
            if (mismatches < 8)
                fprintf(stderr, "host: mismatch[%zu] dsp=%d ref=%d\n",
                        i, C[i], Cref[i]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("host: PASS - all %zu elements verified correct.\n", szC);
    else
        fprintf(stderr, "host: FAIL - %d/%zu mismatches.\n", mismatches, szC);

    cdsp_gemm_close(handle);
    free(A); free(B); free(C); free(Cref);
    return (mismatches == 0) ? 0 : -4;
}
