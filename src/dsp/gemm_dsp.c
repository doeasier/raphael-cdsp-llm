//============================================================================
// gemm_dsp.c - cDSP (Hexagon 690 / V66) implementation of the int8 GEMM kernel.
//
// Build target: Hexagon ELF shared object (libgemm_skel.so) loaded by the cDSP
// via FastRPC. Compiled for the V66 DSP WITH HVX (1024-bit / 128-byte vectors).
//
// Layout convention (row-major):
//   A is M x K   (activations),   int8
//   B is K x N   (weights),        int8
//   C is M x N   (accumulators),   int32   C[m][n] = sum_k A[m][k] * B[k][n]
// The host is responsible for any per-row/per-tensor rescale of C to float.
//
// HVX kernel (Stage 4):
//   vrmpy (vector reduce multiply) over a group of 4 bytes does:
//       out[g] += a0*b0 + a1*b1 + a2*b2 + a3*b3      (signed int8, per group g)
//   We process K in steps of 4 and N in tiles of 32 columns:
//     * B is prepacked into an interleaved tile layout so that one 128-byte
//       load yields, for column-block c0..c0+31 and k-block k0..k0+3:
//           group g = { B[k0+0][c0+g], B[k0+1][c0+g], B[k0+2][c0+g], B[k0+3][c0+g] }
//     * A is splatted (Q6_V_vsplat_R) so each group = {a0,a1,a2,a3}.
//   One vrmpyacc instruction thus computes 32 partial dot products at once.
//============================================================================

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm.h"                 // QAIC-generated header (named after the .idl file)

#include <hexagon_types.h>       // HVX_Vector
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>  // Q6_* HVX intrinsics (vrmpy, vsplat)
#include <qurt/qurt_hvx.h>       // qurt_hvx_lock / qurt_hvx_unlock

/* ---- HVX geometry ------------------------------------------------------- */
#define HVX_VECTOR_BYTES 128u    /* 1024-bit HVX vector                   */
#define K_STEP           4       /* vrmpy reduces over 4 bytes per group  */
#define N_TILE           32      /* output columns handled by one vector  */

/* ---- per-session open/close (DSP side) ---------------------------------- */
int __QAIC_HEADER(cdsp_gemm_open)(const char* uri, remote_handle64* handle) {
    (void)uri;
    void* tptr = malloc(1);
    *handle = (remote_handle64)(uintptr_t)tptr;
    return 0;
}

int __QAIC_HEADER(cdsp_gemm_close)(remote_handle64 handle) {
    if (handle) free((void*)(uintptr_t)handle);
    return 0;
}

/* ---- one-time DSP init ------------------------------------------------- */
int __QAIC_HEADER(cdsp_gemm_rpc_init)(remote_handle64 _h) {
    (void)_h;
    /* HVX unit is locked/unlocked per gemm_int8 invocation, so nothing to
     * persist here. We could probe HVX availability, but the lock call itself
     * is the authoritative check. */
    return 0;
}

/* ---- scalar fallback (reference; never runs under -mhvx build) ---------- */
static int gemm_int8_scalar(remote_handle64 _h,
                            const int8* A, const int8* B,
                            int M, int N, int K, int32* C) {
    (void)_h;
    for (int m = 0; m < M; m++) {
        const int8  *Ar = A + (size_t)m * K;
        int32       *Cr = C + (size_t)m * N;
        for (int n = 0; n < N; n++) Cr[n] = 0;
        for (int k = 0; k < K; k++) {
            int8 a = Ar[k];
            const int8 *Br = B + (size_t)k * N;
            for (int n = 0; n < N; n++) Cr[n] += (int32)a * (int32)Br[n];
        }
    }
    return 0;
}

/* ---- int8 GEMM: C = A * B  (int8 x int8 -> int32), HVX path ------------- */
int __QAIC_HEADER(cdsp_gemm_gemm_int8)(remote_handle64 _h,
                                        const int8* A, int ALen,
                                        const int8* B, int BLen,
                                        const int32* dims, int dimsLen,
                                        int32* C,  int CLen) {
    (void)_h;

    /* dimensions come in the "dims" sequence: [M, N, K] */
    if (dims == NULL || dimsLen < 3) return -1;
    int M = dims[0], N = dims[1], K = dims[2];

    /* bounds / contract checks */
    if (C == NULL || A == NULL || B == NULL) return -2;
    if (M <= 0 || N <= 0 || K <= 0) return -3;
    if (ALen < M * K) return -4;
    if (BLen < K * N) return -5;
    if (CLen < M * N) return -6;

    /* Lock the 128-byte HVX unit up front. This must wrap ALL code in this
     * function, including the prepack loop - the -mhvx build lets the compiler
     * emit HVX for any hot loop, and executing HVX outside the lock faults. */
    int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (rc != QURT_EOK) {
        /* No HVX available: fall back to the scalar reference. (On the K20 Pro
         * cDSP this should never happen, but be defensive.) */
        return gemm_int8_scalar(_h, A, B, M, N, K, C);
    }

    const int Kblk   = (K + K_STEP - 1) / K_STEP;          /* k-blocks of 4   */
    const int nTiles = (N + N_TILE - 1) / N_TILE;          /* 32-col tiles   */
    const size_t bp_size = (size_t)nTiles * (size_t)Kblk * HVX_VECTOR_BYTES;

    char  *raw = (char*)malloc(bp_size + (HVX_VECTOR_BYTES - 1));
    if (raw == NULL) { qurt_hvx_unlock(); return -10; }
    /* 128-byte align the prepack buffer so vector loads/stores are aligned. */
    int8_t *Bp = (int8_t*)(((uintptr_t)raw + (HVX_VECTOR_BYTES - 1)) &
                           ~((uintptr_t)(HVX_VECTOR_BYTES - 1)));

    /* --- prepack B into interleaved tile layout --------------------------
     *   Bp[(t*Kblk + kb)*128 + j*4 + i] = B[(k0+i)*N + (c0+j)]   (0 outside range)
     * so that one 128-byte load at (t,kb) gives, for group g (=column j):
     *   { B[k0+0][c0+g], B[k0+1][c0+g], B[k0+2][c0+g], B[k0+3][c0+g] }
     * ------------------------------------------------------------------ */
    for (int t = 0; t < nTiles; t++) {
        const int c0 = t * N_TILE;
        for (int kb = 0; kb < Kblk; kb++) {
            const int k0 = kb * K_STEP;
            int8_t *dst = Bp + ((size_t)t * Kblk + kb) * HVX_VECTOR_BYTES;
            for (int j = 0; j < N_TILE; j++) {
                const int col = c0 + j;
                for (int i = 0; i < K_STEP; i++) {
                    const int kk = k0 + i;
                    dst[j * K_STEP + i] =
                        (kk < K && col < N) ? B[(size_t)kk * N + col] : 0;
                }
            }
        }
    }

    /* 128-byte aligned accumulator scratch (static is fine: single-threaded
     * FastRPC invoke). We store the vector here, then copy only the valid
     * columns to C (so C needs no alignment). */
    static int32_t accbuf[32] __attribute__((aligned(HVX_VECTOR_BYTES)));

    for (int m = 0; m < M; m++) {
        const int8_t *Ar = A + (size_t)m * K;
        int32_t      *Cr = C + (size_t)m * N;

        for (int t = 0; t < nTiles; t++) {
            const int c0  = t * N_TILE;
            const int cols = (c0 + N_TILE <= N) ? N_TILE : (N - c0);

            HVX_Vector Vacc = Q6_V_vsplat_R(0);   /* zero accumulator */

            for (int kb = 0; kb < Kblk; kb++) {
                const int k0 = kb * K_STEP;

                /* Build the A operand: 4 consecutive A values, repeated across
                 * all 32 groups via a word-splat. Out-of-range k -> 0. */
                uint8_t a[4];
                for (int i = 0; i < K_STEP; i++) {
                    const int kk = k0 + i;
                    const int8_t v = (kk < K) ? Ar[kk] : 0;
                    a[i] = (uint8_t)v;            /* 2's-complement byte */
                }
                const uint32_t aw = (uint32_t)a[0]
                                  | ((uint32_t)a[1] << 8)
                                  | ((uint32_t)a[2] << 16)
                                  | ((uint32_t)a[3] << 24);
                const HVX_Vector va = Q6_V_vsplat_R(aw);

                /* Load the prepacked B tile (aligned 128-byte load). */
                const HVX_Vector vb =
                    *(const HVX_Vector*)(Bp + ((size_t)t * Kblk + kb) * HVX_VECTOR_BYTES);

                /* signed int8 x signed int8 reduce-multiply-accumulate. */
                Vacc = Q6_Vw_vrmpyacc_VwVbVb(Vacc, va, vb);
            }

            *(HVX_Vector*)accbuf = Vacc;
            for (int j = 0; j < cols; j++) Cr[c0 + j] = accbuf[j];
        }
    }

    qurt_hvx_unlock();
    free(raw);
    return 0;
}
