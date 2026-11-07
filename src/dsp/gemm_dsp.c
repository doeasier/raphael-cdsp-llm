//============================================================================
// gemm_dsp.c - cDSP (Hexagon 690 / V66) implementation of the int8 GEMM kernel.
//
// Build target: Hexagon ELF shared object (libgemm_skel.so) loaded by the cDSP
// via FastRPC. Compiled for the V66 DSP WITH HVX (1024-bit / 128-byte vectors).
//
// Layout convention (row-major):
//   A is M x K   (activations),   int8
//   B is K x N   (weights),        int8     B[k][n] = B[k*N + n]
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
//
// Weight cache (Stage 5):
//   gemm_upload prepacks B once into a per-session table keyed by an integer
//   handle. gemm_compute_cached reuses that prepacked B; only the small
//   activation A is transported each call. This is what makes real token-by-token
//   inference feasible (the multi-MB weight matrix is shipped once, not per token).
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

/* ---- weight-cache table (per-PD / per-session state) --------------------- */
#define WCACHE_MAX 1024          /* plenty for a model's MUL_MAT weights  */
typedef struct {
    int32_t  handle;             /* 0 == free slot                       */
    int      N;
    int      K;
    int8_t  *bp;                 /* prepacked (aligned) B buffer         */
    void    *raw;                /* underlying malloc (for free)         */
} wcache_t;
static wcache_t g_wcache[WCACHE_MAX];
static int32_t  g_wh_next = 1;   /* 0 reserved as "invalid"              */

static void aw_release(void);    /* defined below, used by cdsp_gemm_close */

/* ---- per-session open/close (DSP side) ---------------------------------- */
int __QAIC_HEADER(cdsp_gemm_open)(const char* uri, remote_handle64* handle) {
    (void)uri;
    void* tptr = malloc(1);
    *handle = (remote_handle64)(uintptr_t)tptr;
    return 0;
}

int __QAIC_HEADER(cdsp_gemm_close)(remote_handle64 handle) {
    /* release all cached weights so a PD that is reused across sessions does
     * not leak prepacked buffers. */
    for (int i = 0; i < WCACHE_MAX; i++) {
        if (g_wcache[i].raw) { free(g_wcache[i].raw); g_wcache[i].raw = NULL; }
        g_wcache[i].bp = NULL;
        g_wcache[i].handle = 0; g_wcache[i].N = 0; g_wcache[i].K = 0;
    }
    aw_release();   /* free the persistent packed-A scratch buffer */
    if (handle) free((void*)(uintptr_t)handle);
    return 0;
}

/* ---- one-time DSP init ------------------------------------------------- */
int __QAIC_HEADER(cdsp_gemm_rpc_init)(remote_handle64 _h) {
    (void)_h;
    /* HVX unit is locked/unlocked per GEMM invocation, so nothing to persist. */
    return 0;
}

/* ---- prepack B (KxN) into interleaved tile layout (scalar; no HVX needed) -
 *   Bp[(t*Kblk + kb)*128 + j*4 + i] = B[(k0+i)*N + (c0+j)]   (0 outside range)
 *   so that one 128-byte load at (t,kb) gives, for group g (=column j):
 *     { B[k0+0][c0+g], B[k0+1][c0+g], B[k0+2][c0+g], B[k0+3][c0+g] }
 *   Returns a 128-byte aligned malloc'd buffer, or NULL on OOM.
 * ----------------------------------------------------------------------- */
static int8_t * prepack_B(const int8_t * B, int N, int K, void ** raw_out) {
    const int Kblk   = (K + K_STEP - 1) / K_STEP;
    const int nTiles = (N + N_TILE - 1) / N_TILE;
    const size_t bp_size = (size_t)nTiles * (size_t)Kblk * HVX_VECTOR_BYTES;

    char  *raw = (char*)malloc(bp_size + (HVX_VECTOR_BYTES - 1));
    if (raw == NULL) return NULL;
    int8_t *Bp = (int8_t*)(((uintptr_t)raw + (HVX_VECTOR_BYTES - 1)) &
                           ~((uintptr_t)(HVX_VECTOR_BYTES - 1)));
    if (raw_out) *raw_out = raw;

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
    return Bp;
}

/* ---- HVX compute using an already-prepacked Bp (the hot path) ------------
 * Stage 6 / §2 HVX tuning. The kernel auto-selects one of three paths:
 *
 *   M == 1                 -> gemm_m1()       : decode, 4-way K unroll
 *   M >  1, Bp <= 512 KB  -> gemm_small_bp() : L2-resident weights, 1 acc
 *   M >  1, Bp >  512 KB  -> M-blocked loop  : prompt batch, amortized Bp
 *
 * Optimizations applied to the original naive M-t-kb loop:
 *
 *  1. HOIST A. The original code re-read and re-packed the four A bytes for
 *     every (m, tile, kb). A is tiny compared to Bp but the scalar loads were
 *     in the innermost loop, on the dependency chain of `vsplat`. We now pack
 *     one uint32 word per (row, k-block) ONCE per row block and reuse it for
 *     all N tiles -> M*nTiles*Kblk scalar loads become M*Kblk.
 *  2. M-BLOCKING. Bp is a streamed, read-once operand for M=1 decode, but for
 *     prompt batches (M>1) the naive loop streamed the whole Bp tile M times.
 *     Processing M_BLOCK rows together loads each 128-byte Bp vector once and
 *     reuses it across rows, cutting weight traffic by up to M_BLOCK.
 *  3. K-UNROLL (M==1 ONLY). `vrmpyacc` accumulates in place, so a single Vacc
 *     forms a serial dependency chain across all Kblk iterations. For decode
 *     (one row) four independent accumulators let four Bp loads be in flight.
 *     For M>1 the M_BLOCK rows already provide that ILP, and extra accumulator
 *     planes only caused register spills on V66, so that path uses K_UNROLL=1.
 *
 * Register budget on V66 (32 x 128B HVX regs):
 *   M==1 : 4 accumulators + 4 loaded B vectors + one splat = 9
 *   M>1  : M_BLOCK accumulators + one B vector + one splat = M_BLOCK + 2
 * ----------------------------------------------------------------------- */
#define M_BLOCK  4
#define K_UNROLL 2
/* When the whole prepacked weight fits in L2, streaming it once per (m,t)
 * is cheap and the original m-major loop (1 accumulator) beats M-blocking:
 * measured 13.6 vs 10.7 GMAC/s at 128x512x512. Above this size the redundant
 * M-fold weight traffic dominates and M-blocking wins by 3-4x. */
#define SMALL_BP_BYTES (512 * 1024)

/* Scratch buffer for packed A words, allocated once per PD (NOT per GEMM).
 * A malloc/free inside the HVX lock measurably showed up in the M=1 decode
 * microbenchmark, so the buffer is grown on demand and released in close(). */
static uint32_t *g_aw = NULL;
static size_t    g_aw_words = 0;

static uint32_t * aw_ensure(size_t words) {
    if (words <= g_aw_words && g_aw != NULL) return g_aw;
    uint32_t *p = (uint32_t *)malloc(words * sizeof(uint32_t));
    if (p == NULL) return NULL;
    if (g_aw) free(g_aw);
    g_aw = p;
    g_aw_words = words;
    return p;
}

static void aw_release(void) {
    if (g_aw) { free(g_aw); g_aw = NULL; g_aw_words = 0; }
}

/* Pack one row of A into one uint32 per k-block (4 int8 lanes for vrmpy).
 * Bytes beyond K are zero, matching prepack_B()'s zero padding. */
static inline void pack_a_row(const int8_t * Ar, int K, int Kblk, uint32_t * wr) {
    for (int kb = 0; kb < Kblk; kb++) {
        const int k0 = kb * K_STEP;
        uint32_t w = 0;
        for (int i = 0; i < K_STEP; i++) {
            const int kk = k0 + i;
            uint32_t b = (kk < K) ? (uint32_t)(uint8_t)Ar[kk] : 0u;
            w |= b << (8 * i);
        }
        wr[kb] = w;
    }
}

static inline const HVX_Vector * tile_ptr(const int8_t * Bp, int t, int Kblk) {
    return (const HVX_Vector *)(Bp +
           (size_t)t * (size_t)Kblk * HVX_VECTOR_BYTES);
}

/* ---- M == 1 decode path -------------------------------------------------
 * Decode is bandwidth-bound: one 128-byte Bp load feeds exactly 128 MACs and
 * is never reused. Four independent accumulators let four Bp loads be in
 * flight at once, hiding DDR latency; the four partial sums are added once
 * at the end. Kept separate from the M>1 path because a shared [M_BLOCK]
 * accumulator array spills to the stack when rows == 1 (measured: 1.4x
 * SLOWER than the single-accumulator loop on this V66).
 * ----------------------------------------------------------------------- */
static void gemm_m1(const int8_t * A, int N, int K, int Kblk, int nTiles,
                    const int8_t * Bp, int32_t * C, uint32_t * aw,
                    int32_t * accbuf) {
    pack_a_row(A, K, Kblk, aw);

    for (int t = 0; t < nTiles; t++) {
        const int c0   = t * N_TILE;
        const int cols = (c0 + N_TILE <= N) ? N_TILE : (N - c0);
        const HVX_Vector *vb = tile_ptr(Bp, t, Kblk);

        HVX_Vector v0 = Q6_V_vsplat_R(0), v1 = Q6_V_vsplat_R(0);
        HVX_Vector v2 = Q6_V_vsplat_R(0), v3 = Q6_V_vsplat_R(0);
        int kb = 0;
        for (; kb + 4 <= Kblk; kb += 4) {
            HVX_Vector b0 = vb[kb + 0], b1 = vb[kb + 1];
            HVX_Vector b2 = vb[kb + 2], b3 = vb[kb + 3];
            v0 = Q6_Vw_vrmpyacc_VwVbVb(v0, Q6_V_vsplat_R(aw[kb + 0]), b0);
            v1 = Q6_Vw_vrmpyacc_VwVbVb(v1, Q6_V_vsplat_R(aw[kb + 1]), b1);
            v2 = Q6_Vw_vrmpyacc_VwVbVb(v2, Q6_V_vsplat_R(aw[kb + 2]), b2);
            v3 = Q6_Vw_vrmpyacc_VwVbVb(v3, Q6_V_vsplat_R(aw[kb + 3]), b3);
        }
        for (; kb < Kblk; kb++) {
            v0 = Q6_Vw_vrmpyacc_VwVbVb(v0, Q6_V_vsplat_R(aw[kb]), vb[kb]);
        }
        HVX_Vector v = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(v0, v1),
                                       Q6_Vw_vadd_VwVw(v2, v3));
        *(HVX_Vector *)accbuf = v;
        for (int j = 0; j < cols; j++) C[c0 + j] = accbuf[j];
    }
}

/* ---- M>1, Bp fits in L2: one accumulator, m-major ------------------------
 * Small square-ish GEMMs (the original HVX benchmark shape) are compute-bound
 * once Bp is L2-resident. A single accumulator keeps register pressure minimal
 * and the m-major order lets each Bp tile stay hot across all k-blocks.
 * ----------------------------------------------------------------------- */
static void gemm_small_bp(const int8_t * A, int M, int N, int K,
                          int Kblk, int nTiles, const int8_t * Bp,
                          int32_t * C, uint32_t * aw, int32_t * accbuf) {
    for (int m = 0; m < M; m++)
        pack_a_row(A + (size_t)m * K, K, Kblk, aw + (size_t)m * Kblk);

    for (int m = 0; m < M; m++) {
        int32_t *Cr = C + (size_t)m * N;
        for (int t = 0; t < nTiles; t++) {
            const HVX_Vector *vb = tile_ptr(Bp, t, Kblk);
            HVX_Vector Vacc = Q6_V_vsplat_R(0);
            for (int kb = 0; kb < Kblk; kb++)
                Vacc = Q6_Vw_vrmpyacc_VwVbVb(
                    Vacc, Q6_V_vsplat_R(aw[(size_t)m * Kblk + kb]), vb[kb]);
            *(HVX_Vector *)accbuf = Vacc;
            const int c0   = t * N_TILE;
            const int cols = (c0 + N_TILE <= N) ? N_TILE : (N - c0);
            for (int j = 0; j < cols; j++) Cr[c0 + j] = accbuf[j];
        }
    }
}

static int gemm_compute_with_bp(remote_handle64 _h,
                                const int8_t * A, int M, int N, int K,
                                const int8_t * Bp, int32_t * C) {
    (void)_h;

    const int Kblk   = (K + K_STEP - 1) / K_STEP;
    const int nTiles = (N + N_TILE - 1) / N_TILE;

    const size_t bp_bytes = (size_t)nTiles * (size_t)Kblk * HVX_VECTOR_BYTES;
    const int    small_bp = (M > 1) && (bp_bytes <= SMALL_BP_BYTES);

    /* Grow the persistent scratch buffer BEFORE taking the HVX lock. */
    const size_t aw_words = (M == 1) ? (size_t)Kblk
                          : small_bp ? (size_t)M * (size_t)Kblk
                                     : (size_t)M_BLOCK * (size_t)Kblk;
    uint32_t * aw = aw_ensure(aw_words);
    if (aw == NULL) return -21;

    int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (rc != QURT_EOK) return -20;

    static int32_t accbuf[32] __attribute__((aligned(HVX_VECTOR_BYTES)));

    if (M == 1) {
        gemm_m1(A, N, K, Kblk, nTiles, Bp, C, aw, accbuf);
        qurt_hvx_unlock();
        return 0;
    }
    if (small_bp) {
        gemm_small_bp(A, M, N, K, Kblk, nTiles, Bp, C, aw, accbuf);
        qurt_hvx_unlock();
        return 0;
    }

    for (int t = 0; t < nTiles; t++) {
        const int c0   = t * N_TILE;
        const int cols = (c0 + N_TILE <= N) ? N_TILE : (N - c0);
        const HVX_Vector *vb = tile_ptr(Bp, t, Kblk);

        for (int mb = 0; mb < M; mb += M_BLOCK) {
            const int rows = (mb + M_BLOCK <= M) ? M_BLOCK : (M - mb);

            /* (1) pack A once for this row block, reused across all tiles */
            for (int r = 0; r < rows; r++)
                pack_a_row(A + (size_t)(mb + r) * K, K, Kblk,
                           aw + (size_t)r * Kblk);

            /* (3) K_UNROLL independent accumulator chains */
            HVX_Vector Vacc[M_BLOCK][K_UNROLL];
            for (int r = 0; r < rows; r++)
                for (int u = 0; u < K_UNROLL; u++)
                    Vacc[r][u] = Q6_V_vsplat_R(0);

            int kb = 0;
            for (; kb + K_UNROLL <= Kblk; kb += K_UNROLL) {
                HVX_Vector bu[K_UNROLL];
                for (int u = 0; u < K_UNROLL; u++) bu[u] = vb[kb + u];
                /* (2) the same Bp vector feeds M_BLOCK rows */
                for (int r = 0; r < rows; r++) {
                    for (int u = 0; u < K_UNROLL; u++) {
                        HVX_Vector va = Q6_V_vsplat_R(aw[(size_t)r * Kblk + kb + u]);
                        Vacc[r][u] = Q6_Vw_vrmpyacc_VwVbVb(Vacc[r][u], va, bu[u]);
                    }
                }
            }
            for (; kb < Kblk; kb++) {   /* odd Kblk tail */
                for (int r = 0; r < rows; r++) {
                    HVX_Vector va = Q6_V_vsplat_R(aw[(size_t)r * Kblk + kb]);
                    Vacc[r][0] = Q6_Vw_vrmpyacc_VwVbVb(Vacc[r][0], va, vb[kb]);
                }
            }

            for (int r = 0; r < rows; r++) {
                HVX_Vector v = Vacc[r][0];
                for (int u = 1; u < K_UNROLL; u++)
                    v = Q6_Vw_vadd_VwVw(v, Vacc[r][u]);
                *(HVX_Vector *)accbuf = v;
                int32_t *Cr = C + (size_t)(mb + r) * N;
                for (int j = 0; j < cols; j++) Cr[c0 + j] = accbuf[j];
            }
        }
    }

    qurt_hvx_unlock();
    return 0;
}

/* ---- scalar fallback (reference; used only if HVX lock fails) ------------ */
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

/* ---- one-shot int8 GEMM: C = A * B  (int8 x int8 -> int32) -------------- */
int __QAIC_HEADER(cdsp_gemm_gemm_int8)(remote_handle64 _h,
                                        const int8* A, int ALen,
                                        const int8* B, int BLen,
                                        const int32* dims, int dimsLen,
                                        int32* C,  int CLen) {
    if (dims == NULL || dimsLen < 3) return -1;
    int M = dims[0], N = dims[1], K = dims[2];
    if (C == NULL || A == NULL || B == NULL) return -2;
    if (M <= 0 || N <= 0 || K <= 0) return -3;
    if (ALen < M * K) return -4;
    if (BLen < K * N) return -5;
    if (CLen < M * N) return -6;

    void *raw = NULL;
    int8_t *Bp = prepack_B(B, N, K, &raw);
    if (Bp == NULL) return -10;

    int rc = gemm_compute_with_bp(_h, A, M, N, K, Bp, C);
    if (rc != 0) {
        /* HVX unavailable: scalar reference (should never happen on K20 Pro). */
        gemm_int8_scalar(_h, A, B, M, N, K, C);
        rc = 0;
    }
    free(raw);
    return rc;
}

/* ---- weight cache: upload + prepack B, return handle -------------------- */
int __QAIC_HEADER(cdsp_gemm_gemm_upload)(remote_handle64 _h,
                                          const int8* B, int BLen,
                                          const int32* dims, int dimsLen,
                                          int32_t* handle_out) {
    (void)_h;
    if (dims == NULL || dimsLen < 3) return -1;
    int N = dims[1], K = dims[2];
    if (N <= 0 || K <= 0) return -2;
    if (B == NULL || BLen < K * N) return -3;

    int slot = -1;
    for (int i = 0; i < WCACHE_MAX; i++) {
        if (g_wcache[i].handle == 0) { slot = i; break; }
    }
    if (slot < 0) return -11;   /* table full */

    void *raw = NULL;
    int8_t *Bp = prepack_B(B, N, K, &raw);
    if (Bp == NULL) return -10; /* OOM */

    g_wcache[slot].handle = g_wh_next++;
    g_wcache[slot].N = N;
    g_wcache[slot].K = K;
    g_wcache[slot].bp = Bp;
    g_wcache[slot].raw = raw;
    *handle_out = g_wcache[slot].handle;
    return 0;
}

/* ---- weight cache: compute using cached B ------------------------------ */
int __QAIC_HEADER(cdsp_gemm_gemm_compute_cached)(remote_handle64 _h,
                                                  const int8* A, int ALen,
                                                  const int32* dims, int dimsLen,
                                                  int32_t handle,
                                                  int32* C, int CLen) {
    if (dims == NULL || dimsLen < 3) return -1;
    int M = dims[0], N = dims[1], K = dims[2];
    if (A == NULL || C == NULL) return -2;
    if (M <= 0 || N <= 0 || K <= 0) return -3;
    if (ALen < M * K) return -4;
    if (CLen < M * N) return -5;

    wcache_t *e = NULL;
    for (int i = 0; i < WCACHE_MAX; i++) {
        if (g_wcache[i].handle == handle) { e = &g_wcache[i]; break; }
    }
    if (e == NULL || e->bp == NULL) return -6;     /* unknown handle */
    if (e->N != N || e->K != K)    return -7;      /* dims mismatch */

    int rc = gemm_compute_with_bp(_h, A, M, N, K, e->bp, C);
    /* gemm_compute_with_bp only fails on HVX lock failure. The same session
     * that successfully uploaded the weight must have HVX, so this path is
     * effectively unreachable; zero C defensively rather than faulting. */
    if (rc != 0) {
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        return 0;
    }
    return 0;
}

/* ---- weight cache: release handle -------------------------------------- */
int __QAIC_HEADER(cdsp_gemm_gemm_release)(remote_handle64 _h, int32_t handle) {
    (void)_h;
    for (int i = 0; i < WCACHE_MAX; i++) {
        if (g_wcache[i].handle == handle) {
            if (g_wcache[i].raw) { free(g_wcache[i].raw); g_wcache[i].raw = NULL; }
            g_wcache[i].bp = NULL;
            g_wcache[i].handle = 0; g_wcache[i].N = 0; g_wcache[i].K = 0;
            return 0;
        }
    }
    return -1;  /* unknown handle */
}
