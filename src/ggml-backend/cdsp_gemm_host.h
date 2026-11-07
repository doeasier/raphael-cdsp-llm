//============================================================================
// cdsp_gemm_host.h - host (AP) side interface to our cDSP int8 GEMM skel.
//
// This is the only thing the ggml backend plugin needs to call. The actual
// FastRPC plumbing (libadsprpc / our QAIC-generated stub) lives in
// cdsp_gemm_host.cpp and is compiled only on Linux (the target device).
// On any other platform the functions are stubbed so the plugin still
// compiles (used for compile-time API validation on the Windows build host).
//============================================================================
#ifndef CDSP_GEMM_HOST_H
#define CDSP_GEMM_HOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-shot int8 general matrix multiplication:
//   C (M x N) = A (M x K) * B (K x N),  int8 in, int32 out (no scaling).
// The ggml backend applies per-vector scales on the host afterwards.
// Returns 0 on success, non-zero on failure (e.g. DSP not reachable).
int cdsp_gemm_compute(const int8_t * A, int M, int K,
                      const int8_t * B, int N, int32_t * C);

// ---- WEIGHT-CACHE API (Stage 5) -------------------------------------------
// Upload + prepack the (large) int8 weight Bt (K x N) on the DSP ONCE and get
// back a handle. `N` and `K` are the weight dimensions. Returns 0 on success.
int cdsp_gemm_upload(const int8_t * Bt, int K, int N, int32_t * handle_out);

// Compute C (M x N) = A (M x K) @ cached weight (K x N) using `handle`.
// Only the small activation A is transported; the weight stays on the DSP.
int cdsp_gemm_compute_cached(const int8_t * A, int M, int K, int N,
                             int32_t handle, int32_t * C);

// Release a previously uploaded weight handle, freeing its DSP memory.
int cdsp_gemm_release(int32_t handle);

#ifdef __cplusplus
}
#endif

#endif // CDSP_GEMM_HOST_H
