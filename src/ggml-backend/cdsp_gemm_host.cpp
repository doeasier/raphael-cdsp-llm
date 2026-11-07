//============================================================================
// cdsp_gemm_host.cpp - host (AP) side FastRPC wrapper for our cDSP int8 GEMM.
//
// Linux (target device): links against libadsprpc + the QAIC-generated stub
//   (gen/cdsp_gemm_stub.c) and talks to libgemm_skel.so running on the cDSP.
// Other platforms: stubbed out so the plugin compiles for API validation.
//============================================================================
#include "cdsp_gemm_host.h"

#include <cstring>
#include <mutex>

#ifdef __linux__
extern "C" {
#include "gemm.h"   // cdsp_gemm_open/close/rpc_init/gemm_int8, cdsp_gemm_URI
}
#include <remote.h>  // remote_session_control, DSPRPC_CONTROL_UNSIGNED_MODULE

// Full skel URI = base + "&_dom=cdsp" so the PD loads on the cDSP domain.
#define CDSP_FULL_URI cdsp_gemm_URI "&_dom=cdsp"

static int enable_unsigned_pd(void) {
    struct remote_rpc_control_unsigned_module data;
    memset(&data, 0, sizeof(data));
    data.enable = 1;
    data.domain = 3; // CDSP_DOMAIN_ID
    int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                    (void *)&data, (uint32_t)sizeof(data));
    return rc; // non-fatal if it fails; open() will report the real error
}
#else
// ---- host / non-Linux stub ------------------------------------------------
typedef uint64_t remote_handle64;
static int cdsp_gemm_open(const char *, remote_handle64 * h) { *h = 0; return -1; }
static int cdsp_gemm_close(remote_handle64) { return -1; }
static int cdsp_gemm_rpc_init(remote_handle64) { return -1; }
static int cdsp_gemm_gemm_int8(remote_handle64, const int8_t *, int,
                               const int8_t *, int, const int32_t *, int,
                               int32_t *, int) { return -1; }
#define CDSP_FULL_URI "file:///libgemm_skel.so?cdsp_gemm_skel_handle_invoke&_modver=1.0&_dom=cdsp"
#endif

static remote_handle64 g_h   = 0;
static bool             g_open = false;
static std::mutex       g_mtx;

static int ensure_session(void) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_open) return 0;
#ifdef __linux__
    enable_unsigned_pd();
    int rc = cdsp_gemm_open(CDSP_FULL_URI, &g_h);
    if (rc != 0) return rc;
    cdsp_gemm_rpc_init(g_h); // probe HVX; non-fatal
    g_open = true;
    return 0;
#else
    return -1;
#endif
}

int cdsp_gemm_compute(const int8_t * A, int M, int K,
                      const int8_t * B, int N, int32_t * C) {
    int rc = ensure_session();
    if (rc != 0) return rc;

    int32_t dims[3] = { M, N, K };
    return cdsp_gemm_gemm_int8(
        g_h,
        A, (int)((size_t)M * K),
        B, (int)((size_t)K * N),
        dims, 3,
        C, (int)((size_t)M * N));
}
