# src/ggml-backend — ggml cDSP backend plugin (Stage 2/2.5/5/6)

A `ggml` backend plugin that offloads `GGML_OP_MUL_MAT` to the cDSP.

| File | Purpose |
|---|---|
| `ggml-cdsp.cpp` | Backend/device/reg registration + `cdsp_mul_mat` (dequant → quant → FastRPC → rescale) + **LRU weight cache** (Stage 5) + unified `cdsp_can_mul_mat()` predicate and diagnostic modes (Stage 6). |
| `cdsp_gemm_host.{h,cpp}` | AP-side FastRPC session wrapper (`enable_unsigned_pd`, `cdsp_gemm_open`, `cdsp_gemm_compute`, `cdsp_gemm_upload`/`compute_cached`/`release`). |
| `ggml-backend-impl.h` | Verbatim copy of ggml 0.9.0 `ggml-backend-impl.h` (ABI pin). |
| `cdsp_preload.c` | `LD_PRELOAD` shim registering the plugin into llama.cpp builds with `GGML_BACKEND_DL=OFF` (where `GGML_BACKEND_PATH` is never reached). |
| `build_device.sh` | On-device build → `libggml-cdsp.so` (compiles the QAIC stub as C, links into the C++ lib). |
| `test_cdsp.cpp` | Single-backend correctness test (cDSP MUL_MAT vs int8 emulation). |
| `test_sched.cpp` | Multi-backend scheduler integration test (MUL_MAT→cDSP, relu→CPU). |
| `test_cache.cpp` | Scheduler-level weight-cache reuse test (compute twice, 2nd is a cache hit). |
| `test_q4k.cpp` | Real Q4_K / Q8_0 blocks built by ggml's own quantizer, CPU vs cDSP (guards the 144 B / 34 B block layouts). |

Build (on device, after headers are installed by `deploy_ggml_cdsp.py`):

```bash
GGML_INCLUDE=/data/local/tmp/ggml-include FASTRPC_INCLUDE=/usr/local/include/fastrpc \
ADSPRPC_LIBDIR=/usr/local/lib bash build_device.sh
```

Run a test:

```bash
GGML_CDSP_BACKEND=/data/local/tmp/ggml-backend/libggml-cdsp.so \
GGML_BACKEND_DIR=/data/local/tmp/ggml-backend \
LD_LIBRARY_PATH=/path/to/libggml:/usr/local/lib ./test_cdsp
```

**ggml semantics that bit us (do not regress):**

- `ggml_mul_mat(src0=weight [K,N], src1=activation [K,M])` → `dst [N,M]`.
- `Q4_K` block = `{uint16 d, uint16 dmin, uint8 scales[12], uint8 qs[128]}` = 144 B.
- `Q8_0` block = `{uint16 d (fp16 scale), int8 qs[32]}` = 34 B.
- `supports_op()` and `cdsp_mul_mat()` **must** use the same predicate, or the
  scheduler aborts the whole graph with `-1`.

Environment switches: `GGML_CDSP_CACHE_MB`, `GGML_CDSP_MAX_ELEMS`,
`GGML_CDSP_MIN_ELEMS`, `GGML_CDSP_DEBUG`, `GGML_CDSP_TRACE`,
`GGML_CDSP_NO_CACHE`, `GGML_CDSP_HOST_F32`, `GGML_CDSP_BACKEND_PATH`.
See `docs/llama-integration.md` for the full table and the llama.cpp recipe.
