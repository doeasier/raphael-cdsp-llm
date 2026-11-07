# src/ggml-backend — ggml cDSP backend plugin (Stage 2/2.5)

A `ggml` backend plugin that offloads `GGML_OP_MUL_MAT` to the cDSP.

| File | Purpose |
|---|---|
| `ggml-cdsp.cpp` | Backend/device/reg registration + `cdsp_mul_mat` (quant → FastRPC → rescale) + **LRU weight cache** (Stage 5). |
| `cdsp_gemm_host.{h,cpp}` | AP-side FastRPC session wrapper (`enable_unsigned_pd`, `cdsp_gemm_open`, `cdsp_gemm_compute`, `cdsp_gemm_upload`/`compute_cached`/`release`). |
| `ggml-backend-impl.h` | Verbatim copy of ggml 0.9.0 `ggml-backend-impl.h` (ABI pin). |
| `build_device.sh` | On-device build → `libggml-cdsp.so` (compiles the QAIC stub as C, links into the C++ lib). |
| `test_cdsp.cpp` | Single-backend correctness test (cDSP MUL_MAT vs int8 emulation). |
| `test_sched.cpp` | Multi-backend scheduler integration test (MUL_MAT→cDSP, relu→CPU). |
| `test_cache.cpp` | Scheduler-level weight-cache reuse test (compute twice, 2nd is a cache hit). |

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
