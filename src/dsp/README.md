# src/dsp — cDSP GEMM kernel (Stage 0/1)

The DSP-side implementation of the int8 GEMM kernel, compiled into
`libgemm_skel.so` and loaded by the cDSP via FastRPC.

| File | Purpose |
|---|---|
| `gemm.idl` | FastRPC interface: `gemm_int8` (one-shot) + `gemm_upload`/`gemm_compute_cached`/`gemm_release` (weight-cache, Stage 5). |
| `gemm_dsp.c` | HVX (vrmpy) int8 GEMM + per-session weight-cache table. `open/close/rpc_init/gemm_int8/gemm_upload/gemm_compute_cached/gemm_release` all defined. |
| `build_win.sh` | Hexagon SDK build → `../../prebuilt/libgemm_skel.so`. Needs `-mhvx` (not `-mhvx=128`). Regenerates `../gen`. |
| `AEEStdDef.h`, `AEEStdErr.h` | SDK headers referenced by the generated stub. |

Build (Windows, needs Hexagon SDK 4.3.0.0):

```bash
bash build_win.sh
```

The output is also committed at `../../prebuilt/libgemm_skel.so` so the rest of the
pipeline works without the SDK. See `docs/architecture.md` §3 for the HVX kernel.
