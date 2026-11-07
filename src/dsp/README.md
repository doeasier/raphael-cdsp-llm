# src/dsp — cDSP GEMM kernel (Stage 0/1)

The DSP-side implementation of the int8 GEMM kernel, compiled into
`libgemm_skel.so` and loaded by the cDSP via FastRPC.

| File | Purpose |
|---|---|
| `gemm.idl` | FastRPC interface: `C(M×N,int32) = A(M×K,int8) · B(K×N,int8)`. |
| `gemm_dsp.c` | HVX (vrmpy) int8 GEMM. `open/close/rpc_init/gemm_int8` are all defined. |
| `build_win.sh` | Hexagon SDK build → `../../prebuilt/libgemm_skel.so`. Needs `-mhvx` (not `-mhvx=128`). |
| `AEEStdDef.h`, `AEEStdErr.h` | SDK headers referenced by the generated stub. |

Build (Windows, needs Hexagon SDK 4.3.0.0):

```bash
bash build_win.sh
```

The output is also committed at `../../prebuilt/libgemm_skel.so` so the rest of the
pipeline works without the SDK. See `docs/architecture.md` §3 for the HVX kernel.
