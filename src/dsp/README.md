# src/dsp — cDSP GEMM kernel (Stage 0/1/6/7)

The DSP-side implementation of the int8 GEMM kernel, compiled into
`libgemm_skel.so` and loaded by the cDSP via FastRPC.

| File | Purpose |
|---|---|
| `gemm.idl` | FastRPC interface: `gemm_int8` (one-shot) + `gemm_upload`/`gemm_compute_cached`/`gemm_release` (weight-cache, Stage 5). |
| `gemm_dsp.c` | HVX (vrmpy) int8 GEMM + per-session weight-cache table. `open/close/rpc_init/gemm_int8/gemm_upload/gemm_compute_cached/gemm_release` all defined. `gemm_compute_cached` also accepts the Stage 7 **batched** encoding (`dims[0] = -count`). |
| `build_win.sh` | Hexagon SDK build → `../../prebuilt/libgemm_skel.so`. Needs `-mhvx` (not `-mhvx=128`). Regenerates `../gen`. |
| `AEEStdDef.h`, `AEEStdErr.h` | SDK headers referenced by the generated stub. |

**Stage 6:** `gemm_compute_with_bp()` now auto-selects one of three HVX paths —
`gemm_m1()` (M=1 decode, 4-way K unroll), `gemm_small_bp()` (Bp ≤ 512 KB,
L2-resident, 1 accumulator) and the M-blocked loop for large prompt batches.
`pack_a_row()` hoists A out of the innermost loop, and `g_aw` is a persistent
per-PD scratch buffer (allocated outside the HVX lock, freed in `close()`).
Measured up to **32.2 GMAC/s** on `128×512×512`; see `docs/architecture.md` §3.5
and `docs/next-steps.md` §2.

**Stage 7:** `gemm_compute_cached` recognises a negative `dims[0]` as a batch
marker: `dims = { -count, M0,N0,K0,h0, M1,N1,K1,h1, ... }`, with `A` and `C`
concatenated across groups. It walks up to 8 groups, validates each against its
cached weight and calls the same HVX kernel with A/C offsets — **one FastRPC
round-trip for a whole scheduler split**. No IDL change was needed because
`dimsLen`/`ALen`/`CLen` are already variable. Measured 2.505 → 1.962 ms for a
2-op pair (−21.6%); see `src/host/host_test_batch.c`.

Build (Windows, needs Hexagon SDK 4.3.0.0) — **run from this directory**, QAIC
resolves `gemm.idl` relative to the CWD:

```bash
cd src/dsp && bash build_win.sh
```

The output is also committed at `../../prebuilt/libgemm_skel.so` so the rest of the
pipeline works without the SDK. After replacing the skel, install it to **all
three** device paths (`/usr/lib/rfsa/cdsp`, `/usr/share/qcom/dsp/cdsp`,
`/data/local/tmp/cdsp_gemm`) and `systemctl restart cdsprpcd` — a stale copy in
any of them silently produces wrong results.
