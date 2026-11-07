# Architecture & kernel notes

## 1. Big picture

The Redmi K20 Pro (raphael) SoC is the Snapdragon 855 with a **Hexagon 690**
DSP. The "compute" Hexagon core (aDSP/cDSP domain) can run hand-written kernels
and is reached from the applications processor (AP, where Linux/llama.cpp runs)
through **FastRPC** — a Qualcomm RPC mechanism that marshals arguments into shared
memory and invokes a "skeleton" (`.so`) loaded in the DSP's userspace (the "PD",
process domain).

We exploit this to turn the cDSP into an **`ggml` accelerator**: a custom backend
plugin (`libggml-cdsp.so`) that claims only the `GGML_OP_MUL_MAT` node, quantizes
the operands to int8 on the AP, ships them to the DSP, lets the DSP do the int8
GEMM with HVX, and rescales the int32 accumulator back to float on the AP.

```
 AP (Linux, llama.cpp)                 cDSP (Hexagon 690 V66)
 ┌───────────────────────────┐        ┌──────────────────────────────┐
 │ ggml backend scheduler     │        │ libgemm_skel.so             │
 │   MUL_MAT ─► cDSP backend  │  RPC   │   gemm_dsp.c                │
 │     quant A (per-row)      │ ─────► │     HVX vrmpy int8 GEMM     │
 │     quant B (per-col)      │        │     qurt_hvx_lock/unlock    │
 │     transpose B            │ ◄───── │   C (MxN, int32)            │
 │   C = sA * sB * C32 (f32)  │        └──────────────────────────────┘
 └───────────────────────────┘
```

The scheduler keeps **everything except `MUL_MAT`** on the CPU backend, so only
matmuls leave the AP. This is the same design llama.cpp uses to offload to GPU/NPU
backends.

## 2. FastRPC contract (gemm.idl)

```c
interface cdsp_gemm : remote_handle64 {
    long rpc_init();
    // One-shot GEMM (kept for the standalone test + fallback):
    long gemm_int8(in  sequence<int8>  A,     // M*K  activations
                   in  sequence<int8>  B,     // K*N  weights
                   in  sequence<int32>  dims, // [M, N, K]
                   rout sequence<int32> C);   // M*N  int32 accumulator
    // Weight-cache API (Stage 5): upload the (large) weight ONCE, then only
    // ship the small activation A on every subsequent token:
    long gemm_upload(in  sequence<int8>  B,     // K*N  weights
                     in  sequence<int32>  dims, // [0, N, K]
                     rout int32          handle);
    long gemm_compute_cached(in  sequence<int8>  A,     // M*K  activations
                             in  sequence<int32>  dims, // [M, N, K]
                             in  int32           handle,
                             rout sequence<int32> C);   // M*N  int32 accumulator
    long gemm_release(in int32 handle);
};
```

`QAIC` generates:

- `gen/gemm.h` — `cdsp_gemm_URI` (e.g.
  `file:///libgemm_skel.so?cdsp_gemm_skel_handle_invoke&_modver=1.0&_dom=cdsp`),
  and the **seven** dispatch entry points `cdsp_gemm_open / close / rpc_init /
  gemm_int8 / gemm_upload / gemm_compute_cached / gemm_release`.
- `gen/cdsp_gemm_skel.c` — DSP-side dispatch; it calls straight into the
  `__QAIC_HEADER(...)` functions you implement in `gemm_dsp.c`.
- `gen/cdsp_gemm_stub.c` — AP-side marshaling used by `host_test`/`host_test_cached`
  and the plugin's `cdsp_gemm_host.cpp`.

> **Critical rule:** the skel `.so` *must* be named after the `.idl` file
> (`libgemm_skel.so`) because `cdsp_gemm_URI` bakes that name in. And all the
> `__QAIC_HEADER` functions **must** be defined in `gemm_dsp.c` — if `open`/`close`
> are missing, the DSP `dlopen` fails with `0x80000406`.

> **QAIC quirk:** output scalars use `rout int32 handle` (the only accepted
> output qualifier is `in` / `rout` / `inrout`; a bare `out` is rejected).

## 3. The HVX int8 GEMM kernel (gemm_dsp.c)

### 3.1 Math
`C[m][n] = Σ_k A[m][k] · B[k][n]`, A(M×K) int8, B(K×N) int8, C(M×N) int32,
row-major.

### 3.2 HVX building blocks used

| Intrinsic | Role |
|---|---|
| `Q6_Vw_vrmpyacc_VwVbVb` | **signed** int8 × signed int8 reduce-multiply-accumulate over a group of 4 bytes. No offset/zero-point hack needed. |
| `Q6_V_vsplat_R` | word-splat: repeats a 4-byte value `{a0,a1,a2,a3}` across the 128-byte vector — builds the A operand. |
| `qurt_hvx_lock(QURT_HVX_MODE_128B)` / `qurt_hvx_unlock()` | claim/release the 128-byte HVX unit for the whole call. |

### 3.3 Layout trick (why it's fast)

A single `vrmpyacc` computes **32 partial dot products at once** (32 output
columns × 4 k-elements). To feed it, B is **prepacked** at the start of the call
into an interleaved tile layout:

```
Bp[(t*Kblk + kb)*128 + j*4 + i] = B[(k0+i)*N + (c0+j)]   (0 outside range)
```

so that one 128-byte aligned load at `(tile t, k-block kb)` yields, for group
`g = column j`:

```
{ B[k0+0][c0+g], B[k0+1][c0+g], B[k0+2][c0+g], B[k0+3][c0+g] }
```

That is exactly the `{b0,b1,b2,b3}` operand `vrmpyacc` wants. K runs in steps of 4
(`K_STEP`), N in tiles of 32 (`N_TILE`). The accumulation happens in one HVX
vector per tile, then the valid columns are spilled to the (unaligned) C buffer.

### 3.4 Lock discipline

The **compute** path (`gemm_compute_with_bp`) runs under the HVX lock. The
`-mhvx` compiler may emit HVX for any hot scalar loop; executing HVX outside the
lock faults the DSP. (If `qurt_hvx_lock` fails, the code falls back to a scalar
reference so the kernel still returns a correct result.)

> Note: the **prepack** loop is pure scalar byte shuffling and therefore runs
> *outside* the lock (in `gemm_upload`, once per weight) — it never emits HVX, so
> no lock is needed there. This is exactly why moving prepack to upload time
> removes it from the per-token critical path.

## 4. The ggml backend plugin (ggml-cdsp.cpp)

- Registers as backend **`cdsp`** of type `GGML_BACKEND_DEVICE_TYPE_ACCEL`.
- `supports_op()` returns true **only** for `GGML_OP_MUL_MAT` with F32 activation
  and F32 or Q8_0 weight, and matching contraction dim. The scheduler therefore
  never hands it anything else.
- `graph_compute()` walks the graph and calls `cdsp_mul_mat()` for each MUL_MAT
  node. Tensors live in ordinary CPU host buffers (`ggml_backend_cpu_buffer_type`)
  — FastRPC copies them to/from the DSP — so no custom buffer type is needed.
- **Weight cache (Stage 5).** On the first MUL_MAT touching a given weight tensor,
  the plugin dequantizes the weight (if Q8_0) → float, quantizes its columns →
  int8 (per-column scale `sB`), transposes it to `(K×N)`, and calls
  `cdsp_gemm_upload()` — the multi-MB weight is shipped to the DSP **once**. The
  returned handle is stored in an **LRU table keyed by `(weight_tensor_ptr, batch)`**
  with a byte capacity (`GGML_CDSP_CACHE_MB`, default 256 MB). On every
  subsequent token the plugin only quantizes the (tiny) activation A and calls
  `cdsp_gemm_compute_cached(A, handle)`; the DSP reuses the already-prepacked
  weight. On eviction (capacity exceeded) the handle is released on the DSP,
  freeing its prepacked buffer. `cdsp_backend_free()` releases all handles.
- `cdsp_gemm_host.cpp` wraps the FastRPC session: `enable_unsigned_pd()`,
  `cdsp_gemm_open(URI "&_dom=cdsp")`, and the four compute/upload/cached/release
  calls. On non-Linux it stubs out so the plugin still compiles for API
  validation.

The whole plugin is pinned to the **ggml 0.9.0** ABI via a verbatim copy of
`ggml-backend-impl.h` (byte-identical to the upstream source) and the committed
`deps/ggml-0.9.0-headers/`.

## 5. Build & deploy chains

| Stage | Build host | Output | Deploys to device | Verifies |
|---|---|---|---|---|
| 0 | Windows + Hexagon SDK | `prebuilt/libgemm_skel.so` | (committed; skip if prebuilt used) | `nm -D` 7×`T` |
| 1 | on-device `gcc` | `host_test`, `host_test_cached` | skel → `/usr/lib/rfsa/cdsp` | `host_test` + `host_test_cached` PASS |
| 2 | on-device `g++` | `libggml-cdsp.so` | → `GGML_BACKEND_DIR` | `test_cdsp` PASS |
| 2.5 | on-device `g++` | `test_sched` | (same dir) | `SCHED_TEST PASS` |
| 5 | on-device `g++` | `test_cache` | (same dir) | `CACHE_TEST PASS` + cache-hit log |
