# Next steps — making cDSP inference actually useful

Stages 0–5 implemented and verified the transport path. **Stage 6 (this update)
closed both open items**: §2 HVX peak tuning and §3 real llama.cpp inference.
Everything below records the measured results and what remains.

---

## 1. Weight-cache RPC — DONE (Stage 5)

The structural blocker is solved. The backend uploads each weight tensor to the
DSP **once** (`gemm_upload`) and reuses it across tokens
(`gemm_compute_cached`), keyed by an LRU cache (`GGML_CDSP_CACHE_MB`, default
256 MB). Measured at real model dimensions:

| Scenario | M×N×K | Naive (re-ship B) | **Cached (ship A only)** | Upload (1×) |
|---|---|---|---|---|
| decode attention | 1×4096×4096 | 281 ms | **14 ms** | 253 ms |
| decode FFN | 1×11008×4096 | 762 ms | **37 ms** | 672 ms |

### 1.1 cDSP memory capacity — confirmed, now quantified

The cache holds the model as **int8** on the DSP. A 4B model is ~4 GB int8, far
more than the cDSP can hold. Measured on `Qwen3.5-4B-Q4_K_M.gguf`:

- With `GGML_CDSP_MAX_ELEMS=3000000`, 59 unique weight matrices stay cached
  (~32 MB prepacked), giving ~97% hit rate and no thrash.
- With `MAX_ELEMS=0` (full offload) the cache fills to 256 MB and thrashes;
  the run becomes effectively hung and was killed. **Do not use full offload
  on a 4B model.**
- Conversely, 74% of the offloaded ops in that selective run were the tiny
  `1×32×2560` shape (82k MACs each). Each still pays a ~0.25 ms FastRPC
  round-trip. `GGML_CDSP_MIN_ELEMS=100000` leaves those on the CPU and recovers
  ~20 percentage points of throughput.

So the practical policy for large models is a **band-pass** on matrix size:
`MIN_ELEMS` avoids RPC-dominated fragments, `MAX_ELEMS` avoids cache thrash.
See `docs/llama-integration.md` §3–4.

---

## 2. HVX peak tuning — DONE (Stage 6)

Measured with `src/host/bench_cached.c`, shipped and run by
`scripts/device/deploy_cdsp_gemm.py`. A warm-up pass is essential: the cDSP
ramps its clock under sustained load and cold-clock numbers are 30–50% low.

Numbers below are the **final clean run** recorded in
`evidence/stage6-dsp-regression.txt` (200-iteration warm-up first, then the
suite). The "before" column was captured with the same `bench_cached` driver
before the warm-up stage existed, so its **early small shapes are cold-clock
pessimistic**; the two large shapes (which had already warmed the DSP) are
directly comparable and are the strongest evidence.

| Shape | Before (ms/call) | Before (GMAC/s) | **Stage 6 (ms/call)** | **Stage 6 (GMAC/s)** | Speedup |
|---|---|---|---|---|---|
| 1×32×4 (FastRPC ping) | 0.315 | — | **0.340** | — | — |
| 1×32×2560 | 0.352 | 0.47 | **0.348** | **0.47** | ~1× |
| 1×1024×2560 (decode attn) | 2.654 | 1.98 | **1.743** | **3.01** | 1.5× |
| 1×4096×4096 (decode FFN) | 14.84 | 2.26 | **9.204** | **3.65** | 1.6× |
| 128×512×512 (reference) | 4.953 | 13.55 | **2.188** | **30.67** (32.19 warmed) | 2.3× |
| 32×4096×4096 (prompt) | 421.24 | 2.55 | **94.28** | **11.39** | 4.5× |

Against the **original one-shot `gemm_int8` figure of 4.2 GMAC/s** quoted in
the previous revision (which included prepack in the hot path), the reference
shape is now **~32 GMAC/s — a 7.6× improvement**.

### 2.1 What was changed in `src/dsp/gemm_dsp.c`

The kernel now auto-selects one of three paths:

| Condition | Path | Rationale |
|---|---|---|
| `M == 1` | `gemm_m1()` | decode: 4-way K unroll, 4 independent accumulators to keep four 128 B Bp loads in flight |
| `M > 1`, Bp ≤ 512 KB | `gemm_small_bp()` | L2-resident weights: single accumulator, m-major loop, minimal register pressure |
| `M > 1`, Bp > 512 KB | M-blocked loop (`M_BLOCK=4`, `K_UNROLL=2`) | amortizes each Bp vector load across 4 rows; prompt batch is 4.5× faster |

Supporting changes:

- **A hoisted out of the innermost loop.** The original re-packed four A bytes
  for every `(m, tile, kb)`. One `uint32` per `(row, k-block)` is now packed
  once per row block and reused across all N tiles.
- **Persistent scratch buffer.** `g_aw` is grown on demand per PD and freed in
  `cdsp_gemm_close()`; a `malloc/free` inside the HVX lock measurably hurt the
  small decode shapes.
- **M blocking** removes up to `M_BLOCK×` redundant weight traffic for prompt
  batches (the dominant win: 421 → 94 ms).
- **K unroll** breaks the `vrmpyacc` in-place dependency chain. It is applied
  only to the M=1 path: on V66, extra accumulator planes in the M>1 path
  caused register spills and made the square shape slower (measured).

All nine `host_test` / `host_test_cached` cases still report `PASS`
(`evidence/stage6-dsp-regression.txt`), so the three paths are numerically
identical to the scalar reference.

### 2.2 Why ~150 GMAC/s is not reachable for decode

The target in the previous revision was the Hexagon 690 int8 *arithmetic*
ceiling. For `M=1` decode this is a **roofline** problem, not an instruction
issue: each 128-byte Bp load feeds exactly 128 MACs and is never reused, so
int8 decode is fundamentally **1 byte/MAC memory-bound**. Hitting 150 GMAC/s
would require ~150 GB/s of sustained weight bandwidth from DDR, which this DSP
cannot provide; the measured ~3.65 GMAC/s decode corresponds to ~3.65 GB/s.

High GMAC/s is only attainable when Bp is L2/TCM-resident and reused across M
rows — hence the ~32 GMAC/s square result. Further gains for real LLM decode
require changing the dataflow (keep packed weights in TCM, stream tiles from
the AP, or batch several MUL_MATs into one FastRPC call), not more HVX
unrolling.

---

## 3. Real llama.cpp inference — DONE (Stage 6)

Fully documented in **`docs/llama-integration.md`** (integration recipe, why an
`LD_PRELOAD` shim is required with `GGML_BACKEND_DL=OFF`, all environment
variables, troubleshooting).

Key results on `Qwen3.5-4B-Q4_K_M.gguf`, 64 tokens, same-window comparison:

| Configuration | Prompt | Generation | Output |
|---|---|---|---|
| CPU only | 5.5–6.4 t/s | 3.6–4.1 t/s | coherent, `Capital: Paris` |
| cDSP, pre-Stage-6 | 4.9 t/s | 3.0 t/s | coherent (after Q4_K/Q8_0 fix) |
| cDSP, Stage 6 recommended | 5.3–6.0 t/s | 3.4–3.8 t/s | coherent, `Capital: Paris` |

Three correctness bugs were found and fixed while closing §3:

1. **`MUL_MAT` src0/src1 were reversed.** ggml uses
   `ggml_mul_mat(src0=weight [K,N], src1=activation [K,M])`; the backend had
   assumed the opposite.
2. **Q4_K block layout**: ggml 0.9.0 uses `scales[12]` / 144 B block
   (`K_SCALE_SIZE=12`, `QK_K=256`), not `scales[8]` / 140 B. The wrong layout
   produced Cyrillic gibberish.
3. **Q8_0 block layout**: ggml 0.9.0 uses `[fp16 d][int8 qs[32]]` = **34 B**,
   not a f32 scale / 36 B. `dequantize_row_q8_0()` uses
   `GGML_FP16_TO_FP32(x[i].d)`.

`src/ggml-backend/test_q4k.cpp` now builds Q4_K and Q8_0 weights with ggml's
own `ggml_quantize_chunk()` and compares CPU vs cDSP (0.42% / 0.69% relative
error). `GGML_CDSP_HOST_F32=1` was used to prove the dequant/layout/scheduler
path was clean before the DSP path was rebuilt — a stale device `.so` had been
masking the Q8_0 fix.

---

## 4. What remains (in priority order)

1. **Batch RPC calls.** One FastRPC round-trip per `MUL_MAT` is now the dominant
   cost (~50–60 calls/token). A batched `gemm_compute_multi` RPC (or a scheduler
   hook that groups small MUL_MATs) is the highest-leverage next step.
2. **Weight staging for large models** (§1.1): keep packed weights on the AP,
   stream tiles to the DSP, amortize per-tile rather than per-layer. Needed
   before FFN matrices can be offloaded for a 4B model.
3. **Fuse the output rescale.** The host still multiplies each result by
   `sA[m] * sB[n]`; doing this on the DSP with a fixed-point scale would remove
   one host pass per op.
4. **Better quantization** than symmetric per-vector int8 (per-block / per-K
   group) to reduce the ~0.4–0.7% per-layer error.
5. **Small-model benchmark.** A model whose int8 weights fit the cache is where
   this backend should actually beat the CPU; that measurement is still open.

---

## 5. Known limitations

- Only `GGML_OP_MUL_MAT` is offloaded; RoPE, RMSNorm, softmax and sampling stay
  on the CPU. By design for this PoC.
- Weight tensors must be contiguous F32 / Q8_0 / Q4_K; `K` must be a multiple
  of 32 (Q8_0) or 256 (Q4_K). All guards live in `cdsp_can_mul_mat()`, shared
  by `supports_op()` and `cdsp_mul_mat()`.
- The weight cache is keyed by `(data pointer, batch index)`. It is safe for
  static model weights. Dynamic tensors whose buffer is reused with changed
  contents would need a version/content guard; `GGML_CDSP_NO_CACHE=1` exists to
  detect this class of problem and was used during §3 debugging.
- Quantization is symmetric per-vector int8 (scale = maxabs/127), no per-block
  grouping.
- The cDSP kernel cannot hold a 4B model; large models need the band-pass
  `MIN_ELEMS` / `MAX_ELEMS` policy. Full offload thrashes and hangs.
- `ggml_backend_load_all()` is bypassed via `LD_PRELOAD` because this device's
  llama.cpp has `GGML_BACKEND_DL=OFF`; a build with `GGML_BACKEND_DL=ON` can
  use `GGML_BACKEND_PATH` directly.
