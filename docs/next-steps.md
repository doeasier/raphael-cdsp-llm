# Next steps — making cDSP inference actually useful

Stages 0–5 implemented and verified the transport path. **Stage 6** closed
§2 HVX peak tuning and §3 real llama.cpp inference. **Stage 7** closed the
former top priority, batched RPC (§4) — with a deliberately honest negative
end-to-end result. Everything below records the measurements and what remains.

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

## 4. Batched RPC — DONE (Stage 7), but not an end-to-end win

`GGML_CDSP_SPLIT_STATS=1` showed why batching looked promising: on a 64-token
run, 528 scheduler splits contained 726 `MUL_MAT` nodes and **198 splits
(37.5%) held two independent matmuls**. Implementation (`GGML_CDSP_BATCH=1`):

- `cdsp_graph_compute()` collects the split's `MUL_MAT`s, checks that none
  consumes another's output (views resolved to their parent), and prepares all
  of them — dequant, weight-cache lookup, activation quantization — before
  issuing a single FastRPC call.
- `dims = { -count, M0,N0,K0,h0, ... }` with `A = concat(A_i)` and
  `C = concat(C_i)`. The negative marker reuses the existing QAIC method, so
  **no IDL / stub regeneration** was required; only `gemm_dsp.c`, the host
  wrapper and the backend changed.
- `src/host/host_test_batch.c` guards correctness with two *different* weight
  shapes (1056 elements, `BATCH_TEST PASS`). With `--seed 42 --temp 0`,
  batch-on and batch-off generate **identical text**.

| Level | Scalar | Batched | Delta |
|---|---|---|---|
| RPC pair latency (`host_test_batch 30`) | 2.505 ms | **1.962 ms** | **−21.6%** |
| 64-token llama-cli (`MIN=100000`) | 3.3–3.5 t/s | 3.2–3.4 t/s | none measurable |

The honest result: **saving 78–198 round-trips does not move tokens/s.**
Consecutive deterministic A/B runs slow monotonically (3.5 → 3.4 → 3.3 →
3.2 t/s) under thermal throttling; reversing the order (ABBA: on 3.4, off 3.2)
shows whichever mode runs first wins ~0.1–0.2 t/s. The theoretical ceiling is
~0.8 ms/pair × ~3 pairs/token ≈ 2.4 ms against ~280 ms/token (<1%), far inside
that noise. This is consistent with §2.2: decode is **weight-bandwidth bound**
and the CPU still computes the large FFN matrices. Batch mode is therefore
kept **opt-in** and the recommended config is unchanged.

A useful corollary: letting tiny `1×32×2560` ops piggyback (`MIN=0` + batch)
gives 3.1 t/s vs 2.9 t/s scalar, still below the 3.3 t/s of leaving them on
the CPU — their DSP compute + quantization cost more than the RPC saved.

Full measurements and the ordering-bias analysis: `docs/llama-integration.md`
§4.1, `evidence/stage7-*.txt`, `evidence/stage7-ab/`.

---

## 5. What remains (in priority order)

1. **Small-model benchmark.** A model whose int8 weights fit the 256 MB cache
   is where this backend should actually beat the CPU; that measurement is
   still open and is now the highest-value experiment. The device currently
   only has `Qwen3.5-4B-Q4_K_M.gguf` (2.74 GB).
2. **Weight staging for large models** (§1.1): keep packed weights on the AP,
   stream tiles to the DSP, amortize per-tile rather than per-layer. Needed
   before FFN matrices can be offloaded for a 4B model.
3. **Change the decode dataflow.** More HVX unrolling cannot help: M=1 decode
   is 1 byte/MAC memory-bound. Keeping packed weights in TCM or batching
   *across tokens* are the remaining structural options.
4. **Fuse the output rescale.** The host still multiplies each result by
   `sA[m] * sB[n]`; doing this on the DSP with a fixed-point scale would remove
   one host pass per op.
5. **Better quantization** than symmetric per-vector int8 (per-block / per-K
   group) to reduce the ~0.4–0.7% per-layer error.

---

## 6. Known limitations

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
- Batched RPC only merges splits whose `MUL_MAT`s are mutually independent and
  single-batch (`B0 == 1`, `B1 <= 1`), at most 8 ops; everything else falls
  back to the scalar path. It is opt-in because it does not improve tokens/s.
- Sustained back-to-back llama-cli runs can trip the cDSP watchdog even with
  clean shutdowns (`SFR Init: wdog or kernel error suspected`). Only
  `systemctl reboot` clears a `crashed` remoteproc; sysfs stop/start does not.
