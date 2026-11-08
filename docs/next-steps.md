# Next steps — making cDSP inference actually useful

Stages 0–5 implemented and verified the transport path. **Stage 6** closed
§2 HVX peak tuning and §3 real llama.cpp inference. **Stage 7** closed the
former top priority, batched RPC (§4) — with a deliberately honest negative
end-to-end result. **Stage 8** ran the small-model benchmark (§5) and
**falsified the cache-residency hypothesis**: even a model whose int8 weights
fit entirely in the cDSP cache is ~3.4× slower than the CPU. Everything below
records the measurements and what remains.

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

## 5. Small-model benchmark — DONE (Stage 8), hypothesis FALSIFIED

The previously highest-priority experiment asked: *if the int8 weights fit in
the 256 MB cache, does the cDSP finally beat the CPU?* Model used:
**`smollm2-135m-q4_k_m.gguf`** (101 MB GGUF, llama arch, 30 layers, emb 576,
FFN 1536, vocab 49152), transferred over USB-NCM via a temporary HTTP server
(see `docs/reproduce.md` §4). All runs use `--seed 42 --temp 0 --no-debug`,
64 tokens, clean device state after `systemctl reboot`.

### 5.1 Headline result: no, it does not

| Configuration | Prompt | Generation | vs CPU |
|---|---|---|---|
| **CPU only** | 123.2 t/s | **81.2 t/s** | 100% |
| cDSP, best (`MIN=500000 MAX=2000000`, FFN only) | 47.1 t/s | **23.8 t/s** | **29%** |
| cDSP, full offload (`MIN=0 MAX=0`) | 44.3 t/s | **13.4 t/s** | 17% |
| cDSP, `GGML_CDSP_BATCH=1` | 48.7 t/s | 19.0 t/s | no gain |
| cDSP, `GGML_CDSP_NO_CACHE=1` | 17.4 t/s | 1.4 t/s | 17.6× cache value |

**The cache-capacity hypothesis is falsified.** The weights really do all fit:
full offload reported `cache hits=1984 misses=31 bytes=44015616` (**42 MB**,
zero thrash) — exactly the scenario Stage 6 predicted would win. It is still
3.4× slower than four CPU threads. Caching is necessary but nowhere near
sufficient.

### 5.2 Where the time actually goes

`GGML_CDSP_SPLIT_STATS=1` on this model: **2015 splits / 2015 MUL_MAT nodes /
0 splits with >1 matmul**. Unlike Qwen3.5-4B (37.5% multi-matmul splits), the
135M graph is a strictly serial chain of CPU-only ops (RMSNorm, RoPE, softmax,
SiLU) between matmuls, so **Stage 7 batching has nothing to merge here**.
Offload shapes: 1008× `1×576×1536`, 882× `1×192×576`, 65× `1×49152×576`.

`GGML_CDSP_PROFILE=1` splits each offloaded MUL_MAT into four host-side stages
(1040 ops, best config):

| Stage | Total | Share | Per op |
|---|---|---|---|
| weight / cache (incl. 31 cold uploads) | 709.7 ms | 36.5% | — |
| activation quantization | 62.3 ms | 3.2% | 0.060 ms |
| **FastRPC + DSP GEMM** | **1157.1 ms** | **59.5%** | **1.113 ms** |
| output rescale | 16.4 ms | 0.8% | 0.016 ms |
| **total inside `cdsp_mul_mat`** | **1945.6 ms** | 100% | 1.871 ms |

Steady state (excluding the one-off uploads) is ~1.19 ms/op, or ~19.3 ms of the
43.3 ms/token the run actually took. The remaining **~24 ms/token is outside
the backend**: ggml's multi-backend scheduler must switch CPU↔ACCEL for every
split. Loading the backend with zero offloads (`MIN_ELEMS=100000000`) still
gives 71.0 t/s vs CPU 80.6 t/s, so the overhead is not backend registration —
it is the per-split synchronization that only happens when ops are offloaded.
This also explains why saving round-trips inside a split (Stage 7) barely
moved tokens/s: the expensive unit is the **split boundary**, not the RPC.

### 5.3 Persistent zero-copy buffers — investigated, NOT viable here

`src/host/bench_regbuf.c` times the same `gemm_compute_cached` with three A/C
buffer strategies (`1 576 1536 120`):

| Strategy | ms/call | Result |
|---|---|---|
| plain `malloc` (current backend) | **0.937–1.017** | fastest |
| `rpcmem_alloc_def` (DMA-BUF) | 1.033–1.280 | **1.6–36.6% slower**, noisy |
| RPCMEM + `fastrpc_mmap(FASTRPC_MAP_STATIC)` | — | **unsupported** |

`fastrpc_mmap()` returns `20 = 0x14 = AEE_EUNSUPPORTED` and the kernel logs
`qcom,fastrpc-cb ...: mem mmap error`. This device's fastrpc driver has no
persistent mapping path. **Do not retry `remote_register_buf` / `fastrpc_mmap`
on this kernel**; it is a driver capability gap, not an argument error. The
benchmark is kept in the regression (`REGBUF_TEST PASS`) so the negative result
stays reproducible.

### 5.4 Numerical correctness on a small model

`GGML_CDSP_NO_CACHE=1` produces **the same text as the normal cached path**, so
stale cache entries (a worry from §3) are ruled out; the divergence from CPU is
pure int8 quantization error. Under greedy decoding CPU and cDSP agree for the
first ~10 tokens (`"The first time I saw the picture of the"`) and then branch;
with `--temp 0.7` both continuations are fluent but different. A 135M model has
much narrower logit margins than a 4B model, so the ~0.4–0.7% per-layer error
can flip an argmax. **The backend is not bit-exact on small models**, although
output remains coherent. Per-block/per-K-group quantization would address this,
but would not change the speed conclusion.

### 5.5 Conclusion

What decides the winner is not cache residency but **how much compute amortizes
each cross-backend boundary**. A 135M decode needs ~16–30 small GEMMs per token;
each pays ~1.1 ms FastRPC plus ~0.7 ms scheduler synchronization, while four
CPU threads finish the whole token in ~13 ms. The same structural reason makes
the 4B model lose (there it is cache thrash plus the CPU-side large FFN). For
both model scales tested, single-operator `MUL_MAT` offload to an ACCEL backend
is the wrong granularity.

---

## 6. What remains (in priority order)

1. **Remove split boundaries, not RPCs.** The only lever that touches the
   ~24 ms/token scheduler overhead is getting several consecutive matmuls into
   one ACCEL split — e.g. implementing RMSNorm/RoPE/SiLU/softmax as DSP kernels,
   or fusing a whole transformer layer into one FastRPC call. This is a much
   larger change than a backend plugin and is the honest "next project".
2. **Weight staging / TCM residency for large models** (§1.1): keep packed
   weights on the AP, stream tiles to the DSP, amortize per tile rather than
   per layer. Needed before 4B FFN matrices can be offloaded.
3. **Better quantization** (per-block / per-K group) to fix the §5.4 divergence
   on narrow-margin models. Correctness, not throughput.
4. **Fuse the output rescale on the DSP.** Profile says this can recover at most
   0.8% of backend time — worth doing only as part of a larger DSP-side fusion.
5. **Change the decode dataflow** (TCM-resident weights, cross-token batching):
   M=1 decode remains 1 byte/MAC memory-bound, see §2.2.

> **Scope note.** Stages 6–8 establish a consistent, measured negative result
> for *single-operator offload* at 135M and 4B model scales. Further tuning of
> the current backend is unlikely to beat the CPU; the remaining items above
> are architectural, not incremental.

---

## 7. Known limitations

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
- Persistent FastRPC mappings are unavailable on this kernel
  (`fastrpc_mmap` → `AEE_EUNSUPPORTED`, dmesg `mem mmap error`); plain
  RPCMEM buffers are slower than `malloc`. See §5.3.
- On small models the int8 path is **not bit-exact**: greedy decoding diverges
  from CPU after ~10 tokens (§5.4).
- Every offloaded `MUL_MAT` creates a CPU↔ACCEL scheduler split. On SmolLM2-135M
  this costs ~24 ms/token outside the backend and dominates the outcome (§5.2).
