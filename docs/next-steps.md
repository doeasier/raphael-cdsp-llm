# Next steps — making cDSP inference actually useful

Stages 0–5 are now implemented and verified on the device (correctness + the
weight-cache transport fix). This document defines what remains before real
llama.cpp inference is genuinely useful.

---

## 1. Weight-cache RPC — DONE (Stage 5)

The structural blocker is solved. The backend now uploads each weight tensor to
the DSP **once** (`gemm_upload`) and reuses it across tokens
(`gemm_compute_cached`), keyed by an LRU cache (`GGML_CDSP_CACHE_MB`, default 256
MB). Measured at real model dimensions:

| Scenario | M×N×K | Naive (re-ship B) | **Cached (ship A only)** | Upload (1×) |
|---|---|---|---|---|
| decode attention | 1×4096×4096 | 281 ms | **14 ms** | 253 ms |
| decode FFN | 1×11008×4096 | 762 ms | **37 ms** | 672 ms |

The 253/672 ms upload is a one-time, per-weight cost. After it, the per-token cost
is dominated by the matmul itself.

### 1.1 Remaining caveat: cDSP memory capacity

The cache holds the model as **int8** on the DSP. A 4B model is ~4 GB int8 — far
more than the cDSP can hold — so the 256 MB default cache will **thrash** for such
models (least-recently-used weights get released and re-uploaded within a token).
This means:

- **Small models** (that fit as int8 in DSP RAM, or fit in the cache) get the full
  win and become genuinely interactive.
- **Large models (e.g. Qwen3.5-4B)** need either (a) more DSP RAM, (b) a smarter
  staging scheme (keep packed weights on the AP and stream tiles — still a transport
  cost, but amortized per-tile, not per-layer), or (c) accept per-layer eviction
  and measure the real token/s. This is the next real research problem.

The repo's `Qwen3.5-4B-Q4_K_M.gguf` is already on the device at `/data2/model/`
(see §3 to actually run it and measure).

---

## 2. HVX peak tuning

Even the kernel's compute phase is far from the V66 HVX ceiling. Verified throughput
(128×512×512 ≈ 33.5M MAC in 7.92 ms) is ~4.2 GMAC/s; the Hexagon 690 HVX peak for
int8 is on the order of **~150–200 GMAC/s**. Gaps to close:

- **Avoid recomputing the prepack in the hot path.** The current kernel prepacks B
  inside every `gemm_int8` call. With the weight-cache RPC (§1) the prepack moves to
  upload time, removing it from the per-token critical path entirely.
- **Build the A operand cheaply.** `Q6_V_vsplat_R` from a scalar is fine, but A can
  be prepacked into the same interleaved tile format once per activation when K is
  reused across tiles.
- **Double HVX unit / dual-issue.** V66 exposes two HVX contexts; see if two
  `vrmpyacc` chains can be interleaved per instruction to double utilization.
- **Larger K unrolling / software pipelining.** Process more k-blocks per outer
  iteration to hide the prepack/load latency.
- **Consider `Q6_Vw_vrmpy_VubVb`** variants if the activation quantization can be
  kept unsigned, to widen the supported operand forms.

---

## 3. Real llama.cpp inference test

Once §1 + §2 land, run an actual model end-to-end on the device:

```bash
# on device, after the backend is installed in GGML_BACKEND_DIR
export GGML_BACKEND_DIR=/data/local/tmp/ggml-backend
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin:/usr/local/lib:$LD_LIBRARY_PATH
# make sure Qwen3.5-4B (or any GGUF) is present, e.g. at /data2/model/
llama-server -m /data2/model/Qwen3.5-4B-Q4_K_M.gguf -ngl 999 --host 0.0.0.0 -p "Hello"
# or: llama-cli -m ... -p "..." --no-warmup
```

Watch the logs for the `[ggml-cdsp] offloaded MUL_MAT ...` lines (set
`GGML_CDSP_DEBUG=1`) to confirm the scheduler is actually routing matmuls to the
cDSP, then measure tokens/s vs the pure-CPU path. Compare against the CPU baseline
(to be captured on the same build) to quantify the offload benefit.

> **Reproducing the model push** (one-time, device has no internet): the 2.55 GB
> `Qwen3.5-4B-Q4_K_M.gguf` was served from the host over HTTP and pulled by the
> device with `curl -C -` (SFTP is flaky for large files). Verify integrity with
> `md5sum` on both ends.

---

## 4. Suggested task order

1. **HVX peak tuning** (§2) — now the dominant lever; the prepack is already out of
   the hot path thanks to Stage 5. Close the ~4 → ~150 GMAC/s gap.
2. Run the **real llama.cpp** inference test (§3) with `Qwen3.5-4B` and measure
   tokens/s CPU-vs-cDSP (start with a small `GGML_CDSP_CACHE_MB` to see the thrash
   behavior, then a large one if DSP RAM allows).
3. If large-model thrash is unacceptable, design the smarter staging scheme (§1.1),
   e.g. keep packed weights on the AP and stream tiles.

---

## 5. Known limitations

- Only `GGML_OP_MUL_MAT` is offloaded; all other ops (RoPE, RMSNorm, attention
  softmax, sampling) run on the CPU. That is by design for this PoC.
- Quantization is symmetric per-vector int8 (scale = maxabs/127). No per-block
  grouping. Accuracy is fine for a PoC but not optimal; it mirrors the reference
  path used for verification.
- The weight cache holds the model as int8 on the DSP; models larger than the cache
  (and larger than DSP RAM) thrash (§1.1). Small models are fully covered.
- The scheduler integration test (`test_sched.cpp`) and cache test (`test_cache.cpp`)
  prove the offload/cache path with a tiny graph; they are not a substitute for the
  full `llama.cpp` inference test in §3.
