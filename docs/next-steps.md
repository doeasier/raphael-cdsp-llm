# Next steps — making cDSP inference actually useful

Stages 0–2.5 prove the **correctness and integration path**. They do **not** yet
make real llama.cpp inference viable, because of one structural bottleneck and one
performance gap. This document defines the roadmap.

---

## 1. The hard blocker: weight re-shipping on every call

Today the backend (in `ggml-cdsp.cpp` → `cdsp_gemm_compute`) ships **both** A and
B to the DSP on every `MUL_MAT`. For a real model that means re-transmitting the
entire weight matrix (tens of MB per layer) over FastRPC **on every token** during
decode. Measured impact at real model dimensions:

| Scenario | M×N×K | Round-trip time |
|---|---|---|
| decode attention | 1×4096×4096 | 281 ms |
| decode FFN | 1×11008×4096 | 762 ms |
| prefill attention | 32×4096×4096 | 691 ms |

Effective compute is only ~1–4 GMAC/s — far below the HVX ceiling (see §2) — and
decode is dominated by re-shipping weights, not by the matmul itself. This is
fundamentally unusable for token generation.

### Fix: weight-cache RPC (the next milestone)

Add FastRPC methods so weights are uploaded **once** and subsequent calls only ship
the small activation matrix A:

1. `gemm_upload(B_int8, dims)` — quantize + prepack B on the AP, upload to the DSP,
   store the prepacked buffer + dimensions in a **session-scoped handle**
   (`remote_handle64` or a slot table inside the skel).
2. `gemm_compute_cached(A_int8, B_handle, dims)` — only A is marshaled; the skel
   reuses the already-prepacked B and runs the same HVX kernel.
3. `gemm_release(B_handle)` — free the cached buffer.

On the ggml side, the plugin must cache the quantized+prepacked weight per
`src1` tensor (keyed by tensor pointer / an opaque id) so the upload happens once
per weight tensor (i.e. once per model load), not per `MUL_MAT` invocation. The
currently-correct `cdsp_gemm_compute(A,B,...)` path can remain as a fallback.

**Expected win:** decode cost drops from "ship weights every time" to "ship a
1×K activation" → from hundreds of ms to single-digit ms per MUL_MAT at these
sizes, making small models (and prefill) genuinely interactive.

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

1. **Weight-cache RPC** (`gemm_upload` / `gemm_compute_cached` / `gemm_release` in
   `gemm.idl` + `gemm_dsp.c`; cache handle in the plugin). *Highest leverage.*
2. Re-run Stages 1–2.5 with the cached path; add a decode-loop benchmark
   (`host_test 1 <N> <K>` many times) to measure steady-state token cost.
3. HVX peak tuning from §2 (prepack-out-of-hot-path is already solved by step 1).
4. Real `llama.cpp` inference test (§3); publish tokens/s CPU-vs-cDSP numbers.

---

## 5. Known limitations

- Only `GGML_OP_MUL_MAT` is offloaded; all other ops (RoPE, RMSNorm, attention
  softmax, sampling) run on the CPU. That is by design for this PoC.
- Quantization is symmetric per-vector int8 (scale = maxabs/127). No per-block
  grouping. Accuracy is fine for a PoC but not optimal; it mirrors the reference
  path used for verification.
- The scheduler integration test (`test_sched.cpp`) proves the offload path with a
  tiny graph; it is not a substitute for the full `llama.cpp` inference test in §3.
