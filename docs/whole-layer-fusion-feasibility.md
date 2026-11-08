# Stage 9 — Whole-layer DSP fusion: feasibility assessment

> **Assessment only. Not implemented.** Task #74. No source file was changed in
> `src/`, no skel/`.so` was rebuilt, no `llama-cli` run was performed, no device
> state (firmware, services, DTB) was touched and no reboot was issued. All
> conclusions come from reading the unmodified llama.cpp 0.9.0 sources on the
> device plus the already-recorded Stage 8 measurements.
>
> Source excerpts with exact line numbers:
> [`evidence/stage9-scheduler-excerpts.txt`](../evidence/stage9-scheduler-excerpts.txt).

---

## 0. Verdict

**Do not continue toward "beat the CPU at LLM decode" through whole-layer
fusion on this device.**

| Question | Answer |
|---|---|
| Is the mechanism feasible? | **Yes, and easier than expected.** ggml 0.9.0 would merge the entire graph into one ACCEL split if `cdsp_dev_supports_op()` covered the whole op chain. No scheduler patch is required. |
| Would it reach the goal? | **No.** Even if every split boundary were free, the arithmetic ceiling is ~**51.8 t/s** against CPU **81.2 t/s** (64%). Beating the CPU additionally needs the DSP GEMM itself ~1.6× faster, which contradicts the `M=1` roofline measured in §2.2 of `next-steps.md`. |
| What would it cost? | Re-implement ~**46,600 lines** of upstream-equivalent host + DSP code (40+ kernels, a new IDL, persistent DSP-side KV/state, TCM management), on a **V66** DSP that has no HMX, no QNN/HTP runtime and no upstream skel to port. |
| Upstream already solved it? | Yes — llama.cpp 0.9.0 ships `ggml-hexagon` with `htp-fusion`, `htp-repack`, `rope/softmax/glu/flash-attn` kernels. It builds **v73–v85 only**; this SoC is **V66**. Not portable. |
| Recommendation | Freeze the backend as a **verified, reproducible negative result**. Redirect to compute-bound workloads (prefill/batch), power/CPU-offload metrics, or newer hardware. See §7. |

The PoC has already answered its research question cleanly and consistently at
two model scales (135M and 4B). The remaining items are architectural, not
incremental — and the architectural path is blocked by the SoC generation.

---

## 1. Why this assessment was opened

Stage 8 (`docs/next-steps.md` §5) produced the decisive profile on
`SmolLM2-135M-Q4_K_M`:

| Quantity | Value |
|---|---|
| CPU only | **81.2 t/s** = 12.3 ms/token |
| cDSP, best config (`MIN=500000 MAX=2000000`) | **23.8 t/s** = 42.0 ms/token (43.3 ms in the PROFILE run) |
| Time **inside** `cdsp_mul_mat` (steady state) | **19.3 ms/token** |
| Time **outside** the backend | **~24 ms/token** |
| Scheduler splits | **2,015 splits / 2,015 MUL_MAT / 0 splits with >1 matmul** |

Stage 7 proved that removing FastRPC round-trips *inside* a split does nothing
measurable; Stage 8 proved that persistent zero-copy mappings are unavailable on
this kernel (§5.3) and that full weight-cache residency is not the blocker
(§5.1). The only remaining lever is the CPU↔ACCEL split boundary. This document
establishes (a) exactly what removing it requires and (b) whether it could pay
off.

---

## 2. How ggml 0.9.0 actually forms splits

Reference: `/root/llama.cpp-0.9.0/ggml/src/ggml-backend.cpp` (2,897 lines),
function `ggml_backend_sched_split_graph()` at **L1474**. Excerpts E1–E6 in
`evidence/stage9-scheduler-excerpts.txt`.

1. **Pass 1 (L1495–1530)** calls
   `ggml_backend_sched_backend_id_from_cur()` (L1329). For any node with a
   `WEIGHTS` source it returns **L1387** the first backend satisfying
   `supports_buft(buf) && supports_op(op)` (L1296–1316). Our cDSP is backend 0
   and accepts all host buffers, so `MUL_MAT` is pinned to cDSP here, while
   `RMS_NORM` / `SILU` / `SOFT_MAX` / `ADD` (whose `supports_op` is false) are
   pinned to the **last** backend (CPU). This assignment is final: pass 2 only
   fills nodes still `-1`.
2. **Pass 2 (L1532–1610)** expands GPU (non-last) assignments up/down via
   `ggml_backend_sched_set_if_supported()` (L1466), which is again gated on
   `supports_op()`. CPU is explicitly excluded as an expansion source (L1547,
   L1568).
3. **Pass 3 (L1612–1671)** may "upgrade" a CPU node to a higher-priority
   backend when `sched->bufts[b] == sched->bufts[*node_backend_id]`. Our
   `cdsp_dev_get_buffer_type()` returns `ggml_backend_cpu_buffer_type()`
   (`src/ggml-backend/ggml-cdsp.cpp:859`), so **this upgrade path is already
   live** — it just never fires because `supports_op()` rejects the op.
4. **Pass 5 (L1705–1809)** is the boundary: **L1752**
   `if (node_backend_id != cur_backend_id || need_new_split)` starts a new
   split. There is **no cost model, no fusion heuristic, no minimum split
   size** — adjacency of backend IDs is the only merge rule.
5. **Cross-split copies (L1783–1804)** are inserted only when
   `!buffer_supported(sched, src, cur_backend_id)`. Because cDSP declares
   `supports_buft = ggml_backend_buft_is_host()` (L873) and both backends use
   the CPU buffer type, **no tensor copy is inserted on this backend**. The
   ~24 ms/token is *not* DMA/weight copying.
6. **Execution (`compute_splits()`, L2125–2204)** calls
   `ggml_backend_graph_compute_async()` once per split (L2160) and synchronizes
   the previous backend (L2142). Our device declares `async=false`,
   `events=false`, `mmap_support=false` (L850–856) and
   `cdsp_backend_synchronize()` is a no-op because `cdsp_graph_compute()` is
   fully synchronous. Every split is therefore a serial
   `CPU graph → FastRPC round trip → CPU graph` handoff.

### Consequence (the important one)

> To merge consecutive ops into one ACCEL split, **no ggml source patch is
> needed**. It is sufficient to (a) widen `cdsp_dev_supports_op()` to the whole
> op chain and (b) actually execute those ops in `cdsp_graph_compute()`.

But (b) is not optional: `cdsp_graph_compute()` currently **skips** every
non-`MUL_MAT` node (L792–800). Widening `supports_op()` alone would leave their
`dst` tensors unwritten and produce silent numerical corruption, not a crash.
The entire compute path would have to be rewritten.

---

## 3. What "one fused layer" actually requires

### 3.1 Operator inventory

From `/root/llama.cpp-0.9.0/src/llama-graph.cpp` (E7). A llama transformer
layer uses, beyond `MUL_MAT`:

`RMS_NORM`, `ROPE` (via `ggml_rope_ext` in the arch-specific model files, e.g.
`models/llama.cpp:146,152` — 293 call sites under `models/`), `SOFT_MAX_EXT`
(masked, with scale/alibi/sinks), `SILU` / `SWIGLU` / `GEGLU` variants
(several), `ADD` (residual), `MUL`, `SCALE`, `DIV`, `CLAMP`, `CAST`, `CONT`,
`GET_ROWS` / `SET_ROWS` (KV cache), `CPY`, plus view / reshape / permute /
transpose / pad. Views are skipped by the split loop (L1542) so they can stay
host-side, but their strides still have to be resolved by any DSP-side consumer.

Upstream already attacks split count at the graph level:
`llama-graph.cpp:3280-3285` deliberately adds `q_cur`/`v_cur`/`k_cur` together
"so that they are not reordered ... the number of splits in the graph is
reduced", and enables "rope fusion which directly writes into k-v cache". A
DSP-fused layer must therefore reproduce **in-place KV-cache write semantics**,
not merely standalone RoPE math — which is exactly the persistent DSP state our
stateless IDL does not have (§3.3).

### 3.2 DSP kernel work

| Group | Work on Hexagon 690 (V66, HVX) | Risk |
|---|---|---|
| `RMS_NORM` | sum-of-squares reduction, `rsqrt`, broadcast scale; must match ggml's `eps` and fp32 accumulation | HVX V66 fp32 vector capability unverified; fixed-point approximation changes numerics |
| `ROPE` | sin/cos table or polynomial, complex rotation, normal vs NeoX layout, freq/scale/ext factors | several ggml variants; tables need DSP-side residency |
| `SILU` / `SWIGLU` / `GEGLU` | `exp`/`sigmoid` LUT or polynomial, clamp/split variants | LUT precision affects the whole downstream layer |
| `SOFT_MAX_EXT` | streaming max/sum, mask, sinks, Alibi | needs KV/QK data on the DSP, not just a matmul result |
| `ADD/MUL/SCALE/CAST/CPY` | ggml broadcasting + non-contiguous stride semantics | easy to get subtly wrong, hard to debug without tracing |
| **KV cache** | `GET_ROWS`/`SET_ROWS` or in-place KV update **persistent across tokens** | new DSP-side lifetime/state management; current IDL is stateless per call |
| **Quantization boundaries** | our GEMM emits `int32` and the host rescales; a fused chain needs int8↔fp32 conversion at every step, or fp32 DSP kernels | V66 has no HMX; fp32 HVX is far slower than the int8 `vrmpyacc` path |

Correctness is already marginal: Stage 8 §5.4 shows greedy decoding diverging
from CPU after ~10 tokens on the 135M model from ~0.4–0.7% per-layer int8
error. Fusing six GEMMs per layer without intermediate requantisation would
amplify this; per-block / per-K-group quantisation is a separate project.

### 3.3 QAIC / IDL work

Current `src/dsp/gemm.idl` has 4 methods, all `sequence<int8>` /
`sequence<int32>`, one `C` output and a weight-handle table. A fused-layer call
must instead carry activation, position, rope tables, attention mask, norm
weights and a KV slice, and return the updated KV plus the layer output — i.e.
`inout` buffers, multiple `rout` sequences, session state keyed by layer and
token, a DSP-side allocator and error/rollback. That means regenerating
`src/gen/*` (1,266 lines), rewriting the host wrapper and replacing
`cdsp_graph_compute()` wholesale.

Memory headroom is unproven: Stage 8's full-offload run already held **42 MB**
of prepacked weights on the DSP (`hits=1984 misses=31 bytes=44015616`) with
zero thrash. KV cache, LUTs and intermediate activations would come on top, on
a V66 cDSP with no HMX/TCM fast path. Persistent `fastrpc_mmap` is unavailable
(§5.3), so all of it must live in ordinary DSP heap.

### 3.4 Effort benchmark against upstream

llama.cpp 0.9.0 already contains the exact architecture being considered:
`ggml/src/ggml-hexagon/` (E8).

| | This PoC | Upstream `ggml-hexagon` (fused) |
|---|---|---|
| Host side | ~1,600 hand-written lines (`ggml-cdsp.cpp` 946 + host wrapper 122 + DSP 516) | **12,785 lines** (`*.cpp` + `*.h`) |
| DSP side | 516 lines, **1** kernel | **33,800 lines**, 40+ `*-ops.c` files |
| Fusion rules | none (Stage 7 merges independent matmuls only) | `htp-fusion.cpp` 649 + `htp-kparams.cpp` 5,287 + `htp-repack.cpp` 1,636 |
| Ops | `MUL_MAT` only | matmul, rope, softmax, glu, flash-attn, unary, binary, cpy, set/get-rows, … |
| Build deps | QAIC + FastRPC + Hexagon SDK | `HEXAGON_SDK_ROOT` + `HEXAGON_TOOLS_ROOT` + HTP skel signing + QNN runtime |
| Target arch | **v66** | **v73 / v75 / v79 / v81 / v85** |

`ggml-hexagon/CMakeLists.txt:93-97` builds skels for v73–v85 only;
`ggml-hexagon.cpp:3430-3434` explicitly caps any lower arch at v73; `htp/`
compiles `hmx-queue.c` and `opt_nhmx = 1` (L87). Our SoC is Hexagon 690 = **V66**
(`docs/05` §1; `GET_DSP_INFO attr=6 → 0x00018466`), HMX/HTP skels start at V68,
and the device build has `GGML_HEXAGON:BOOL=OFF`. **There is no ready-made fused
implementation to port to this chip**, and enabling it would need an SDK and
signed skels that the offline device does not have.

---

## 4. The ceiling, even if fusion fully succeeds

Using Stage 8's best-config numbers (§5.1/§5.2 of `next-steps.md`):

```
CPU                       81.2 t/s  = 12.3 ms/token
cDSP best                 23.8 t/s  = 42.0 ms/token
  inside backend                    19.3 ms/token
    FastRPC + DSP GEMM              18.1 ms/token   (1.113 ms x 16.25 ops)
    quant / rescale / cache          ~1.2 ms/token
  outside backend                    ~24 ms/token   <-- split boundary + CPU graph
```

If whole-layer fusion removed **100%** of the ~24 ms:

```
19.3 ms/token  ->  51.8 t/s  =  64% of CPU
```

To merely *tie* the CPU, the backend must fall from 19.3 to ≤12.3 ms/token
(**−36%**), i.e. per GEMM from **1.113 ms to ≤0.68 ms (~1.6×)**, with the
remaining ~1.2 ms of quant/rescale unchanged. §2.2 already established why:
`M=1` decode is **1 byte/MAC memory-bound** — every 128-byte weight load feeds
exactly 128 MACs and is never reused. Measured decode throughput is ~3.65
GMAC/s ≈ 3.65 GB/s of weight traffic. A 1.6× gain requires keeping packed
weights in TCM and streaming tiles from the AP — that is item 2 in
`next-steps.md` §6, a **second architectural project**, on a V66 DSP whose
persistent-mapping path is already proven unavailable (§5.3).

Two further corrections make 51.8 t/s an **optimistic upper bound**:

- RMSNorm/RoPE/SiLU/softmax currently run on four CPU threads and are not
  free; fusion moves them onto the DSP's serial path.
- Fusing the output rescale and some activation quantisation into the DSP
  recovers at most the 0.8% + 3.2% from the PROFILE table (~0.8 ms/token), far
  short of the 7 ms needed.

So the honest statement is: **fusion alone cannot win; fusion plus TCM weight
streaming might, at roughly double the engineering, with no guarantee on V66.**

---

## 5. Device-specific risks

- **V66**: no HMX, no QNN/HTP, no upstream skel, no float HVX fast path proven.
- **Debug surface**: no `strace`, no device internet, no persistent FastRPC
  mapping, QAIC regeneration only on the Windows host. A fused DSP kernel that
  produces wrong numbers has almost no observability (Stage 8 §5.4 already
  needed `HOST_F32` / `NO_CACHE` A/B to isolate a 0.4% error).
- **Stability**: 8+ consecutive long runs trip the cDSP watchdog
  (`SFR Init: wdog or kernel error suspected` → `remoteproc1: crashed`); only
  `systemctl reboot` clears it, **never kexec**. A long kernel-development loop
  will hit this repeatedly.
- **Correctness**: int8 path is not bit-exact on narrow-margin models.

---

## 6. Verdict

**Close the performance thread.** Four independent reasons:

1. The measured ceiling (64% of CPU) is below the goal even under a perfect,
   cost-free fusion.
2. Reaching parity needs a second architectural change (TCM weight streaming)
   that is itself unproven on this kernel/DSP combination.
3. The engineering volume is two orders of magnitude beyond the PoC
   (~46.6k upstream-equivalent lines), with no library to reuse on V66.
4. Stages 6–8 already give a consistent, reproducible negative result at both
   135M and 4B scales; further single-operator or RPC-level tuning was proven
   twice not to move tokens/s.

What is worth preserving is the **result**, not a bigger implementation: the
backend, IDL, QAIC flow, `LD_PRELOAD` integration, weight cache, three-path HVX
kernel, batching, profiling instrumentation and all evidence are complete and
pushed. That is a solid research artefact.

### If the split-boundary hypothesis should still be tested

There is a **zero-DSP-cost** experiment that settles how much of the ~24 ms is
really scheduler overhead, without writing any kernel or regenerating QAIC
output: temporarily widen `cdsp_dev_supports_op()` to a few cheap CPU-style ops
(`ADD`, `MUL`, `RMS_NORM`) and implement them **on the host inside
`cdsp_graph_compute()`** (scalar ggml-equivalent math on `dst->data`). The
scheduler will then merge those nodes into the cDSP split, so the run measures
"fewer split boundaries with identical arithmetic". If tokens/s does not move,
the ~24 ms is CPU graph/setup cost that fusion cannot remove either, and the
case is closed with no DSP work. This is a suggestion only — **not performed in
Stage 9** — and any such patch must keep `supports_op()` and `graph_compute()`
in exact agreement (see the `graph_compute -1` failure documented in §3 of
`next-steps.md`).

---

## 7. Alternatives, ranked by value / effort

| # | Direction | Why | Effort |
|---|---|---|---|
| 1 | **Change the workload shape, not the kernel.** Target prompt/prefill (`M>1`), batched inference, embeddings, ASR encoders or vision convs. | Square shapes already reach **32.2 GMAC/s** and `32×4096×4096` improved **4.5×** (421 → 94 ms). HVX is compute-efficient exactly where decode is not. | Low — reuse `bench_cached` and the existing backend; mostly a new benchmark harness. |
| 2 | **Change the success metric.** Measure joules/token or "CPU cycles freed" instead of tokens/s. | A DSP that is slower but lets four CPU cores sleep may be the actual win on a phone; §7 currently only reports latency. | Low — needs a power-measurement setup, no kernel work. |
| 3 | **Close the correctness gap.** Per-block / per-K-group quantisation to fix §5.4; fuse the output rescale on the DSP. | Completeness, not throughput; rescale is worth ≤0.8%. | Low–medium. |
| 4 | **Move to a newer SoC (v73+).** | Enables the upstream `ggml-hexagon` fused backend with HMX and all official kernels; the integration knowledge from this PoC transfers directly. | Hardware change. |
| 5 | QNN legacy `DSP(V66)` backend | Small graphs only, no LLM operator coverage (`docs/05` §1). | Not an alternative for this goal. |

---

## 8. Explicit scope of Stage 9

**Done:** source-level analysis of ggml 0.9.0's split scheduler (passes 1–5 and
`compute_splits`), our backend's capability declarations, the real op inventory
of a llama layer, the upstream `ggml-hexagon` implementation and its arch
constraints, the engineering-volume comparison, and the arithmetic ceiling.

**Not done (and deliberately so):** no `src/` edits, no QAIC regeneration, no
skel/`.so` rebuild, no `llama-cli` execution, no firmware/service/DTB change,
no reboot. cDSP heap headroom and V66 HVX fp32 vector capability remain
**unverified**; both would need separate read-only probes before any
implementation decision, and both can only make the verdict *more* negative.

**Status:** `docs/next-steps.md` §6 item 1 is closed as "assessed, not worth
implementing".
