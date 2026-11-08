# Integrating the cDSP backend with llama.cpp (Stage 6 / 7 / 8)

This document is the runnable recipe for loading `libggml-cdsp.so` into the
device's **llama.cpp 0.9.0** build and offloading `GGML_OP_MUL_MAT` to the
Redmi K20 Pro cDSP. It records the measured end-to-end results on
`Qwen3.5-4B-Q4_K_M.gguf` (Stage 6/7) and on `smollm2-135m-q4_k_m.gguf`
(Stage 8, §4.2).

Prerequisites: `scripts/device/deploy_cdsp_gemm.py` and
`scripts/device/deploy_ggml_cdsp.py` have both completed successfully (they
install `libgemm_skel.so`, build `libggml-cdsp.so` and
`libggml_cdsp_preload.so`, and run all correctness tests).

> **Stage 10 addition.** Everything below still applies, but there is now a
> turnkey alternative to hand-running `llama-cli`:
> [`llama-cdsp.service`](llama-cdsp-service.md) serves the same backend over
> HTTP/WebUI on port 8082 (stock CPU service stays on 8081). Start it with
> `python3 scripts/device/deploy_llama_cdsp.py`, then run
> `python3 scripts/device/llama_cdsp_smoke.py` for an A/B check.

---

## 1. Why an `LD_PRELOAD` shim is needed

llama.cpp's `ggml_backend_load_all()` has two behaviours that work against an
out-of-tree backend on this device:

1. It only scans a **hard-coded list** of backend names (plus the optional
   `GGML_BACKEND_PATH` at the very end).
2. `llama_backend_init()` only calls `ggml_backend_load_all()` when
   `ggml_backend_reg_count() == 0`.

The device's llama.cpp was built with `GGML_BACKEND_DL:BOOL=OFF`
(`CMakeCache.txt` also shows `GGML_BACKEND_DIR` empty), so the CPU backend is
**statically registered** before `llama_backend_init()` runs. That means the
`GGML_BACKEND_PATH` branch is never reached — setting it alone produces
`offload count = 0` with no error.

`src/ggml-backend/cdsp_preload.c` solves this with a library constructor:

```c
void * ggml = dlopen("libggml.so", RTLD_NOW | RTLD_GLOBAL);
ggml_backend_load_fn load = dlsym(ggml, "ggml_backend_load");
load(getenv("GGML_CDSP_BACKEND_PATH") ?: DEFAULT_PATH);
```

`ggml_backend_load()` is exported by ggml regardless of `GGML_BACKEND_DL`, and
our plugin exports `ggml_backend_init()` via `GGML_BACKEND_DL_IMPL`, so the
registration works without rebuilding llama.cpp. Verify with:

```bash
LD_PRELOAD=/data/local/tmp/ggml-backend/libggml_cdsp_preload.so \
  ./llama-cli --version
# -> load_backend: loaded cdsp backend from .../libggml-cdsp.so
# -> [cdsp-preload] registered cDSP backend from .../libggml-cdsp.so
```

---

## 2. Run command

All paths below are the ones used by the deploy scripts. `-st` (`--single-turn`)
is required for non-interactive runs; 0.9.0 has **no** `-no-cnv` flag.

```bash
BIN=/root/llama.cpp-0.9.0/build/bin
MDL=/root/models/Qwen3.5-4B-Q4_K_M.gguf
BE=/data/local/tmp/ggml-backend

cd $BIN
LD_LIBRARY_PATH=$BIN:/usr/local/lib \
LD_PRELOAD=$BE/libggml_cdsp_preload.so \
GGML_CDSP_BACKEND_PATH=$BE/libggml-cdsp.so \
GGML_CDSP_MIN_ELEMS=100000 \
GGML_CDSP_MAX_ELEMS=3000000 \
timeout 300 ./llama-cli -m $MDL \
  -p "The capital of France is" -n 64 -st -c 512 -t 4 --no-warmup
```

> Environment assignments must come **before** `timeout`; putting them after
> makes the shell treat `LD_LIBRARY_PATH=...` as the command and fails with
> `RC=127`.

For a repeatable one-shot diagnostic (captures generated text, offload count,
cache hits/misses and tokens/s), use the bundled driver:

```bash
# from the repo root on the host (uses USB-NCM SSH to 172.16.42.1)
python3 scripts/device/llama_diag.py normal --tokens 64 --no-debug \
    --min-elems 100000 --max-elems 3000000
python3 scripts/device/llama_diag.py cpu    --tokens 64 --no-debug   # baseline

# Stage 7: batched RPC, with split statistics
python3 scripts/device/llama_diag.py normal --tokens 64 --no-debug \
    --min-elems 100000 --max-elems 3000000 \
    --env "GGML_CDSP_BATCH=1 GGML_CDSP_SPLIT_STATS=1"

# Deterministic A/B: fixed seed + greedy decoding removes sampling variance
python3 scripts/device/llama_diag.py normal --tokens 48 --no-debug \
    --seed 42 --temp 0 --min-elems 100000 --max-elems 3000000

# Stage 8: a second, small model (filename is resolved under /root/models)
python3 scripts/device/llama_diag.py cpu --model smollm2-135m-q4_k_m.gguf \
    --tokens 64 --no-debug --seed 42 --temp 0
python3 scripts/device/llama_diag.py normal --model smollm2-135m-q4_k_m.gguf \
    --tokens 64 --no-debug --seed 42 --temp 0 \
    --min-elems 500000 --max-elems 2000000

# Stage 8: attribute where backend time goes
python3 scripts/device/llama_diag.py normal --model smollm2-135m-q4_k_m.gguf \
    --tokens 64 --no-debug --seed 42 --temp 0 \
    --min-elems 500000 --max-elems 2000000 \
    --env "GGML_CDSP_PROFILE=1"
```

Modes: `normal` (cache + int8 + cDSP), `nocache` (bypass weight cache),
`hostf32` (exact host F32 GEMM, no DSP/int8/cache), `cpu` (no preload).
`--env` appends arbitrary `KEY=VAL` pairs; `--seed` / `--temp` make runs
deterministic; `--model` takes a **bare GGUF filename** (or a device-absolute
path), `--cache-mb` overrides `GGML_CDSP_CACHE_MB`, `--threads` sets `-t`.
The driver refuses to run when the cDSP remoteproc is not `running` (a crashed
DSP yields a meaningless `0.0 t/s`).

> **Windows/Git Bash:** passing `--model /root/models/x.gguf` gets rewritten by
> MSYS into `C:/Program Files/.../root/models/x.gguf`. Always pass the bare
> filename; `llama_diag.py` resolves it under `/root/models` and rejects a
> rewritten path with a clear error.

---

## 3. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GGML_CDSP_BACKEND_PATH` | `/data/local/tmp/ggml-backend/libggml-cdsp.so` | plugin `.so` the preload shim registers |
| `GGML_CDSP_CACHE_MB` | 256 | LRU weight-cache capacity (DSP-side prepacked int8 bytes) |
| `GGML_CDSP_MAX_ELEMS` | 0 (unlimited) | skip weights with `N*K` **above** this; leave big matrices on CPU to avoid cache thrash |
| `GGML_CDSP_MIN_ELEMS` | 0 (off) | skip weights with `N*K` **below** this; leave tiny matmuls on CPU to avoid FastRPC overhead |
| `GGML_CDSP_DEBUG` | unset | log every offloaded `MUL_MAT` + cumulative cache hits/misses/bytes |
| `GGML_CDSP_TRACE` | unset | additionally dump `w`/`a`/`dst` type, `ne`, `nb`, contiguity, nbytes |
| `GGML_CDSP_NO_CACHE` | unset | diagnostic: upload weights on every call (releases the handle after each), to detect stale cache entries |
| `GGML_CDSP_HOST_F32` | unset | diagnostic: exact F32 GEMM on the AP using the same dequant helpers; bypasses int8, DSP and cache |
| `GGML_CDSP_BATCH` | unset | **Stage 7**: merge all independent `MUL_MAT`s of one scheduler split into a single FastRPC call (see §4.1) |
| `GGML_CDSP_SPLIT_STATS` | unset | print a histogram of `MUL_MAT`-per-split at `.so` unload, plus batched-RPC counts |
| `GGML_CDSP_PROFILE` | unset | **Stage 8**: print a four-stage timing breakdown (weight/cache, activation quant, FastRPC, output rescale) at `.so` unload |

`supports_op()` and `cdsp_mul_mat()` share **one** capability predicate
(`cdsp_can_mul_mat()`), so `MIN_ELEMS` / `MAX_ELEMS` never cause the scheduler
to dispatch an op the compute path would reject (that mismatch previously
produced `graph_compute ... failed with error -1`).

---

## 4. Verified end-to-end result (Qwen3.5-4B-Q4_K_M, 64 tokens)

Output is coherent and matches the CPU baseline (including the correct
`Capital: Paris`); `errors=0`.

| Configuration | Prompt | Generation | vs CPU |
|---|---|---|---|
| CPU only (`cpu`) | 5.5–6.4 t/s | 3.6–4.1 t/s | 100% |
| cDSP, original config (`MIN=0 MAX=3M`, pre-§2 kernel) | 4.9 t/s | 3.0 t/s | ~73% |
| cDSP, Stage 6 recommended (`MIN=100000 MAX=3M`, tuned HVX) | 5.3–6.0 t/s | 3.4–3.8 t/s | **~93–95%** |

Numbers vary ~10% with device thermals; always compare cDSP and CPU runs taken
in the same time window (the device has no active cooling and DVFS throttles
both the CPU and the DSP).

**Honest conclusion:** cDSP offload is now *correct* and close to CPU parity,
but does **not** beat the CPU for this 4B model. Two structural reasons:

- A real decode issues ~50–60 `MUL_MAT` per token. Each offloaded op costs a
  FastRPC round-trip (measured ~0.25–0.32 ms for a 128-MAC ping), so RPC
  overhead alone is ~15 ms/token.
- The 256 MB weight cache cannot hold a 4B model as int8, so
  `MAX_ELEMS=3000000` leaves the large FFN matrices on the CPU. In a
  2006-offload trace, **1488 (74%)** were the tiny `1×32×2560` shape — only
  82k MACs each, i.e. almost pure RPC overhead. `MIN_ELEMS=100000` removes
  those and is the main reason for the 73% → 94% improvement.

`MAX_ELEMS=0` (full offload) was tried and **hangs** on cache thrash; do not
use it for a 4B model. The natural follow-up hypothesis was that a small model,
whose int8 weights fit entirely in the cache, is where this backend should win.
**Stage 8 tested exactly that and falsified it — see §4.2.**

### 4.1 Stage 7: batched RPC — measured, but not an end-to-end win

`ggml_backend_sched` hands the ACCEL backend a whole **split** through
`graph_compute()`. With `GGML_CDSP_SPLIT_STATS=1` on a 64-token run we measured
528 splits / 726 `MUL_MAT` nodes; **198 splits (37.5%) contain two independent
`MUL_MAT`s** (Q/K/V-style projections sharing an activation). Those are
mergeable. The backend now detects them, prepares each op (dequant, cache
lookup, activation quantization), concatenates the A/C buffers and issues
**one** FastRPC call:

```
dims = { -count, M0,N0,K0,h0, M1,N1,K1,h1, ... }
A    = concat(A_i)      C = concat(C_i)
```

The negative `dims[0]` is the batch marker; it reuses the existing QAIC method
(only `dimsLen`/`ALen`/`CLen` change), so **no IDL or stub/skel regeneration
was needed**. Enable with `GGML_CDSP_BATCH=1`; when disabled the original
scalar path runs unchanged. Correctness is guarded by
`src/host/host_test_batch.c` (two *different* weight shapes, 1056 elements
verified against scalar GEMM) and by greedy-decoding A/B, where batch-on and
batch-off produce **byte-identical generated text**.

| Measurement | Scalar (2 RPCs) | Batched (1 RPC) | Saving |
|---|---|---|---|
| `host_test_batch` latency per op-pair | 2.505 ms | 1.962 ms | **21.6%** |
| earlier run, warmer DSP | 2.899 ms | 2.094 ms | 27.8% |
| 64-token llama-cli, `MIN=100000` | 3.3–3.5 t/s | 3.2–3.4 t/s | **not measurable** |
| 48-token, `MIN=0`, batch first | 2.9 t/s | 3.1 t/s | (order-biased, see below) |

**Honest conclusion:** the batched encoding works and removes 78–198
round-trips per run, but tokens/s does **not** improve in a way that survives
controls. Deterministic A/B runs showed a strictly monotonic slowdown
(3.5 → 3.4 → 3.3 → 3.2 t/s) regardless of mode — the uncooled phone throttles
under consecutive runs, so whichever mode runs first wins by ~0.1–0.2 t/s.
Reversing the order (ABBA: on 3.4, off 3.2) confirmed this is an ordering
artefact, not a batching effect. The reason is arithmetic: ~0.8 ms saved per
merged pair × ~3 pairs/token ≈ 2.4 ms against a ~280 ms/token budget (<1%),
well inside the thermal noise. This matches the roofline finding in
`docs/next-steps.md` §2.2 — the bottleneck is decode weight bandwidth and the
CPU-side FFN matrices, not RPC count.

`GGML_CDSP_BATCH=1` is therefore kept **opt-in**. The recommended config is
unchanged: `MIN_ELEMS=100000 MAX_ELEMS=3000000`. Letting the tiny `1×32×2560`
ops piggyback (`MIN=0` + batch) reached 3.1 t/s vs 2.9 t/s scalar, still below
the 3.3 t/s of leaving them on the CPU — their DSP compute + quantization cost
more than the RPC they save.

### 4.2 Stage 8: small model (SmolLM2-135M-Q4_K_M) — weights fit, still slower

Model: `smollm2-135m-q4_k_m.gguf` (101 MB, 30 layers, emb 576, FFN 1536,
vocab 49152). Transfer it to `/root/models/` first (see `docs/reproduce.md`
§4). Clean-state runs, 64 tokens, `--seed 42 --temp 0 --no-debug`:

| Configuration | Prompt | Generation | vs CPU |
|---|---|---|---|
| CPU only | 123.2 t/s | **81.2 t/s** | 100% |
| cDSP, best (`MIN=500000 MAX=2000000`, FFN only) | 47.1 t/s | **23.8 t/s** | 29% |
| cDSP, full offload (`MIN=0 MAX=0`) | 44.3 t/s | 13.4 t/s | 17% |
| cDSP, `GGML_CDSP_BATCH=1` | 48.7 t/s | 19.0 t/s | no gain |
| cDSP, `GGML_CDSP_NO_CACHE=1` | 17.4 t/s | 1.4 t/s | cache is worth 17.6× |

**Cache residency was confirmed, then falsified as the explanation**: full
offload reports `cache hits=1984 misses=31 bytes=44015616` — 42 MB, zero
thrash — exactly the scenario Stage 6 predicted would win. It is still 3.4×
slower than four CPU threads.

Where the time goes (full analysis in `docs/next-steps.md` §5.2):

- `GGML_CDSP_SPLIT_STATS=1`: 2015 splits / 2015 `MUL_MAT` nodes /
  **0 splits with >1 matmul**. The 135M graph alternates CPU-only ops and
  matmuls, so Stage 7 batching has nothing to merge (`BATCH=1`: 19.0 vs
  19.4 t/s).
- `GGML_CDSP_PROFILE=1`: FastRPC + DSP GEMM takes 1157.1 ms over 1040 ops
  (**1.113 ms/op, 59.5%** of backend time); activation quant 3.2%, output
  rescale 0.8%, cold weight uploads 36.5%.
- ~24 ms/token sits **outside** the backend in CPU↔ACCEL split
  synchronization. Loading the backend with zero offloads
  (`MIN_ELEMS=100000000`) still gives 71.0 t/s vs CPU 80.6 t/s, so this is not
  registration overhead.
- Persistent zero-copy is ruled out on this kernel: `fastrpc_mmap()` returns
  `AEE_EUNSUPPORTED (0x14)` with dmesg `mem mmap error`, and
  `rpcmem_alloc_def` buffers are 1.6–36.6% **slower** than plain `malloc`
  (`src/host/bench_regbuf.c`, kept in the regression as `REGBUF_TEST PASS`).

Correctness: `GGML_CDSP_NO_CACHE=1` produces the same text as the cached path,
so there is no stale-cache bug; but greedy decoding diverges from CPU after
~10 tokens because the 135M model's narrow logit margins let the ~0.4–0.7%
int8 error flip an argmax. **Not bit-exact on small models**, though the text
remains fluent.

Practical takeaway: for SmolLM2-135M the best cDSP config is
`MIN_ELEMS=500000 MAX_ELEMS=2000000` (FFN only), and the CPU is still 3.4×
faster. What matters is how much compute amortizes each cross-backend split,
not whether the weights fit in DSP memory.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No `loaded cdsp backend` line | `LD_PRELOAD` shim missing or `GGML_CDSP_BACKEND_PATH` wrong; `GGML_BACKEND_PATH` alone does **not** work with `GGML_BACKEND_DL=OFF` |
| `graph_compute ... -1` | capability predicate mismatch (fixed in Stage 6); also seen with full offload on a 4B model — use `MAX_ELEMS` |
| Garbled output / Cyrillic gibberish | stale `libgemm_skel.so` or `libggml-cdsp.so` on the device. Rebuild both; the ggml 0.9.0 Q8_0 block is **34 B with an fp16 scale** and Q4_K is **144 B with `scales[12]`** — using the older 36 B / 140 B layouts silently corrupts every layer |
| `invalid argument: -no-cnv` | 0.9.0 uses `-st` / `--single-turn` |
| `RC=127` | environment assignments placed after `timeout` |
| Very slow / hangs | cache thrash: lower `GGML_CDSP_MAX_ELEMS`, raise `GGML_CDSP_MIN_ELEMS`, or reduce `GGML_CDSP_CACHE_MB`. Check for a stray `llama-cli` with `pgrep -a llama-cli` |
| `Compute error` from DSP | `systemctl restart cdsprpcd` and confirm `/usr/lib/rfsa/cdsp/libgemm_skel.so` is the current build |
| `cdsp_gemm_open` returns `-1` / `0x8000041d` after a killed run | the cDSP subsystem has crashed. Check `cat /sys/class/remoteproc/remoteproc1/state` → `crashed` and `dmesg` for `qcom_q6v5_pas ... fatal error received`. Writing `stop`/`start` to that sysfs node does **not** recover it; use **`systemctl reboot`** (never `kexec`). Killing `llama-cli` with `kill -9` mid-FastRPC is one trigger — prefer the built-in `timeout` wrapper and let it exit cleanly |
| `0.0 t/s` + `Error: Compute error` with no kill involved | **cDSP watchdog** after many consecutive long runs: `dmesg` shows `SFR Init: wdog or kernel error suspected` / `remoteproc1: crash detected in cdsp`. Seen after 8+ back-to-back 48-token runs. Same recovery (`systemctl reboot`, ~50 s over USB-NCM), then let the phone cool. `llama_diag.py` now checks `remoteproc1/state` before every run and refuses to report a meaningless benchmark when it is `crashed` |
| `error: invalid argument: C:/Program Files/.../root/models/x.gguf` | Git Bash/MSYS rewrote the absolute device path. Pass a **bare filename** to `--model` (`smollm2-135m-q4_k_m.gguf`); the driver resolves it under `/root/models` |
| Small-model output differs from CPU after ~10 tokens (greedy) | expected: per-vector int8 error flips narrow-margin argmaxes on a 135M model. Not a bug; use `--temp 0` and compare a prefix, or use `GGML_CDSP_NO_CACHE=1` to rule out stale cache. See §4.2 |
| `fastrpc_mmap` / `remote_register_buf` returns `20` | `AEE_EUNSUPPORTED` on this kernel (dmesg `mem mmap error`). Persistent mapping is unavailable and plain RPCMEM is slower than `malloc` — do not retry; see `src/host/bench_regbuf.c` and `docs/next-steps.md` §5.3 |

To attribute an accuracy regression, run the same prompt with
`GGML_CDSP_HOST_F32=1` (exact host F32 GEMM). If that is coherent but the
normal path is not, the dequant/layout code is fine and the fault is in int8
quantization, the DSP kernel or the weight cache; then try
`GGML_CDSP_NO_CACHE=1` to isolate stale cache entries.
