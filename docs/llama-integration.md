# Integrating the cDSP backend with llama.cpp (Stage 6)

This document is the runnable recipe for loading `libggml-cdsp.so` into the
device's **llama.cpp 0.9.0** build and offloading `GGML_OP_MUL_MAT` to the
Redmi K20 Pro cDSP. It also records the measured end-to-end result on
`Qwen3.5-4B-Q4_K_M.gguf`.

Prerequisites: `scripts/device/deploy_cdsp_gemm.py` and
`scripts/device/deploy_ggml_cdsp.py` have both completed successfully (they
install `libgemm_skel.so`, build `libggml-cdsp.so` and
`libggml_cdsp_preload.so`, and run all correctness tests).

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
```

Modes: `normal` (cache + int8 + cDSP), `nocache` (bypass weight cache),
`hostf32` (exact host F32 GEMM, no DSP/int8/cache), `cpu` (no preload).

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
use it for a 4B model. Small models whose int8 weights fit in the cache are
the case where this backend should win.

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
| `cdsp_gemm_open` returns `-1` / `0x8000041d` after a killed run | the cDSP subsystem has crashed. Check `cat /sys/class/remoteproc/remoteproc1/state` → `crashed` and `dmesg` for `qcom_q6v5_pas ... fatal error received`. Writing `stop`/`start` to that sysfs node does **not** recover it; use **`systemctl reboot`** (never `kexec`). Killing `llama-cli` with `kill -9` mid-FastRPC is the usual trigger — prefer the built-in `timeout` wrapper and let it exit cleanly |

To attribute an accuracy regression, run the same prompt with
`GGML_CDSP_HOST_F32=1` (exact host F32 GEMM). If that is coherent but the
normal path is not, the dequant/layout code is fine and the fault is in int8
quantization, the DSP kernel or the weight cache; then try
`GGML_CDSP_NO_CACHE=1` to isolate stale cache entries.
