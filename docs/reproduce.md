# Reproduce — step by step

This document walks from a freshly cloned repo to a verified cDSP `MUL_MAT`
backend. Every command is run on the **Windows build host** unless marked
`# on device`.

---

## 0. Environment

### 0.1 Connect the phone over USB-NCM

The phone exposes an Ethernet-over-USB link. On the Windows host it appears as an
adapter with IP **`172.16.42.2`**; the phone is **`172.16.42.1`** (root, password
`1234`). `scripts/device/k20_ssh.py` already encodes these defaults; override via
env vars `K20_HOST`, `K20_PORT`, `K20_USER`, `K20_PASS` if yours differ.

Quick connectivity check from the host:

```bash
python3 scripts/device/k20_ssh.py "uname -a; pgrep -x cdsprpcd >/dev/null && echo DAEMON_OK || echo DAEMON_MISSING"
```

You should see a Linux `uname` line and `DAEMON_OK`. If not, the cDSP FastRPC
channel is not up — set it up first (see the companion
[`raphael-cdsp-fastrpc`](https://github.com/doeasier/raphael-cdsp-fastrpc) repo).

### 0.2 (Optional) Install the Hexagon SDK — only for Stage 0

If you intend to **rebuild** the DSP skel yourself, install Qualcomm
**Hexagon SDK 4.3.0.0** and make sure these exist:

```
C:/Qualcomm/Hexagon_SDK/4.3.0.0/ipc/fastrpc/qaic/WinNT/qaic.exe
C:/Qualcomm/Hexagon_SDK/4.3.0.0/tools/HEXAGON_Tools/8.4.11/Tools/bin/hexagon-clang.exe
```

If you just want to reproduce the pipeline, **skip Stage 0** — the committed
`prebuilt/libgemm_skel.so` is used directly.

---

## 1. Stage 0 — build the cDSP GEMM skeleton (optional)

```bash
cd src/dsp
bash build_win.sh
cd ../..
```

What it does:

1. `qaic.exe -mdll` turns `gemm.idl` into `gen/cdsp_gemm_skel.c` +
   `gen/cdsp_gemm_stub.c` + `gen/gemm.h` (these are **already committed**, so the
   SDK is only needed if you change the IDL).
2. `hexagon-clang.exe -mv66 -mhvx` compiles `gemm_dsp.c` + the skel into
   `prebuilt/libgemm_skel.so`.

> **Gotcha:** the correct HVX flag for this clang is **`-mhvx`** (not
> `-mhvx=128` — that form is rejected and silently disables `__HVX_ARCH__`,
> breaking the HVX headers). The `qurt_hvx_lock(QURT_HVX_MODE_128B)` call in
> `gemm_dsp.c` is what actually enables the 128-byte unit for the call.

Sanity check the build (run from `src/dsp`):

```bash
nm -D ../../prebuilt/libgemm_skel.so | grep -iE "cdsp_gemm_(open|close|rpc_init|gemm_int8|gemm_upload|gemm_compute_cached|gemm_release)"
# all seven must be 'T' (defined). Missing open/close => DSP dlopen fails with 0x80000406.
```

---

## 2. Stage 1 — deploy the skel + verify the GEMM kernel

```bash
python3 scripts/device/deploy_cdsp_gemm.py
```

The script:

1. Uploads `host_test.c`, the QAIC header/stub, the AEEStd headers, and
   `libgemm_skel.so` to `/data/local/tmp/cdsp_gemm/`.
2. Compiles `host_test` on-device (`gcc -Igen -ladsprpc ...`).
3. Installs the skel to `/usr/lib/rfsa/cdsp/libgemm_skel.so` (backing up any
   existing one) and restarts `cdsprpcd` via **systemd** (do NOT `pkill` it
   manually — that drops the env it needs to find skels and can starve sshd).
4. Runs `host_test` at three sizes.

Expected tail:

```
=== host_test 32x128x128 (rc=0) ===
host: cDSP session open, handle=...
host: gemm_int8 done (M=32 N=128 K=128) in 0.99 ms
host: PASS - all 4096 elements verified correct.
=== host_test 64x256x256 (rc=0) ===
host: PASS - all 16384 elements verified correct.
=== host_test 128x512x512 (rc=0) ===
host: PASS - all 65536 elements verified correct.
```

The DSP computes the int8 GEMM; the host compares against an exact int8 reference
(`ref_gemm` in `host_test.c`). All elements must match.

You can also run real model-layer dimensions directly on the device:

```bash
# on device
cd /data/local/tmp/cdsp_gemm
./host_test 1 4096 4096    # decode attention
./host_test 1 11008 4096   # decode ffn
./host_test 32 4096 4096   # prefill attention
```

All three print `host: PASS`.

The same script also builds and runs **`host_test_cached`** — the Stage 5
validation of the weight-cache API. It uploads the weight `B` once
(`gemm_upload` → a handle), then calls `gemm_compute_cached` (only `A` shipped)
and verifies the result matches the reference:

```
=== host_test_cached (weight-cache) 1x11008x4096 (rc=0) ===
host: gemm_upload done -> weight handle=1 in 672.05 ms
host: gemm_compute_cached done (M=1 N=11008 K=4096) in 37.42 ms
host: PASS - all 11008 elements verified correct.
host: gemm_release rc=0
```

The 672 ms `upload` is the **one-time** cost of shipping the ~45 MB weight to the
DSP; the 37 ms `compute_cached` is what every subsequent token pays (only the tiny
activation `A` is shipped). Compare with the naive one-shot path which re-ships the
whole weight on every call (≈ 762 ms/token at this size).

---

## 3. Stage 2 — build the ggml backend + single-backend test

```bash
python3 scripts/device/deploy_ggml_cdsp.py
```

The script:

1. Uploads the ggml **0.9.0** public headers (`deps/ggml-0.9.0-headers/`) to
   `/data/local/tmp/ggml-include` so the plugin is compiled against the exact ABI
   it was written for.
2. Stages the plugin sources into `/data/local/tmp/ggml-cdsp` and uploads them.
3. Runs `build_device.sh` on-device → `libggml-cdsp.so`.
4. Installs `libggml-cdsp.so` into `GGML_BACKEND_DIR=/data/local/tmp/ggml-backend`.
5. Builds and runs `test_cdsp.cpp`, which builds an F32 `MUL_MAT`, asks the cDSP
   backend to compute it, and compares against a CPU int8 emulation of the *same*
   path (tiny tolerance).

Expected tail:

```
OK: loaded backend 'cdsp'
OK: MUL_MAT computed on cDSP; max abs error vs int8-emulation = 0.000000
PASS
>>> Stage 2 VERIFIED: cDSP backend computes MUL_MAT correctly.
```

> **Build notes for the plugin:**
> - The QAIC stub `cdsp_gemm_stub.c` is **C**, not C++. `build_device.sh`
>   compiles it with `gcc -c` into a `.o` and links it into the C++ shared lib;
>   `gemm.h` wraps its prototypes in `extern "C"`.
> - The test links against **all three** ggml libs:
>   `-lggml -lggml-base -lggml-cpu` (on this build `libggml` is split).

---

## 4. Stage 2.5 — scheduler integration test

The same `deploy_ggml_cdsp.py` continues with `test_sched.cpp`, which uses the
**`ggml_backend_sched`** with two backends — cDSP first, CPU second,
`op_offload=true` — exactly how llama.cpp routes ops. It builds
`MUL_MAT(cDSP) → relu(CPU)` and checks the result.

Expected tail:

```
OK: loaded cDSP backend 'cdsp'
OK: loaded CPU backend 'CPU'
OK: scheduler ran MUL_MAT(cDSP)+relu(CPU); max abs error vs int8-emu = 0.000000
SCHED_TEST PASS
>>> CONFIRMED: scheduler offloaded MUL_MAT to the cDSP backend.
>>> Stage 2.5 VERIFIED: backend integrates under the ggml multi-backend scheduler (cDSP + CPU).
```

With `GGML_CDSP_DEBUG=1` you also see the offload line:
`[ggml-cdsp] offloaded MUL_MAT M=64 N=96 K=128`.

---

## 4. Stage 5 — weight-cache RPC (upload once, reuse handle)

The same `deploy_ggml_cdsp.py` continues with `test_cache.cpp`, which builds a
`MUL_MAT` graph, computes it **twice with the same weight tensor**, and verifies:

1. both runs are bit-identical to the CPU int8 emulation (the cache path is
   correct), and
2. the **second** run is a cache **HIT** — the weight is *not* re-uploaded.

With `GGML_CDSP_DEBUG=1` the backend prints `cache hits=1 misses=1`, and the
deploy script greps for it to confirm reuse:

```
OK: 2 computes ran; max abs error vs int8-emu = 0.000000
CACHE_TEST PASS
>>> CONFIRMED: cached weights reused across computes (no re-upload of the weight matrix).
>>> Stage 5 VERIFIED: weight-cache API works end-to-end (correct + weights uploaded once).
```

The plugin's LRU cache size is tunable via `GGML_CDSP_CACHE_MB` (default 256 MB).
For a real model whose int8 weights exceed that, the least-recently-used weights
are released and re-uploaded on next use — see [docs/next-steps.md](next-steps.md)
for the capacity caveat.

---

## 5. Stage 6 — HVX tuning + real llama.cpp inference

### 5.1 Rebuild and deploy the tuned DSP skel

`gemm_dsp.c` now contains three auto-selected HVX paths (§3.5 of
[architecture.md](architecture.md)). Rebuild on the Windows host — **run the
script from `src/dsp`**, QAIC resolves `gemm.idl` relative to the CWD:

```bash
cd src/dsp && bash build_win.sh && cd ../..
# -> prebuilt/libgemm_skel.so  (7 exported T symbols, ~35 vrmpy instructions)
```

`deploy_cdsp_gemm.py` now ships `bench_cached.c`, compiles it, installs the skel
to all cDSP search paths, restarts `cdsprpcd`, runs the 9 correctness cases and
finally the throughput suite (with a 200-iteration warm-up to get the DSP off
its cold clock):

```bash
python3 scripts/device/deploy_cdsp_gemm.py | tee evidence/stage6-dsp-regression.txt
#   -> host: PASS - all ... elements verified correct.   (9 cases)
#   -> BENCH M=128 N=512 K=512 ... 32.2 GMAC/s ... verify=PASS
```

> If you deploy the skel by hand, remember **all three** paths —
> `/usr/lib/rfsa/cdsp`, `/usr/share/qcom/dsp/cdsp`,
> `/data/local/tmp/cdsp_gemm` — and `systemctl restart cdsprpcd`. A stale skel in
> one of them is a silent correctness bug (it looked exactly like "garbled model
> output" during Stage 6 debugging).

### 5.2 Rebuild the ggml backend + integration shim

```bash
python3 scripts/device/deploy_ggml_cdsp.py | tee evidence/stage6-ggml-regression.txt
#   -> PASS / SCHED_TEST PASS / CACHE_TEST PASS
#   -> Q4_K_TEST PASS, Q8_0_TEST PASS
#   -> builds /data/local/tmp/ggml-backend/libggml_cdsp_preload.so
```

### 5.3 Run the real model

Model must be on the phone (`/root/models/Qwen3.5-4B-Q4_K_M.gguf`, 2.55 GB;
served from the host over HTTP + `curl -C -`, SFTP is flaky for large files).
Full recipe in [llama-integration.md](llama-integration.md). Short version:

```bash
python3 scripts/device/llama_diag.py normal --tokens 64 --no-debug \
    --min-elems 100000 --max-elems 3000000
python3 scripts/device/llama_diag.py cpu --tokens 64 --no-debug
```

`llama_diag.py` modes: `normal`, `nocache`, `hostf32`, `cpu`. It saves the full
device log to `tmp/diag_<mode>.host.log` on the host and prints a compact
report (offload count, cache hits/misses, tokens/s, generated text).

Expected: coherent `Thinking Process: ... Capital: Paris`, `errors=0`,
~3.4–3.8 t/s generation vs ~3.6–4.1 t/s CPU. Numbers move ~10% with device
thermals — always compare the two configurations in the same time window.

> **Do not use `MAX_ELEMS=0` on a 4B model:** the 256 MB cache thrashes and the
> run effectively hangs. Let `timeout` stop the process; if you must
> `kill -9 llama-cli`, check `cat /sys/class/remoteproc/remoteproc1/state`
> afterwards. A killed FastRPC client can leave the cDSP in `crashed` state —
> sysfs `stop`/`start` does not recover it, only `systemctl reboot` (never
> `kexec`) does. All subsequent `cdsp_gemm_open` calls then fail with `rc=-1`.

### 5.4 Stage 7 — batched RPC (opt-in)

Rebuild both sides (`build_win.sh` for the skel, `deploy_cdsp_gemm.py` +
`deploy_ggml_cdsp.py` to install and verify). The deploy script now also builds
and runs `host_test_batch.c`:

```bash
python3 scripts/device/deploy_cdsp_gemm.py | tee evidence/stage7-dsp-regression.txt
#   -> host: BATCH_TEST PASS - 1056 elements verified correct.
#   -> host: scalar 2 RPCs = 2.505 ms/pair; batched 1 RPC = 1.962 ms/pair

python3 scripts/device/llama_diag.py normal --tokens 48 --no-debug \
    --seed 42 --temp 0 --min-elems 100000 --max-elems 3000000 \
    --env "GGML_CDSP_BATCH=1 GGML_CDSP_SPLIT_STATS=1"
#   -> [cdsp-split] calls=208 nodes=572 mul_mat=286 splits_with_>1_mm=78
#   -> [cdsp-batch] splits=78 ops=156 rpcs_saved=78
```

`--seed` + `--temp 0` make the run deterministic, so batch-on and batch-off
generated text can be compared byte-for-byte (they are identical). Do **not**
chain more than a handful of long runs back-to-back: the cDSP watchdog can fire
even with clean exits (`SFR Init: wdog or kernel error suspected`), producing
`0.0 t/s`. `llama_diag.py` checks `remoteproc1/state` before each run; recover
with `systemctl reboot` and let the phone cool.

### 5.5 Stage 8 — small-model benchmark (SmolLM2-135M)

#### Getting the model onto the phone

`smollm2-135m-q4_k_m.gguf` (101 MB, sha256
`3dac8a6535e754df3da638794da83508b748bb31c4100499a392173d17cb695a`) is a 30-layer
llama model (emb 576, FFN 1536, vocab 49152) whose int8 weights fit entirely in
the 256 MB cache. **Do not use SFTP for a 100 MB file over this USB-NCM link** —
large transfers stall and exhaust sshd sessions. Use a temporary HTTP server on
the Windows host instead:

```bash
# host (Git Bash). The phone sees the host as 172.16.42.2 (`cat /proc/net/arp`).
cd /f/work/WorkBuddy/k20pro/downloads/models
python -m http.server 8765 --bind 0.0.0.0

# device (separate SSH session)
curl -sS -C - -o /root/models/smollm2-135m-q4_k_m.gguf \
    http://172.16.42.2:8765/smollm2-135m-q4_k_m.gguf
sha256sum /root/models/smollm2-135m-q4_k_m.gguf   # must match the above
```

Stop the server when finished. The device has no internet access, but the
USB-NCM link carries HTTP fine.

#### Running the comparison

`llama_diag.py` accepts a **bare GGUF filename** (`--model`), resolves it under
`/root/models`, and additionally supports `--cache-mb` and `--threads`:

```bash
python3 scripts/device/llama_diag.py cpu --model smollm2-135m-q4_k_m.gguf \
    --tokens 64 --no-debug --seed 42 --temp 0
python3 scripts/device/llama_diag.py normal --model smollm2-135m-q4_k_m.gguf \
    --tokens 64 --no-debug --seed 42 --temp 0 \
    --min-elems 500000 --max-elems 2000000 \
    --env "GGML_CDSP_PROFILE=1 GGML_CDSP_SPLIT_STATS=1"
```

Passing an absolute device path such as `/root/models/x.gguf` from Git Bash is
rewritten by MSYS into `C:/Program Files/.../root/models/x.gguf`; the driver now
detects and rejects that with a clear message. Always use the bare filename.

Expected (clean state after `systemctl reboot`): CPU **81.2 t/s**, best cDSP
**23.8 t/s**, full offload 13.4 t/s. `[cdsp-prof]` shows FastRPC at 1.113 ms/op
(59.5%) and `[cdsp-split]` shows 2015 splits with **0** multi-matmul splits.
`deploy_cdsp_gemm.py` also runs `bench_regbuf.c`, which confirms that
`fastrpc_mmap` is unsupported on this kernel and that RPCMEM is slower than
`malloc` (`REGBUF_TEST PASS`).

Greedy-decoding output diverges from CPU after ~10 tokens on this small model;
that is int8 quantization error flipping narrow-margin argmaxes, not a bug.
`GGML_CDSP_NO_CACHE=1` reproduces the normal path's text, ruling out stale
cache. Full analysis: [next-steps.md](next-steps.md) §5,
[llama-integration.md](llama-integration.md) §4.2.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `0x80000406` on `cdsp_gemm_open` | unsigned-PD not enabled / skel not found | script calls `enable_unsigned_pd()`; ensure `libgemm_skel.so` is in `/usr/lib/rfsa/cdsp` and daemon restarted |
| `nm -D` shows `cdsp_gemm_open`/`close` as `U` | skel missing the open/close impls | rebuild `gemm_dsp.c` (they must be defined); `build_win.sh` |
| `test_cdsp` fails to `dlopen("cdsp")` | passing a backend name instead of a path | `ggml_backend_load()` takes a **.so path**; the test passes the full path already |
| `undefined reference to ggml_backend_tensor_get` | only `-lggml` linked | link `-lggml -lggml-base -lggml-cpu` (build_device.sh / deploy handle this) |
| SFTP `ENOENT` on upload | flaky SFTP subsystem on this device | `deploy_cdsp_gemm.py` uses base64-over-exec (`put_exec`); `deploy_ggml_cdsp.py` uses `put_dir` (worked for headers) |
| `graph_compute ... -1` / `Compute error` | scheduler dispatched an op `cdsp_mul_mat()` rejected, or full offload thrashed | Stage 6 shares one `cdsp_can_mul_mat()` predicate; also set `MAX_ELEMS=3000000` |
| Garbled / Cyrillic model output | stale `libggml-cdsp.so` or `libgemm_skel.so`, or wrong Q4_K/Q8_0 block layout | rebuild **both** (§5.1 + §5.2); Q4_K = 144 B / `scales[12]`, Q8_0 = 34 B / fp16 scale |
| `invalid argument: -no-cnv` | 0.9.0 has no such flag | use `-st` / `--single-turn` |
| `RC=127` | env assignments placed after `timeout` | put `LD_LIBRARY_PATH=... LD_PRELOAD=...` **before** `timeout` |
| No `loaded cdsp backend` line | `GGML_BACKEND_PATH` alone does nothing with `GGML_BACKEND_DL=OFF` | use `LD_PRELOAD=.../libggml_cdsp_preload.so` (§5.3 / llama-integration.md) |
| Benchmark numbers 30–50% low | cDSP DVFS cold clock | run a 200-iteration warm-up shape first (`deploy_cdsp_gemm.py` does this) |
| `0.0 t/s` + `Error: Compute error`, no `kill -9` involved | cDSP **watchdog** after 8+ consecutive long runs (`SFR Init: wdog or kernel error suspected`; `remoteproc1=crashed`) | `systemctl reboot` (never `kexec`), then let the device cool. `llama_diag.py` pre-checks the remoteproc state and refuses to benchmark a crashed DSP |
| A/B results where "whichever ran first" is faster | thermal throttling, not a real speedup | use `--seed 42 --temp 0` and an ABBA run order (on/off/off/on) to cancel the linear trend |
| `error: invalid argument: C:/Program Files/.../root/models/x.gguf` | MSYS/Git Bash rewrote the absolute device path | pass a **bare filename** to `--model` (§5.5) |
| Large GGUF upload stalls / sshd sessions time out | SFTP over USB-NCM is unreliable for big files | serve the file over HTTP from the host and `curl -C -` on the device (§5.5) |
| `fastrpc_mmap` returns `20` / dmesg `mem mmap error` | `AEE_EUNSUPPORTED` on this kernel | persistent mapping is unavailable; RPCMEM buffers are slower than `malloc`. Do not retry — see `src/host/bench_regbuf.c` and next-steps.md §5.3 |
| Small-model output differs from CPU after ~10 greedy tokens | int8 error flips narrow logit margins | expected, not bit-exact; use `GGML_CDSP_NO_CACHE=1` to rule out stale cache (§5.5) |

---

## 7. What is NOT done yet

Stages 0–8 are implemented and verified. Batched RPC (§4 of next-steps.md) is
**done** — 21.6% lower RPC-pair latency but no end-to-end win. The small-model
benchmark (§5) is also **done**: SmolLM2-135M keeps all weights in the cache
(42 MB, zero thrash) yet reaches only 29% of CPU tokens/s, falsifying the
cache-residency hypothesis. **Stage 9** then assessed the last architectural
lever without implementing it (source analysis only, no device changes):
[`whole-layer-fusion-feasibility.md`](whole-layer-fusion-feasibility.md).
**Stage 10** then packaged the backend as an independent HTTP service
(`llama-cdsp.service`, port 8082) for application testing — see
[`llama-cdsp-service.md`](llama-cdsp-service.md). It adds no new performance
claim (89.4% of CPU in an A/B smoke test) and does not change any conclusion
below.

- **Remove split boundaries, not RPCs — CLOSED (assessed, not worth doing).**
  ~24 ms/token is spent in CPU↔ACCEL scheduler synchronization on the 135M
  model. ggml 0.9.0 would merge the graph into one split as soon as
  `supports_op()` covers the whole op chain (no scheduler patch needed), but
  even with that overhead at zero the ceiling is **51.8 t/s = 64% of CPU**.
  V66 has no HMX and no portable upstream `ggml-hexagon` skel (v73+ only), so
  the ~46,600-line fused implementation would have to be written from scratch.
  See the assessment for the derivation and ranked alternatives.
- **Weight staging for large models.** The 256 MB cache cannot hold a 4B model
  as int8, so the large FFN GEMMs stay on the CPU. Streaming tiles from AP
  memory is the next structural step.
- **Better-than-per-vector int8 quantization** (per-block / per-K group) to fix
  the small-model divergence; correctness, not throughput.
- **Fuse the output rescale on the DSP.** `GGML_CDSP_PROFILE` shows it is only
  0.8% of backend time, so it is worth doing only as part of a larger fusion.
- Only `GGML_OP_MUL_MAT` is offloaded; all other ops run on the CPU. By design
  for this PoC. Stages 6–8 establish a consistent measured negative result for
  single-operator offload at both 135M and 4B scales.
