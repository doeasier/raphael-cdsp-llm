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
nm -D ../../prebuilt/libgemm_skel.so | grep -iE "cdsp_gemm_(open|close|rpc_init|gemm_int8)"
# all four must be 'T' (defined). Missing open/close => DSP dlopen fails with 0x80000406.
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

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `0x80000406` on `cdsp_gemm_open` | unsigned-PD not enabled / skel not found | script calls `enable_unsigned_pd()`; ensure `libgemm_skel.so` is in `/usr/lib/rfsa/cdsp` and daemon restarted |
| `nm -D` shows `cdsp_gemm_open`/`close` as `U` | skel missing the open/close impls | rebuild `gemm_dsp.c` (they must be defined); `build_win.sh` |
| `test_cdsp` fails to `dlopen("cdsp")` | passing a backend name instead of a path | `ggml_backend_load()` takes a **.so path**; the test passes the full path already |
| `undefined reference to ggml_backend_tensor_get` | only `-lggml` linked | link `-lggml -lggml-base -lggml-cpu` (build_device.sh / deploy handle this) |
| SFTP `ENOENT` on upload | flaky SFTP subsystem on this device | `deploy_cdsp_gemm.py` uses base64-over-exec (`put_exec`); `deploy_ggml_cdsp.py` uses `put_dir` (worked for headers) |

---

## 6. What is NOT done yet

End-to-end llama.cpp *inference* is **not** yet usable: the backend re-ships the
entire weight matrix on every `MUL_MAT` call (see [docs/next-steps.md](next-steps.md)).
The stages above prove correctness and the integration path; the next milestone
is weight caching + HVX peak tuning.
