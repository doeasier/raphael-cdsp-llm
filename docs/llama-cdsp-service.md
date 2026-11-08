# `llama-cdsp.service` — cDSP-backed `llama-server` for application testing

**Purpose.** Expose the cDSP backend as a normal, long-running HTTP service so
applications (WebUI, chat clients, scripts) can be tested against it. This is
**not** a performance improvement: Stages 6–9 established that single-op
offload is slower than the CPU (see `docs/next-steps.md`). The stock CPU service
stays online on port 8081 so both can be exercised side by side.

| | `llama-4b.service` | `llama-cdsp.service` |
|---|---|---|
| Port | 8081 | **8082** |
| Binary | `/root/llama.cpp-0.9.0/build/bin/llama-server` | same binary, **not recompiled** |
| Backend | CPU only | CPU + `cdsp` ACCEL (`MUL_MAT` offload) |
| Attachment | — | `LD_PRELOAD` + `ggml_backend_load()` shim |
| Server slots | 4 (auto) | **1** |
| Enabled on boot | yes | **no** (manual `systemctl start`) |
| Restart | `always` | `on-failure`, max 3 starts / 300 s |

## Why `LD_PRELOAD`

The device build has `GGML_BACKEND_DL=OFF`, so `GGML_BACKEND_PATH` /
`GGML_BACKEND_DIR` are never consulted (`llama_backend_init()` only calls
`ggml_backend_load_all()` when the registry is empty, but the CPU backend is
registered statically). `libggml_cdsp_preload.so` is an ELF constructor that
resolves `ggml_backend_load()` from `libggml.so` and loads
`libggml-cdsp.so`, which registers the `cdsp` ACCEL device. `llama.cpp` then
picks it up automatically; `--op-offload` is on by default and is passed
explicitly in the unit for clarity.

Three variables are what actually attach the DSP, and the unit writes them
**after** `EnvironmentFile=` so a stray edit cannot silently turn the service
back into a CPU-only one:

```ini
Environment=LD_LIBRARY_PATH=/root/llama.cpp-0.9.0/build/bin:/usr/local/lib
Environment=LD_PRELOAD=/data/local/tmp/ggml-backend/libggml_cdsp_preload.so
Environment=GGML_CDSP_BACKEND_PATH=/data/local/tmp/ggml-backend/libggml-cdsp.so
```

The DSP skel lives at `/usr/lib/rfsa/cdsp/libgemm_skel.so`; the
`cdsprpcd` drop-in already puts that directory on `ADSP_LIBRARY_PATH`, so the
service process needs no `ADSP_LIBRARY_PATH` of its own.

## Install and operate

From the repository root (host side, over USB-NCM SSH):

```bash
python3 scripts/device/deploy_llama_cdsp.py            # precheck, install, start
python3 scripts/device/deploy_llama_cdsp.py --status   # inspect only
python3 scripts/device/deploy_llama_cdsp.py --restart  # restart only
python3 scripts/device/deploy_llama_cdsp.py --force-env  # reset tunables
python3 scripts/device/deploy_llama_cdsp.py --enable   # also start on boot
```

On the device:

```bash
systemctl start|stop|restart|status llama-cdsp
journalctl -u llama-cdsp -f
curl -s http://127.0.0.1:8082/health
```

From the host PC (USB-NCM): <http://172.16.42.1:8082/> for the WebUI,
`/health`, `/completion`, `/v1/chat/completions` for the API.

The unit deliberately is **not enabled on boot**: a crashed cDSP needs a full
reboot, so auto-starting an experimental service on every boot would put the
phone in a permanently degraded state. `Restart=on-failure` plus
`StartLimitBurst=3` / `StartLimitIntervalSec=300` (correctly placed in
`[Unit]`) stops the restart loop after repeated failures, leaving a visible
`failed` state.

## Tunables

Edit `/etc/default/llama-cdsp` (a plain systemd `EnvironmentFile`), then:

```bash
systemctl daemon-reload && systemctl restart llama-cdsp
```

| Key | Default | Note |
|---|---|---|
| `LLAMA_MODEL` | `Qwen3.5-4B-Q4_K_M.gguf` | switch to `smollm2-135m-q4_k_m.gguf` for a 101 MB model |
| `LLAMA_CTX` | 8192 | |
| `LLAMA_SLOTS` | 1 | keep at 1; the FastRPC session is shared and serialized |
| `LLAMA_THREADS` | 8 | matches the CPU service for a fair A/B |
| `LLAMA_PORT` | 8082 | must not collide with 8081 |
| `GGML_CDSP_MIN_ELEMS` / `MAX_ELEMS` | 100000 / 3000000 | Stage 6 recommendation for the 4B model |
| `GGML_CDSP_DEBUG` | unset | **2 log lines per offloaded MUL_MAT** — enable once to prove offload, then disable |
| `GGML_CDSP_BATCH`, `_PROFILE`, `_SPLIT_STATS`, `_CACHE_MB` | unset | opt-in diagnostics |

The deploy script keeps an existing `/etc/default/llama-cdsp` unless
`--force-env` is passed, so on-device tuning survives redeploys.

## A/B smoke test

```bash
python3 scripts/device/llama_cdsp_smoke.py                       # 8081 vs 8082
python3 scripts/device/llama_cdsp_smoke.py --ports 8082 --tokens 32
```

It issues the same deterministic request (`seed 42`, `temperature 0`) to each
port, prints decoded text, decode tokens/s, prompt tokens/s and the ratio, and
checks `remoteproc1/state` before and after each request.

**Interpretation caveats** (learned the hard way in Stages 7–8):

* Both servers run simultaneously on 8 cores and contend for CPU. A single
  A→B pair is not a rigorous benchmark; use ABBA order for any claim.
* Thermal throttling makes "first run faster". Fix seed/temperature and
  alternate the order.
* Outputs are **not bit-exact**: int8 quantization flips narrow logit argmaxes,
  so greedy text diverges from the CPU service after ~10 tokens on the 135M
  model. This is a known correctness limitation, not a service bug.

Measured on 2026-11-08 (24 tokens, both services live):
CPU **3.57 t/s**, cDSP **3.19 t/s** = **89.4%**, consistent with Stage 6's
~94%. See `evidence/stage10-service-smoke.txt`.

## Verification performed (Stage 10)

1. `llama-server --list-devices` with the unit's environment prints
   `cdsp: Redmi K20 Pro cDSP (Hexagon 690)`.
2. `--list-devices` and the running process both show
   `libggml_cdsp_preload.so`, `libggml-cdsp.so`, `libadsprpc.so` loaded.
3. `journalctl` shows `[cdsp-preload] registered cDSP backend ...` and
   `remote_handle64_open: Successfully opened ... on domain 3`.
4. With a temporary `GGML_CDSP_DEBUG=1` drop-in, a 4-token request produced
   **55 new `[ggml-cdsp] offloaded MUL_MAT M=1 N=1024 K=2560` lines** and
   `cache hits=66 misses=11 bytes=28835840`. The drop-in was removed and the
   service restarted in its default configuration afterwards.
5. HTTP `/health` = 200, WebUI `/` = 200 from the PC, 8081 still 200,
   `remoteproc1/state = running`.

## Troubleshooting

* **`open_device_node failed for domain ID 0 ... /dev/fastrpc-adsp-secure`,
  `apps_std_FILE_get`, `mod_table.c` errors** — harmless noise from probing the
  ADSP domain, which is deliberately disabled in the stage-2 DTB. The line that
  matters is `Successfully opened ... on domain 3` (CDSP).
* **`Compute error` / `0.0 t/s` / `remoteproc1: crashed`** — cDSP watchdog.
  `systemctl restart llama-cdsp` will NOT fix it. Run `systemctl reboot`
  (**never kexec**), wait ~50 s, then start the service again. Repeated long
  generations (8+) can trigger this even with clean exits.
* **Service loops in `failed` state** — expected after 3 failed starts; check
  `journalctl -u llama-cdsp` and the remoteproc state, fix the root cause,
  then `systemctl reset-failed llama-cdsp && systemctl start llama-cdsp`.
* **No offload happening** — temporarily set `GGML_CDSP_DEBUG=1` in the env
  file and grep the journal for `[ggml-cdsp] offloaded`. Remember to turn it
  off again (it is extremely verbose and perturbs timing).
* **Want to change model/threads** — edit `/etc/default/llama-cdsp`, not the
  unit; then `daemon-reload` + `restart`.

## Scope

This stage added a systemd unit, an env file, a deploy script and a smoke-test
script. It did **not** rebuild `llama-server`, `libggml-cdsp.so`, the preload
shim or the DSP skel; did not modify `llama-4b.service`; did not change
firmware, DTB, services or the cDSP state; and did not reboot the device.
