# Verification evidence (Stage 6 + Stage 7)

Raw output captured on the Redmi K20 Pro over USB-NCM. Stage 7 uses the
final `libgemm_skel.so` (three-path HVX kernel **plus batched compute**, md5
`a9cceed07e9493e020b53e5c7858cb8e`) and the matching `libggml-cdsp.so`.

| File | Command | Result |
|---|---|---|
| `stage6-dsp-regression.txt` | `python3 scripts/device/deploy_cdsp_gemm.py` | 9/9 `host_test` + `host_test_cached` PASS, 7/7 `bench_cached` PASS (incl. warm-up). No failures. |
| `stage6-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | `PASS`, `SCHED_TEST PASS`, `CACHE_TEST PASS`, `Q4_K_TEST PASS`, `Q8_0_TEST PASS`, preload shim built |
| `stage7-dsp-regression.txt` | same script, Stage 7 build | **10/10 PASS** — the extra case is `host_test_batch` (`BATCH_TEST PASS`, 1056 elements); 7/7 `bench_cached` PASS |
| `stage7-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | **5/5 PASS** — unchanged from Stage 6, confirming the batched backend is backward compatible |
| `stage7-ab/` | `llama_diag.py --seed 42 --temp 0` | deterministic scalar-vs-batch logs; see the thermal-bias analysis below |

## Stage 6 canonical HVX numbers

Quoted in `docs/next-steps.md` §2, from the `bench_cached` section of
`stage6-dsp-regression.txt`:

```
[warm-up] BENCH 128x512x512 iters=200  2.0851 ms/call  32.19 GMAC/s  PASS
BENCH 1x32x4                          0.3398 ms/call
BENCH 1x32x2560                       0.3481 ms/call   0.47 GMAC/s  PASS
BENCH 1x1024x2560                     1.7426 ms/call   3.01 GMAC/s  PASS
BENCH 1x4096x4096                     9.2042 ms/call   3.65 GMAC/s  PASS
BENCH 128x512x512                     2.1879 ms/call  30.67 GMAC/s  PASS
BENCH 32x4096x4096                   94.2803 ms/call  11.39 GMAC/s  PASS
```

## Stage 7 batched-RPC numbers

From `stage7-dsp-regression.txt`:

```
host: uploaded 2 weights -> handles 1 / 2
host: scalar  2 RPCs = 2.505 ms/pair
host: batched 1 RPC  = 1.962 ms/pair  (21.6% less)
host: BATCH_TEST PASS - 1056 elements verified correct.
```

Clean-state batch smoke (24 tokens, fixed seed, greedy decoding):

```
[cdsp-split] calls=208 nodes=572 mul_mat=286 splits_with_>1_mm=78
[cdsp-split] mm-per-split histogram: 0=0 1=130 2=78 3+=0
[cdsp-batch] splits=78 ops=156 rpcs_saved=78
```

Greedy-decoding A/B produced **byte-identical generated text** between
batch-on and batch-off (only the timing line differs).

## Why `stage7-ab/` does not show a speedup — read before benchmarking

`stage7-ab/off{1,2}.txt` / `on{1,2}.txt` are paired scalar/batch runs with a
fixed RNG seed. Generation rates were 3.5, 3.4, 3.3, 3.2 t/s in run order —
strictly monotonic **regardless of mode**. An ABBA-order follow-up (on 3.4,
off 3.2) confirmed this is thermal throttling on an uncooled phone: whichever
mode runs first wins ~0.1–0.2 t/s. The last two ABBA runs hit the cDSP watchdog
(`0.0 t/s`, `Error: Compute error`), which is why `llama_diag.py` now pre-checks
`remoteproc1/state`.

Conclusion: the batched encoding is correct and measurably cheaper at the RPC
layer, but its ~2.4 ms/token ceiling sits inside the thermal noise of a
~280 ms/token decode. Never interpret a single back-to-back A/B pair as a
performance result on this device.

End-to-end llama.cpp runs are driven by `scripts/device/llama_diag.py`; the full
device log stays at `/data/local/tmp/diag_<mode>.log` on the phone and a host
copy in `tmp/diag_<mode>.host.log` (gitignored). The recommended command and
the complete env-var table are in `docs/llama-integration.md`.

> Two distinct DSP-crash causes are documented here. Stage 6: `kill -9
> llama-cli` mid-FastRPC left `remoteproc1=crashed`. Stage 7: 8+ consecutive
> long runs with **clean exits** tripped the cDSP watchdog
> (`SFR Init: wdog or kernel error suspected`). In both cases sysfs
> `stop`/`start` does not recover it — only `systemctl reboot` (never `kexec`),
> ~50 s over USB-NCM. The kernel passes the full suite cleanly afterwards.
