# Verification evidence (Stage 6 + Stage 7 + Stage 8 + Stage 9 + Stage 10)

Raw output captured on the Redmi K20 Pro over USB-NCM. Stage 7/8 use the
final `libgemm_skel.so` (three-path HVX kernel **plus batched compute**, md5
`a9cceed07e9493e020b53e5c7858cb8e`) and the matching `libggml-cdsp.so`.

| File | Command | Result |
|---|---|---|
| `stage6-dsp-regression.txt` | `python3 scripts/device/deploy_cdsp_gemm.py` | 9/9 `host_test` + `host_test_cached` PASS, 7/7 `bench_cached` PASS (incl. warm-up). No failures. |
| `stage6-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | `PASS`, `SCHED_TEST PASS`, `CACHE_TEST PASS`, `Q4_K_TEST PASS`, `Q8_0_TEST PASS`, preload shim built |
| `stage7-dsp-regression.txt` | same script, Stage 7 build | **10/10 PASS** — the extra case is `host_test_batch` (`BATCH_TEST PASS`, 1056 elements); 7/7 `bench_cached` PASS |
| `stage7-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | **5/5 PASS** — unchanged from Stage 6, confirming the batched backend is backward compatible |
| `stage7-ab/` | `llama_diag.py --seed 42 --temp 0` | deterministic scalar-vs-batch logs; see the thermal-bias analysis below |
| `stage8-dsp-regression.txt` | `python3 scripts/device/deploy_cdsp_gemm.py` (Stage 8 build) | **21 PASS, no failures** — now also builds `bench_regbuf` and prints `REGBUF_TEST PASS` |
| `stage8-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | **5/5 PASS** with the `GGML_CDSP_PROFILE` backend, all previous tests still green |
| `stage8/` | `llama_diag.py --model smollm2-135m-q4_k_m.gguf` | small-model CPU/cDSP timing, greedy + sampled correctness, PROFILE/SPLIT_STATS logs |
| `stage9-scheduler-excerpts.txt` | read-only SSH source inspection (no `llama-cli` run) | ggml 0.9.0 split-scheduler excerpts with exact line numbers, op inventory and upstream `ggml-hexagon` code-volume measurement |
| `stage10-service-smoke.txt` | `deploy_llama_cdsp.py`, `llama_cdsp_smoke.py`, HTTP requests | cDSP service on :8082 deployed; 55 `offloaded MUL_MAT` lines prove offload; CPU 3.57 vs cDSP 3.19 t/s (89.4%); PC-side WebUI/health 200 |

## Stage 10 — cDSP HTTP service (application testing)

Unlike Stages 6–9, this is a **long-running service**, not a one-shot CLI run.
[`stage10-service-smoke.txt`](stage10-service-smoke.txt) records the deploy, the
`--list-devices` probe, the temporary `GGML_CDSP_DEBUG=1` offload proof (and its
removal), the A/B results and PC-side reachability. The unit/env templates,
deploy script and usage notes are in
[`docs/llama-cdsp-service.md`](../docs/llama-cdsp-service.md). The 89.4% ratio
is an application-level demonstration, **not** a rigorous benchmark: both
servers contend for the same 8 cores and Stage 7 showed a strong thermal
ordering bias.

## Stage 9 — assessment only

Stage 9 produced **no runtime logs on purpose**: it is a source-level
feasibility study (task #74), and the only artefact is the annotated excerpt
file above plus
[`docs/whole-layer-fusion-feasibility.md`](../docs/whole-layer-fusion-feasibility.md).
All performance numbers quoted in that document come from the Stage 8 logs
already indexed here; no device state, kernel, skel or `src/` file changed.

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

## Stage 8 — small-model benchmark (SmolLM2-135M-Q4_K_M)

Model: 101 MB GGUF, sha256
`3dac8a6535e754df3da638794da83508b748bb31c4100499a392173d17cb695a`,
30 layers / emb 576 / FFN 1536 / vocab 49152. All runs use
`--seed 42 --temp 0 --no-debug --tokens 64` unless noted, on a clean device
state after `systemctl reboot`.

| File | Configuration | Generation |
|---|---|---|
| `stage8/final_cpu.txt` | CPU only | **81.2 t/s** (123.2 prompt) |
| `stage8/final_cdsp.txt` | `MIN=500000 MAX=2000000` (FFN only) | **23.8 t/s** (29%) |
| `stage8/final_cdsp_full.txt` | `MIN=0 MAX=0` (full offload) | 13.4 t/s (17%) |
| `stage8/correct_{cpu,cdsp}.txt` | `--temp 0.7`, same seed | both fluent, different text |
| `stage8/greedy_{cpu,cdsp}.txt` | `--temp 0` | first ~10 tokens identical, then diverge |
| `stage8/greedy24_{cpu,normal,nocache}.txt` | 24 tokens | `normal == nocache` (no stale cache); `nocache` = 1.4 t/s, i.e. the cache is worth 17.6× |

Full offload nevertheless keeps every weight resident —
`cache hits=1984 misses=31 bytes=44015616` (**42 MB, zero thrash**) — so the
"weights fit → cDSP wins" hypothesis is falsified cleanly.

Per-stage attribution (`GGML_CDSP_PROFILE=1`, 1040 ops, best config):

```
[cdsp-prof] ops=1040 total=1945.6 ms (1.871 ms/op) | weight=709.7 (36.5%)
            quant_A=62.3 (3.2%) rpc=1157.1 (59.5%) rescale=16.4 (0.8%)
[cdsp-split] calls=2015 nodes=2925 mul_mat=2015 splits_with_>1_mm=0
[cdsp-split] mm-per-split histogram: 0=0 1=2015 2=0 3+=0
```

The backend accounts for ~19.3 ms of the measured 43.3 ms/token; the remaining
~24 ms/token is CPU↔ACCEL split synchronization (zero-offload control:
71.0 t/s vs CPU 80.6 t/s). With zero multi-matmul splits, Stage 7 batching has
nothing to merge (`BATCH=1`: 19.0 vs 19.4 t/s).

RPC buffer strategies (`stage8-dsp-regression.txt`):

```
REGBUF mode=0 HEAP   | 1.0173 ms/call | +0.0%   | verify=PASS
REGBUF mode=1 RPCMEM | 1.0334 ms/call | +1.6%   | verify=PASS
REGBUF mode=2 MAPPED | MMAP_UNSUPPORTED rcA=20 rcC=20
REGBUF_TEST PASS
```

An earlier run measured RPCMEM at +36.6% (1.2802 vs 0.9374 ms), i.e. the
DMA-BUF path is noisy but never faster. `rc=20 = 0x14 = AEE_EUNSUPPORTED`;
dmesg shows `qcom,fastrpc-cb ...: mem mmap error`. Persistent mapping is a
driver capability gap on this kernel, not an argument error.

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
