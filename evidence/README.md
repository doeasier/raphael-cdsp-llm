# Verification evidence (Stage 6)

Raw output captured on the Redmi K20 Pro over USB-NCM, after the final
`libgemm_skel.so` (three-path HVX kernel, md5 `8f3e5decf1f9bb65db7cfb7fb493cffe`)
and the final `libggml-cdsp.so` were installed.

| File | Command | Result |
|---|---|---|
| `stage6-dsp-regression.txt` | `python3 scripts/device/deploy_cdsp_gemm.py` | 9/9 `host_test` + `host_test_cached` PASS, 7/7 `bench_cached` PASS (incl. warm-up). No failures. |
| `stage6-ggml-regression.txt` | `python3 scripts/device/deploy_ggml_cdsp.py` | `PASS`, `SCHED_TEST PASS`, `CACHE_TEST PASS`, `Q4_K_TEST PASS`, `Q8_0_TEST PASS`, preload shim built |

The canonical HVX numbers quoted in `docs/next-steps.md` §2 come from the
`bench_cached` section of `stage6-dsp-regression.txt`:

```
[warm-up] BENCH 128x512x512 iters=200  2.0851 ms/call  32.19 GMAC/s  PASS
BENCH 1x32x4                          0.3398 ms/call
BENCH 1x32x2560                       0.3481 ms/call   0.47 GMAC/s  PASS
BENCH 1x1024x2560                     1.7426 ms/call   3.01 GMAC/s  PASS
BENCH 1x4096x4096                     9.2042 ms/call   3.65 GMAC/s  PASS
BENCH 128x512x512                     2.1879 ms/call  30.67 GMAC/s  PASS
BENCH 32x4096x4096                   94.2803 ms/call  11.39 GMAC/s  PASS
```

End-to-end llama.cpp runs are driven by `scripts/device/llama_diag.py`; its
full device log is left at `/data/local/tmp/diag_<mode>.log` on the phone and a
host copy in `tmp/diag_<mode>.host.log` (gitignored). The measured tokens/s
table and the exact recommended command are in `docs/llama-integration.md` §4.

> One earlier `deploy_cdsp_gemm.py` run ended with `open failed rc=-1` because
> a `kill -9 llama-cli` had left the cDSP in `crashed` state
> (`remoteproc1`). A `systemctl reboot` restored it; the kernel itself passes
> the complete suite cleanly afterwards. See the troubleshooting tables in
> `docs/llama-integration.md` and `docs/reproduce.md`.
