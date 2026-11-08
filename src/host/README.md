# src/host — AP-side GEMM correctness + throughput tests (Stage 1/5/6/7)

`host_test.c` generates random int8 `A(M×K)`, `B(K×N)`, dispatches the GEMM to the
cDSP over FastRPC, and verifies `C == reference(A·B)` on the host. Also reports
wall-clock DSP compute time.

`host_test_cached.c` is the **Stage 5** test of the weight-cache API: it uploads
`B` once (`gemm_upload`), then calls `gemm_compute_cached` (only `A` shipped) and
verifies the result. This proves the transport fix end-to-end.

`host_test_batch.c` is the **Stage 7** test of the batched FastRPC encoding. It
uploads two DIFFERENT weights (1×1024×2560 and 1×32×2560), computes them as two
scalar calls and as ONE batched call (`dims[0] = -2`), verifies every element
against an int32 scalar GEMM (so the A/C concat offsets are exercised), and
reports the 2-RPC → 1-RPC latency. Typical result: `2.505 → 1.962 ms (−21.6%)`,
`BATCH_TEST PASS - 1056 elements verified correct.`

`bench_cached.c` is the **Stage 6** HVX microbenchmark: upload once, then call
`gemm_compute_cached` repeatedly with warm-ups excluded, reporting ms/call and
effective GMAC/s. A `1×32×4` shape measures the bare FastRPC round-trip. Every
run also spot-checks one output element against a scalar reference.

Build + run on the device (the deploy script does this for you):

```bash
# on device, after the skel is installed at /usr/lib/rfsa/cdsp
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    host_test.c gen/cdsp_gemm_stub.c -o host_test \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    host_test_cached.c gen/cdsp_gemm_stub.c -o host_test_cached \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    host_test_batch.c gen/cdsp_gemm_stub.c -o host_test_batch \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    bench_cached.c gen/cdsp_gemm_stub.c -o bench_cached \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
./host_test 32 128 128                 # M N K  (one-shot)
./host_test_cached 1 11008 4096         # M N K  (weight-cache)
./host_test_batch 30                    # iterations (batched RPC)
./bench_cached 128 512 512 200         # M N K iters (warm-up / throughput)
```

Exit 0 + `host: PASS - all N elements verified correct.` means the DSP kernel is
bit-exact against the int8 reference. Real model-layer sizes also pass:
`./host_test 1 4096 4096`, `./host_test 1 11008 4096`, `./host_test 32 4096 4096`.

> The cDSP ramps its clock under sustained load. Run a 200-iteration warm-up
> shape before recording any `bench_cached` numbers; cold-clock measurements
> are 30–50% pessimistic. `deploy_cdsp_gemm.py` does this automatically.
