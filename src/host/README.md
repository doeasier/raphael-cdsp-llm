# src/host — AP-side GEMM correctness + timing test (Stage 1)

`host_test.c` generates random int8 `A(M×K)`, `B(K×N)`, dispatches the GEMM to the
cDSP over FastRPC, and verifies `C == reference(A·B)` on the host. Also reports
wall-clock DSP compute time.

`host_test_cached.c` is the **Stage 5** test of the weight-cache API: it uploads
`B` once (`gemm_upload`), then calls `gemm_compute_cached` (only `A` shipped) and
verifies the result. This proves the transport fix end-to-end.

Build + run on the device (the deploy script does this for you):

```bash
# on device, after the skel is installed at /usr/lib/rfsa/cdsp
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    host_test.c gen/cdsp_gemm_stub.c -o host_test \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
gcc -O2 -I. -Igen -I/usr/local/include/fastrpc \
    host_test_cached.c gen/cdsp_gemm_stub.c -o host_test_cached \
    -L/usr/local/lib -ladsprpc -lpthread -ldl
./host_test 32 128 128                 # M N K  (one-shot)
./host_test_cached 1 11008 4096         # M N K  (weight-cache)
```

Exit 0 + `host: PASS - all N elements verified correct.` means the DSP kernel is
bit-exact against the int8 reference. Real model-layer sizes also pass:
`./host_test 1 4096 4096`, `./host_test 1 11008 4096`, `./host_test 32 4096 4096`.
