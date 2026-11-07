#!/usr/bin/env python3
"""Run llama-cli on the K20 Pro with the cDSP ggml backend and collect a
compact diagnostic result (generated text + offload/cache stats + timing).

Used to attribute the "garbled generation" issue in Stage 6:
  normal   : cache + int8 + cDSP
  nocache  : no cache + int8 + cDSP   -> isolates STALE cache entries
  hostf32  : exact host F32 GEMM      -> isolates int8 / DSP / cache entirely
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k20_ssh as k  # noqa: E402

BIN = "/root/llama.cpp-0.9.0/build/bin"
MDL = "/root/models/Qwen3.5-4B-Q4_K_M.gguf"
BACKEND = "/data/local/tmp/ggml-backend"

MODES = {
    "normal":  "",
    "nocache": "GGML_CDSP_NO_CACHE=1",
    "hostf32": "GGML_CDSP_HOST_F32=1",
    "cpu":     "__CPU__",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--max-elems", default="3000000")
    ap.add_argument("--min-elems", default="0")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--no-debug", action="store_true",
                    help="omit GGML_CDSP_DEBUG for a fair timing measurement")
    a = ap.parse_args()

    log = "/data/local/tmp/diag_%s.log" % a.mode
    extra = MODES[a.mode]

    if extra == "__CPU__":
        envs = ("LD_LIBRARY_PATH=%s:/usr/local/lib" % BIN)
    else:
        envs = ("LD_LIBRARY_PATH=%s:/usr/local/lib "
                "LD_PRELOAD=%s/libggml_cdsp_preload.so "
                "GGML_CDSP_BACKEND_PATH=%s/libggml-cdsp.so "
                "GGML_CDSP_MAX_ELEMS=%s GGML_CDSP_MIN_ELEMS=%s %s %s"
                % (BIN, BACKEND, BACKEND, a.max_elems, a.min_elems,
                   "" if a.no_debug else "GGML_CDSP_DEBUG=1", extra))

    cmd = ("cd %s && %s timeout %d ./llama-cli -m %s -p '%s' "
           "-n %d -st -c 512 -t 4 --no-warmup > %s 2>&1; echo RC=$?; cat %s"
           % (BIN, envs, a.timeout - 30, MDL, a.prompt.replace("'", "'\\''"),
              a.tokens, log, log))

    c = k.connect()
    try:
        out, err, st = k.run(c, cmd, timeout=a.timeout)
    finally:
        c.close()
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    logdir = os.path.join(ROOT, "tmp")
    os.makedirs(logdir, exist_ok=True)
    hostlog = os.path.join(logdir, "diag_%s.host.log" % a.mode)
    open(hostlog, "w", encoding="utf-8", errors="replace").write(out)

    # compact report
    clean = re.sub(r"[\r\x08]", "", out)
    clean = re.sub(r"\[ggml-cdsp\] offloaded MUL_MAT[^\n]*\n", "", clean)
    # generation is between the prompt line and the timing line
    m = re.search(r"\[ Prompt: ([^\]]+)\]", clean)
    timing = m.group(1) if m else "(no timing line)"
    idx = clean.find("> " + a.prompt)
    body = clean[idx:] if idx >= 0 else clean
    body = re.sub(r"\[ggml-cdsp\][^\n]*\n", "", body)
    body = body.replace("[ Prompt: " + timing + " ]", "")
    body = "\n".join(l for l in body.splitlines() if l.strip())[:1500]

    n_off = len(re.findall(r"offloaded MUL_MAT", out))
    n_err = len(re.findall(r"graph_compute|Compute error|failed with error", out))
    cache = re.findall(r"cache hits=(\d+) misses=(\d+) bytes=(\d+)", out)
    last_cache = cache[-1] if cache else ("0", "0", "0")
    print("MODE       :", a.mode)
    print("OFFLOADS   :", n_off, " ERRORS:", n_err)
    print("CACHE(last): hits=%s misses=%s bytes=%s" % last_cache)
    print("TIMING     :", timing)
    print("--- OUTPUT ---")
    print(body)
    print("--- device log:", log, " host copy:", hostlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
