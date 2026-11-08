#!/usr/bin/env python3
"""Run llama-cli on the K20 Pro with the cDSP ggml backend and collect a
compact diagnostic result (generated text + offload/cache stats + timing).

Modes:
  normal   : cache + int8 + cDSP
  nocache  : no cache + int8 + cDSP   -> isolates STALE cache entries
  hostf32  : exact host F32 GEMM      -> isolates int8 / DSP / cache entirely
  cpu      : no LD_PRELOAD            -> CPU baseline

Stage 7 additions: `--env` injects arbitrary switches (e.g.
GGML_CDSP_BATCH=1 GGML_CDSP_SPLIT_STATS=1), `--seed` / `--temp` make sampling
deterministic for a fair A/B, and a pre-flight check refuses to benchmark a
crashed cDSP remoteproc (which would otherwise report a misleading 0.0 t/s).

Stage 8 additions: `--model` selects another GGUF (a bare filename resolved
under /root/models, avoiding MSYS path rewriting), `--cache-mb` overrides the
DSP weight-cache size and `--threads` sets llama-cli's -t.
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


def resolve_model(spec):
    """Accept a bare GGUF filename (resolved under /root/models).

    Git Bash on Windows silently rewrites a leading '/root/...' argument into
    an MSYS Windows path (e.g. C:/Program Files/.../root/models/...), which
    then reaches llama-cli as garbage. A bare filename avoids that conversion.
    """
    if spec.startswith("/"):
        return spec                       # already a device-absolute path
    if os.path.isabs(spec):
        raise SystemExit(
            "model path was rewritten by MSYS (%s); pass a bare filename "
            "like 'smollm2-135m-q4_k_m.gguf' instead" % spec)
    return "/root/models/" + spec


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
    ap.add_argument("--env", default="",
                    help="extra KEY=VAL pairs, e.g. 'GGML_CDSP_BATCH=1'")
    ap.add_argument("--seed", type=int, default=-1,
                    help="fixed RNG seed for deterministic A/B (-1 = random)")
    ap.add_argument("--temp", default=None,
                    help="sampling temperature, e.g. 0 for greedy decoding")
    ap.add_argument("--model", default=os.path.basename(MDL),
                    help="GGUF filename in /root/models, or a device-absolute path")
    ap.add_argument("--cache-mb", default=None,
                    help="override GGML_CDSP_CACHE_MB (default 256 on device)")
    ap.add_argument("--threads", type=int, default=4,
                    help="llama-cli -t value")
    a = ap.parse_args()
    model = resolve_model(a.model)

    sampling = ""
    if a.seed >= 0:
        sampling += " -s %d" % a.seed
    if a.temp is not None:
        sampling += " --temp %s" % a.temp

    log = "/data/local/tmp/diag_%s.log" % a.mode
    extra = (MODES[a.mode] + " " + a.env).strip()
    cache_env = ("GGML_CDSP_CACHE_MB=%s " % a.cache_mb) if a.cache_mb else ""

    if MODES[a.mode] == "__CPU__":
        envs = ("LD_LIBRARY_PATH=%s:/usr/local/lib %s"
                % (BIN, a.env))
    else:
        envs = ("LD_LIBRARY_PATH=%s:/usr/local/lib "
                "LD_PRELOAD=%s/libggml_cdsp_preload.so "
                "GGML_CDSP_BACKEND_PATH=%s/libggml-cdsp.so "
                "GGML_CDSP_MAX_ELEMS=%s GGML_CDSP_MIN_ELEMS=%s %s%s %s"
                % (BIN, BACKEND, BACKEND, a.max_elems, a.min_elems, cache_env,
                   "" if a.no_debug else "GGML_CDSP_DEBUG=1", extra))

    cmd = ("cd %s && %s timeout %d ./llama-cli -m %s -p '%s' "
           "-n %d -st -c 512 -t %d --no-warmup%s > %s 2>&1; echo RC=$?; cat %s"
           % (BIN, envs, a.timeout - 30, model, a.prompt.replace("'", "'\\''"),
              a.tokens, a.threads, sampling, log, log))

    c = k.connect()
    try:
        # A crashed cDSP produces "Compute error" and 0.0 t/s, which is easy to
        # mistake for a performance regression. Refuse to run in that state.
        state_out, _, _ = k.run(
            c, "cat /sys/class/remoteproc/remoteproc1/state 2>/dev/null",
            timeout=20)
        state = state_out.strip().splitlines()[-1] if state_out.strip() else "unknown"
        if state != "running":
            print("DSP_NOT_RUNNING: remoteproc1 state=%s" % state)
            print("Recover with 'systemctl reboot' (NEVER kexec); sysfs "
                  "stop/start does not clear a crashed PD.")
            return 2
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
    print("MODEL      :", model)
    print("OFFLOADS   :", n_off, " ERRORS:", n_err)
    print("CACHE(last): hits=%s misses=%s bytes=%s" % last_cache)
    print("TIMING     :", timing)
    print("--- OUTPUT ---")
    print(body)
    print("--- device log:", log, " host copy:", hostlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
