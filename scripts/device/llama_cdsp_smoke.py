#!/usr/bin/env python3
"""Application-level smoke test / A-B comparison for llama-cdsp.service.

Runs the SAME deterministic completion request against the CPU reference
service (llama-4b, port 8081) and the cDSP service (port 8082), then prints
output, decode speed and the cDSP/CPU ratio. Requests are issued on the device
over USB-NCM SSH so no extra host-side networking is required.

Usage:
  python3 scripts/device/llama_cdsp_smoke.py
  python3 scripts/device/llama_cdsp_smoke.py --tokens 32 --ports 8081,8082
  python3 scripts/device/llama_cdsp_smoke.py --ports 8082   # cDSP only

Notes:
  * Both servers run at once on an 8-core phone and contend for CPU; a single
    A-B pair is NOT a rigorous benchmark (Stage 7 found a strong "first run is
    faster" thermal bias). Use ABBA order for any performance claim.
  * Repeated long runs can trip the cDSP watchdog -> `remoteproc1: crashed`.
    The script checks remoteproc state before and after each request. Recovery
    is `systemctl reboot` (never kexec), not `systemctl restart llama-cdsp`.
"""
import argparse
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import k20_ssh as k  # noqa: E402


def run(c, cmd, timeout=120):
    out, err, rc = k.run(c, cmd, timeout)
    return out.strip(), err.strip(), rc


def dsp_state(c):
    out, _, _ = run(c, "cat /sys/class/remoteproc/remoteproc1/state 2>/dev/null")
    return out or "unknown"


def completion(c, port, prompt, tokens, seed, temp, timeout):
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": tokens,
        "temperature": temp,
        "seed": seed,
        "stream": False,
    }, separators=(",", ":"))
    cmd = ("curl -s --max-time %d http://127.0.0.1:%d/completion "
           "-H 'Content-Type: application/json' -d %s"
           % (timeout, port, shlex.quote(payload)))
    out, err, rc = run(c, cmd, timeout + 30)
    if rc != 0 or not out:
        return {"port": port, "error": "curl failed: %s %s" % (err[:200], out[:200])}
    try:
        d = json.loads(out)
    except ValueError:
        return {"port": port, "error": "non-JSON reply: %s" % out[:200]}
    t = d.get("timings", {})
    return {
        "port": port,
        "content": d.get("content", ""),
        "tokens": d.get("tokens_predicted", 0),
        "decode_tps": t.get("predicted_per_second", 0.0),
        "prompt_tps": t.get("prompt_per_second", 0.0),
        "model": d.get("model", ""),
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ports", default="8081,8082",
                    help="comma-separated ports (8081=CPU, 8082=cDSP)")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()

    ports = [int(p) for p in a.ports.split(",") if p.strip()]
    c = k.connect()
    results = []
    try:
        print("cDSP remoteproc before: %s" % dsp_state(c))
        for port in ports:
            out, _, _ = run(c, "curl -s -o /dev/null -w '%%{http_code}' "
                               "http://127.0.0.1:%d/health" % port)
            print("\n--- port %d (health=%s) ---" % (port, out))
            r = completion(c, port, a.prompt, a.tokens, a.seed, a.temp, a.timeout)
            results.append(r)
            if r["error"]:
                print("  ERROR:", r["error"])
            else:
                print("  model : %s" % r["model"])
                print("  output: %r" % r["content"])
                print("  decode: %d tokens  %.2f t/s   | prompt %.2f t/s"
                      % (r["tokens"], r["decode_tps"], r["prompt_tps"]))
            print("  cDSP remoteproc after: %s" % dsp_state(c))

        ok = [r for r in results if not r["error"]]
        if len(ok) == 2:
            cpu = next((r for r in ok if r["port"] == 8081), None)
            dsp = next((r for r in ok if r["port"] == 8082), None)
            if cpu and dsp and cpu["decode_tps"]:
                print("\ncDSP / CPU decode ratio: %.1f%%" %
                      (100.0 * dsp["decode_tps"] / cpu["decode_tps"]))
                print("outputs identical: %s" %
                      (dsp["content"] == cpu["content"]))
        return 0 if len(ok) == len(ports) else 1
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())
