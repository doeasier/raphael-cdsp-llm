#!/usr/bin/env python3
"""Install / verify llama-cdsp.service on the Redmi K20 Pro (USB-NCM SSH).

New service for APPLICATION TESTING of the cDSP backend:

    systemctl start llama-cdsp      # port 8082 (default), cDSP ACCEL backend
    systemctl start llama-4b        # port 8081, stock CPU-only  (untouched)

The device's llama.cpp 0.9.0 binary is NOT recompiled or replaced. The cDSP
backend is attached at runtime with LD_PRELOAD (GGML_BACKEND_DL=OFF means the
ordinary GGML_BACKEND_PATH hook never runs). See llama-cdsp.service comments.

What this script does:
  1. Precheck: binary, plugin .so, preload shim, DSP skel, cdsprpcd, cdsp
     remoteproc state, free port, and a real `--list-devices` probe that must
     print the "cdsp:" ACCEL device.
  2. Upload the unit + env file (CRLF-normalized; existing env file is kept
     unless --force-env is passed, so on-device tuning survives redeploys).
  3. daemon-reload, start (or restart), wait for the HTTP health endpoint.
  4. Print status, the backend-registration journal lines and listening ports.

It NEVER writes to /etc/systemd/system/llama-4b.service and never reboots.

Usage:
  python3 scripts/device/deploy_llama_cdsp.py                # install + start
  python3 scripts/device/deploy_llama_cdsp.py --status       # inspect only
  python3 scripts/device/deploy_llama_cdsp.py --restart      # restart only
  python3 scripts/device/deploy_llama_cdsp.py --no-start     # install only
  python3 scripts/device/deploy_llama_cdsp.py --enable       # also autostart
  python3 scripts/device/deploy_llama_cdsp.py --force-env    # reset tunables
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import k20_ssh as k  # noqa: E402

UNIT_SRC = os.path.join(HERE, "llama-cdsp.service")
ENV_SRC = os.path.join(HERE, "llama-cdsp.env")
UNIT_DST = "/etc/systemd/system/llama-cdsp.service"
ENV_DST = "/etc/default/llama-cdsp"
STAGE = "/data/local/tmp"

BIN = "/root/llama.cpp-0.9.0/build/bin/llama-server"
BACKEND = "/data/local/tmp/ggml-backend"
PRELOAD = BACKEND + "/libggml_cdsp_preload.so"
PLUGIN = BACKEND + "/libggml-cdsp.so"
SKEL = "/usr/lib/rfsa/cdsp/libgemm_skel.so"

LIST_DEVICES = (
    "cd /root/llama.cpp-0.9.0 && "
    "LD_LIBRARY_PATH=/root/llama.cpp-0.9.0/build/bin:/usr/local/lib "
    "LD_PRELOAD=%s GGML_CDSP_BACKEND_PATH=%s "
    "./build/bin/llama-server --list-devices 2>&1" % (PRELOAD, PLUGIN)
)


def run(c, cmd, timeout=120):
    out, err, rc = k.run(c, cmd, timeout)
    return out.strip(), err.strip(), rc


def check(c, label, cmd, needle=None, timeout=120):
    out, err, rc = run(c, cmd, timeout)
    text = (out + "\n" + err).strip()
    ok = rc == 0 and (needle is None or needle in text)
    print("  [%s] %-38s %s" % ("OK" if ok else "FAIL", label, text[:150].replace("\n", " | ")))
    return ok


def precheck(c, port):
    print("[1/4] precheck")
    items = [
        ("llama-server binary", "test -x %s && echo present" % BIN, "present"),
        ("preload shim", "test -f %s && echo present" % PRELOAD, "present"),
        ("cDSP backend plugin", "test -f %s && echo present" % PLUGIN, "present"),
        ("DSP skel on search path", "test -f %s && echo present" % SKEL, "present"),
        ("cdsprpcd active", "systemctl is-active cdsprpcd", "active"),
        ("cdsp remoteproc running", "cat /sys/class/remoteproc/remoteproc1/state", "running"),
        ("backend registers as ACCEL device", LIST_DEVICES, "cdsp:"),
        ("port %d free" % port,
         "ss -lnt | grep -q ':%d ' && echo busy || echo free" % port, "free"),
        ("CPU reference service intact",
         "systemctl is-active llama-4b.service", "active"),
    ]
    results = [check(c, label, cmd, needle) for label, cmd, needle in items]
    return all(results)


def status(c):
    print("\n=== llama-cdsp status ===")
    out, _, _ = run(c, "systemctl status llama-cdsp --no-pager -l | head -22")
    print(out)
    print("\n=== backend registration / server startup ===")
    out, _, _ = run(c, "journalctl -u llama-cdsp --no-pager -o cat | "
                       "grep -iE 'cdsp-preload|load_backend|offload|listening|error' "
                       "| head -20")
    print(out or "(no matching lines)")
    print("\n=== loaded cDSP libraries in service process ===")
    out, _, _ = run(c, "pid=$(systemctl show -p MainPID --value llama-cdsp); "
                       "grep -iE 'cdsp|adsprpc|gemm' /proc/$pid/maps 2>/dev/null | "
                       "awk '{print $6}' | sort -u")
    print(out or "(none)")
    print("\n=== ports / cDSP state ===")
    out, _, _ = run(c, "ss -lntp | grep -E ':808[0-9]'; "
                       "printf 'remoteproc1: '; cat /sys/class/remoteproc/remoteproc1/state")
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="inspect only")
    ap.add_argument("--restart", action="store_true", help="restart only")
    ap.add_argument("--no-start", action="store_true", help="install but do not start")
    ap.add_argument("--enable", action="store_true", help="enable on boot (default: off)")
    ap.add_argument("--force-env", action="store_true",
                    help="overwrite an existing /etc/default/llama-cdsp")
    ap.add_argument("--port", type=int, default=8082, help="expected port")
    a = ap.parse_args()

    c = k.connect()
    try:
        if a.status:
            status(c)
            return 0
        if a.restart:
            print("restarting llama-cdsp ...")
            out, err, rc = run(c, "systemctl daemon-reload && systemctl restart llama-cdsp", 180)
            print(out, err)
            status(c)
            return rc

        if not precheck(c, a.port):
            print("\nPRECHECK FAILED - refusing to install. Fix the items above first.")
            return 1

        print("\n[2/4] upload unit + env (CRLF normalized)")
        k.put(c, UNIT_SRC, STAGE + "/llama-cdsp.service")
        out, err, rc = run(c, "sed -i 's/\\r$//' %s/llama-cdsp.service && "
                              "install -m 0644 %s/llama-cdsp.service %s && echo UNIT_OK"
                           % (STAGE, STAGE, UNIT_DST))
        print(" ", out or err)

        if a.force_env:
            env_action = "replace"
        else:
            _, _, exists = run(c, "test -f %s" % ENV_DST)
            env_action = "keep" if exists == 0 else "create"
        if env_action != "keep":
            k.put(c, ENV_SRC, STAGE + "/llama-cdsp.env")
            out, err, rc = run(c, "sed -i 's/\\r$//' %s/llama-cdsp.env && "
                                  "install -m 0644 %s/llama-cdsp.env %s && echo ENV_OK"
                               % (STAGE, STAGE, ENV_DST))
            print(" ", out or err)
        else:
            print("  env file already exists on device - kept (use --force-env to reset)")

        print("\n[3/4] daemon-reload" + (" + enable" if a.enable else "") + " + start")
        enable_cmd = ("systemctl enable llama-cdsp && " if a.enable else "")
        out, err, rc = run(c, "systemctl daemon-reload && %s"
                              "systemctl is-enabled llama-cdsp || true" % enable_cmd, 120)
        print(" ", (out or err).replace("\n", " | "))

        if a.no_start:
            print("  --no-start: unit installed, not started")
            return 0

        out, err, rc = run(c, "systemctl start llama-cdsp && echo START_ISSUED", 180)
        print(" ", out or err)
        if rc != 0:
            print("  start failed:", err)
            status(c)
            return 1

        # Model load takes ~2 s on this device; cDSP session setup may add more.
        out, _, _ = run(c,
                        "for i in $(seq 1 60); do "
                        "  code=$(curl -s -o /dev/null -w '%%{http_code}' "
                        "http://127.0.0.1:%d/health 2>/dev/null); "
                        "  [ \"$code\" = 200 ] && echo READY && break; "
                        "  sleep 1; "
                        "done; echo last=$code" % a.port, 120)
        print("  health:", out.replace("\n", " | "))
        if "READY" not in out:
            print("  WARNING: HTTP health endpoint did not become 200; see journal below")

        print("\n[4/4] status")
        status(c)
        return 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())
