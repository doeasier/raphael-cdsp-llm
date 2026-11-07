#!/usr/bin/env python3
"""Deploy the cDSP int8 GEMM kernel to the K20 Pro and run an on-device
correctness + timing test (Stage 1).

What it does:
  1. SFTP the host test + QAIC-generated stub/header + the DSP .so to the device.
  2. Compile host_test on the device (aarch64 gcc + libadsprpc).
  3. Install libgemm_skel.so into the cDSP search path (/usr/lib/rfsa/cdsp).
  4. Run host_test at a few sizes and report PASS/FAIL + DSP compute time.

The DSP .so is built on Windows beforehand via src/dsp/build_win.sh
(or use the committed prebuilt/libgemm_skel.so).

Usage:
  python3 scripts/device/deploy_cdsp_gemm.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from k20_ssh import connect, run, put_exec

SRC_DSP   = os.path.join(ROOT, "src", "dsp")
SRC_HOST  = os.path.join(ROOT, "src", "host")
GEN_DIR   = os.path.join(ROOT, "src", "gen")
PREBUILT  = os.path.join(ROOT, "prebuilt")

DEVICE_DIR = "/data/local/tmp/cdsp_gemm"
CDSP_PATH  = "/usr/lib/rfsa/cdsp/libgemm_skel.so"

# local file -> remote relative path inside DEVICE_DIR
LOCAL_FILES = [
    (os.path.join(SRC_HOST, "host_test.c"),        "host_test.c"),
    (os.path.join(SRC_HOST, "host_test_cached.c"), "host_test_cached.c"),
    (os.path.join(GEN_DIR,  "gemm.h"),             "gen/gemm.h"),
    (os.path.join(GEN_DIR,  "cdsp_gemm_stub.c"),   "gen/cdsp_gemm_stub.c"),
    (os.path.join(SRC_DSP,  "AEEStdDef.h"),        "AEEStdDef.h"),
    (os.path.join(SRC_DSP,  "AEEStdErr.h"),        "AEEStdErr.h"),
    (os.path.join(PREBUILT, "libgemm_skel.so"),    "libgemm_skel.so"),
]

SIZES = [(32, 128, 128), (64, 256, 256), (128, 512, 512)]

# Real model-dimension shapes for the weight-cache test, to show the decode
# win: a 4B model's attention/MLP weights are tens of MB, so shipping them over
# FastRPC on every token is fatal. With the cache, only the tiny activation A
# (M x K) is shipped per call.
#   (M, N, K)  M=decode rows (1), N=weight cols, K=contraction
SIZES_CACHED = [(1, 4096, 4096), (1, 11008, 4096), (32, 4096, 4096)]


def main():
    c = connect()
    print("[*] connected to device")

    # SFTP write-open is flaky on this device; transfer via exec+base64 instead.
    for local, remote_rel in LOCAL_FILES:
        if not os.path.exists(local):
            print(f"[!] missing local file: {local}")
            continue
        put_exec(c, local, f"{DEVICE_DIR}/{remote_rel}")
        print(f"[+] put {remote_rel}")

    # compile on device
    compile_cmd = (
        f"cd {DEVICE_DIR} && "
        "gcc -O2 -I. -Igen -I/usr/local/include/fastrpc "
        "host_test.c gen/cdsp_gemm_stub.c -o host_test "
        "-L/usr/local/lib -ladsprpc -lpthread -ldl && "
        "gcc -O2 -I. -Igen -I/usr/local/include/fastrpc "
        "host_test_cached.c gen/cdsp_gemm_stub.c -o host_test_cached "
        "-L/usr/local/lib -ladsprpc -lpthread -ldl"
    )
    out, err, st = run(c, compile_cmd, timeout=120)
    print("--- compile ---")
    print(out)
    if err:
        print("STDERR:", err)
    if st != 0:
        print("[!] compile failed"); c.close(); return

    # install skel into cDSP path (backup any existing)
    run(c, f"cp {CDSP_PATH} {CDSP_PATH}.bak 2>/dev/null; "
           f"cp {DEVICE_DIR}/libgemm_skel.so {CDSP_PATH} && "
           f"chmod 644 {CDSP_PATH} && echo installed")
    run(c, f"ls -la {CDSP_PATH}")

    # Restart the fastrpc daemon VIA SYSTEMD so it re-scans /usr/lib/rfsa/cdsp
    # and picks up the ADSP_LIBRARY_PATH from its unit drop-in. Do NOT pkill it
    # manually -- a bare `pkill` drops the env the daemon needs to find skels
    # and can trigger a systemd restart storm that starves sshd.
    run(c, "systemctl restart cdsprpcd 2>/dev/null; sleep 2; "
           "pgrep -x cdsprpcd >/dev/null && echo DAEMON_OK || echo DAEMON_MISSING")

    # run tests
    for (M, N, K) in SIZES:
        cmd = f"cd {DEVICE_DIR} && ./host_test {M} {N} {K}"
        out, err, st = run(c, cmd, timeout=60)
        print(f"\n=== host_test (one-shot) {M}x{N}x{K} (rc={st}) ===")
        print(out)
        if err:
            print("STDERR:", err)

    # Stage-5 weight-cache test (upload once, reuse handle; only A shipped)
    for (M, N, K) in SIZES + SIZES_CACHED:
        cmd = f"cd {DEVICE_DIR} && ./host_test_cached {M} {N} {K}"
        out, err, st = run(c, cmd, timeout=120)
        print(f"\n=== host_test_cached (weight-cache) {M}x{N}x{K} (rc={st}) ===")
        print(out)
        if err:
            print("STDERR:", err)

    c.close()


if __name__ == "__main__":
    main()
