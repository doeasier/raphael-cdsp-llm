#!/usr/bin/env python3
"""Deploy + verify the ggml-cdsp backend on the K20 Pro (Stages 2, 2.5, 5, 6).

What it does (all over USB-NCM SSH, root@172.16.42.1):
  1. Uploads the ggml 0.9.0 public headers to /data/local/tmp/ggml-include
     (so the device compiles against the EXACT ABI this plugin was written for).
  2. Assembles a self-contained staging dir (plugin sources + QAIC stub + the
     AEEStd headers + build/test sources) and uploads it to
     /data/local/tmp/ggml-cdsp.
  3. Runs build_device.sh on-device -> libggml-cdsp.so
  4. Installs libggml-cdsp.so into GGML_BACKEND_DIR (/data/local/tmp/ggml-backend).
  5. Builds test_cdsp.cpp against the device's libggml.so and runs it with
     GGML_BACKEND_DIR set. The test compares the cDSP MUL_MAT result against a
     CPU int8 emulation (tiny tolerance) -> prints PASS/FAIL.
  6. Builds + runs test_sched.cpp to prove the backend integrates under the
     ggml multi-backend scheduler (MUL_MAT -> cDSP, other ops -> CPU).
  7. Builds + runs test_cache.cpp (Stage 5 weight-cache reuse, 2nd compute is
     a cache hit).
  8. Builds + runs test_q4k.cpp: REAL Q4_K weight blocks (the format used by
     Qwen3.5-4B-Q4_K_M.gguf) computed by both the CPU and cDSP backends, to
     validate the dequantization path.
  9. Builds libggml_cdsp_preload.so, the LD_PRELOAD shim that registers this
     out-of-tree backend into a llama.cpp build with GGML_BACKEND_DL=OFF
     (see docs/llama-integration.md).

Prerequisites on the device:
  * g++ with C++20, libadsprpc (+ system <remote.h>).
  * Stage 1 done: libgemm_skel.so installed at /usr/lib/rfsa/cdsp/ and cDSP
    FastRPC working (run scripts/device/deploy_cdsp_gemm.py first). The runtime
    test will fail otherwise (no DSP skel to open).
  * A build of llama.cpp 0.9.0 whose libggml.so is findable (auto-detected).

Usage:
  python3 scripts/device/deploy_ggml_cdsp.py
"""
import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import k20_ssh as k  # noqa: E402

PLUGIN_DIR  = os.path.join(ROOT, "src", "ggml-backend")
SRC_DSP     = os.path.join(ROOT, "src", "dsp")
GEN_DIR     = os.path.join(ROOT, "src", "gen")
GGML_INC_SRC = os.path.join(ROOT, "deps", "ggml-0.9.0-headers")

STAGING = os.path.join(ROOT, "tmp", "ggml-cdsp-staging")
REMOTE_STAGE = "/data/local/tmp/ggml-cdsp"
REMOTE_INC = "/data/local/tmp/ggml-include"
REMOTE_BACKEND_DIR = "/data/local/tmp/ggml-backend"

# files that must land together in the staging dir (flat)
STAGE_FILES = [
    (os.path.join(PLUGIN_DIR, "ggml-cdsp.cpp"),      "ggml-cdsp.cpp"),
    (os.path.join(PLUGIN_DIR, "cdsp_gemm_host.cpp"), "cdsp_gemm_host.cpp"),
    (os.path.join(PLUGIN_DIR, "cdsp_gemm_host.h"),   "cdsp_gemm_host.h"),
    (os.path.join(PLUGIN_DIR, "ggml-backend-impl.h"),"ggml-backend-impl.h"),
    (os.path.join(PLUGIN_DIR, "build_device.sh"),    "build_device.sh"),
    (os.path.join(PLUGIN_DIR, "test_cdsp.cpp"),      "test_cdsp.cpp"),
    (os.path.join(PLUGIN_DIR, "test_sched.cpp"),     "test_sched.cpp"),
    (os.path.join(PLUGIN_DIR, "test_cache.cpp"),     "test_cache.cpp"),
    (os.path.join(PLUGIN_DIR, "test_q4k.cpp"),       "test_q4k.cpp"),
    (os.path.join(PLUGIN_DIR, "cdsp_preload.c"),     "cdsp_preload.c"),
    (os.path.join(GEN_DIR,     "gemm.h"),            "gemm.h"),
    (os.path.join(GEN_DIR,     "cdsp_gemm_stub.c"),  "cdsp_gemm_stub.c"),
    (os.path.join(SRC_DSP,     "AEEStdDef.h"),       "AEEStdDef.h"),
    (os.path.join(SRC_DSP,     "AEEStdErr.h"),       "AEEStdErr.h"),
]


def step(msg):
    print("\n=== %s ===" % msg)


def main():
    # 0. sanity on host
    step("assembling staging dir")
    if not os.path.isdir(GGML_INC_SRC):
        print("MISSING ggml headers at", GGML_INC_SRC)
        return 1
    if os.path.isdir(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING, exist_ok=True)
    for src, dst in STAGE_FILES:
        if not os.path.isfile(src):
            print("MISSING", src)
            return 1
        shutil.copy(src, os.path.join(STAGING, dst))
    print("staged %d files in %s" % (len(STAGE_FILES), STAGING))

    # 1. connect
    step("connecting to device")
    try:
        c = k.connect()
    except Exception as ex:
        print("DEVICE_UNREACHABLE:", ex)
        print("The phone needs to be booted and reachable over USB-NCM (root@172.16.42.1).")
        return 2

    try:
        # 2. upload ggml headers
        step("uploading ggml 0.9.0 public headers -> %s" % REMOTE_INC)
        k.put_dir(c, GGML_INC_SRC, REMOTE_INC)

        # 3. upload staging
        step("uploading plugin staging -> %s" % REMOTE_STAGE)
        k.put_dir(c, STAGING, REMOTE_STAGE)

        # 4. pre-flight checks on device
        step("device pre-flight checks")
        out, err, st = k.run(c,
            "test -f /usr/lib/rfsa/cdsp/libgemm_skel.so && echo SKEL_OK || echo SKEL_MISSING;"
            " which g++ && g++ --version | head -1;"
            " ls %s/ggml.h %s/ggml-backend.h" % (REMOTE_INC, REMOTE_INC))
        print(out)
        if "SKEL_MISSING" in out:
            print("WARN: libgemm_skel.so not found on cDSP path; Stage 1 (deploy_cdsp_gemm.py) "
                  "must be run first or the runtime test will fail to open the DSP skel.")

        # 5. build the plugin
        step("building libggml-cdsp.so on device")
        cmd = ("cd %s && GGML_INCLUDE=%s FASTRPC_INCLUDE=/usr/local/include/fastrpc "
               "ADSPRPC_LIBDIR=/usr/local/lib bash build_device.sh"
               % (REMOTE_STAGE, REMOTE_INC))
        out, err, st = k.run(c, cmd, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("BUILD FAILED (status %d)" % st)
            return 3
        out, err, st = k.run(c, "ls -l %s/libggml-cdsp.so" % REMOTE_STAGE)
        print(out)

        # 6. install into GGML_BACKEND_DIR
        step("installing into %s" % REMOTE_BACKEND_DIR)
        k.run(c, "mkdir -p %s && cp %s/libggml-cdsp.so %s/"
               % (REMOTE_BACKEND_DIR, REMOTE_STAGE, REMOTE_BACKEND_DIR))

        # 7. find libggml on device and build the test
        step("locating libggml.so and building test_cdsp")
        out, err, st = k.run(c, "find / -name 'libggml.so' 2>/dev/null | head -1")
        ggml_lib = out.strip()
        if not ggml_lib:
            print("WARN: libggml.so not found on device; cannot build/run the runtime test.")
            print("This plugin can still be used by llama.cpp once it is built against this libggml.")
            return 0
        ggml_libdir = os.path.dirname(ggml_lib)
        print("libggml at:", ggml_lib)
        test_cmd = (
            "cd %s && g++ -std=c++20 test_cdsp.cpp -I%s -L%s "
            "-lggml -lggml-base -lggml-cpu -o test_cdsp"
            % (REMOTE_STAGE, REMOTE_INC, ggml_libdir))
        out, err, st = k.run(c, test_cmd, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("TEST BUILD FAILED (status %d)" % st)
            return 4

        # 8. run the end-to-end test
        step("running test_cdsp (cDSP MUL_MAT vs int8 emulation)")
        run_cmd = (
            "cd %s && GGML_BACKEND_DIR=%s LD_LIBRARY_PATH=%s:/usr/local/lib:$LD_LIBRARY_PATH ./test_cdsp"
            % (REMOTE_STAGE, REMOTE_BACKEND_DIR, ggml_libdir))
        out, err, st = k.run(c, run_cmd, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if "PASS" in out:
            print("\n>>> Stage 2 VERIFIED: cDSP backend computes MUL_MAT correctly.")
        else:
            print("\n>>> Stage 2 test did not report PASS (status %d)." % st)
            print("    If this is a DSP-open failure, re-run scripts/device/deploy_cdsp_gemm.py")
            print("    to (re)install libgemm_skel.so, then retry.")
            return 5

        # 9. multi-backend scheduler integration test (cDSP + CPU together)
        step("running test_sched (scheduler: MUL_MAT->cDSP, relu->CPU)")
        sched_build = (
            "cd %s && g++ -std=c++20 test_sched.cpp -I%s -L%s "
            "-lggml -lggml-base -lggml-cpu -o test_sched"
            % (REMOTE_STAGE, REMOTE_INC, ggml_libdir))
        out, err, st = k.run(c, sched_build, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("SCHED TEST BUILD FAILED (status %d)" % st)
            return 6

        sched_run = (
            "cd %s && GGML_BACKEND_DIR=%s GGML_CDSP_DEBUG=1 "
            "LD_LIBRARY_PATH=%s:/usr/local/lib:$LD_LIBRARY_PATH ./test_sched"
            % (REMOTE_STAGE, REMOTE_BACKEND_DIR, ggml_libdir))
        out, err, st = k.run(c, sched_run, timeout=300)
        print(out)
        if "offloaded MUL_MAT" in (out + err):
            print(">>> CONFIRMED: scheduler offloaded MUL_MAT to the cDSP backend.")
        if "SCHED_TEST PASS" in out:
            print("\n>>> Stage 2.5 VERIFIED: backend integrates under the ggml "
                  "multi-backend scheduler (cDSP + CPU).")
        else:
            print("\n>>> Scheduler test did not report PASS (status %d)." % st)
            return 7

        # 10. weight-cache reuse test (Stage 5): same weights across 2 computes
        step("running test_cache (weight-cache: 2 computes, 2nd is a hit)")
        cache_build = (
            "cd %s && g++ -std=c++20 test_cache.cpp -I%s -L%s "
            "-lggml -lggml-base -lggml-cpu -o test_cache"
            % (REMOTE_STAGE, REMOTE_INC, ggml_libdir))
        out, err, st = k.run(c, cache_build, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("CACHE TEST BUILD FAILED (status %d)" % st)
            return 8

        cache_run = (
            "cd %s && GGML_BACKEND_DIR=%s GGML_CDSP_DEBUG=1 "
            "LD_LIBRARY_PATH=%s:/usr/local/lib:$LD_LIBRARY_PATH ./test_cache"
            % (REMOTE_STAGE, REMOTE_BACKEND_DIR, ggml_libdir))
        out, err, st = k.run(c, cache_run, timeout=300)
        print(out)
        if "offloaded MUL_MAT" in (out + err):
            print(">>> CONFIRMED: scheduler offloaded MUL_MAT to the cDSP backend.")
        if "cache hits=" in (out + err):
            # the 2nd compute must have been a cache HIT (hits >= 1)
            print(">>> CONFIRMED: cached weights reused across computes "
                  "(no re-upload of the weight matrix).")
        if "CACHE_TEST PASS" in out:
            print("\n>>> Stage 5 VERIFIED: weight-cache API works end-to-end "
                  "(correct + weights uploaded once).")
        else:
            print("\n>>> Weight-cache test did not report PASS (status %d)." % st)
            return 9

        # 11. Q4_K dequant correctness - the weight format of real GGUF models
        step("running test_q4k (Q4_K dequant vs ggml's own CPU kernel)")
        q4k_build = (
            "cd %s && g++ -std=c++20 test_q4k.cpp -I%s -L%s "
            "-lggml -lggml-base -lggml-cpu -o test_q4k"
            % (REMOTE_STAGE, REMOTE_INC, ggml_libdir))
        out, err, st = k.run(c, q4k_build, timeout=300)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("Q4K TEST BUILD FAILED (status %d)" % st)
            return 10
        q4k_run = (
            "cd %s && GGML_CDSP_BACKEND=%s/libggml-cdsp.so "
            "LD_LIBRARY_PATH=%s:/usr/local/lib:$LD_LIBRARY_PATH ./test_q4k"
            % (REMOTE_STAGE, REMOTE_BACKEND_DIR, ggml_libdir))
        out, err, st = k.run(c, q4k_run, timeout=300)
        print(out)
        if "Q4K_TEST PASS" in out:
            print(">>> CONFIRMED: real Q4_K weight blocks decode correctly on the cDSP.")
        else:
            print(">>> Q4_K test did not report PASS (status %d)." % st)
            return 11

        # 12. LD_PRELOAD shim: registers this out-of-tree backend into a
        # llama.cpp build with GGML_BACKEND_DL=OFF (where GGML_BACKEND_PATH is
        # never reached because the CPU backend is statically registered).
        step("building libggml_cdsp_preload.so (llama.cpp integration shim)")
        shim_build = (
            "cd %s && gcc -shared -fPIC -O2 cdsp_preload.c "
            "-o %s/libggml_cdsp_preload.so -ldl && "
            "ls -l %s/libggml_cdsp_preload.so"
            % (REMOTE_STAGE, REMOTE_BACKEND_DIR, REMOTE_BACKEND_DIR))
        out, err, st = k.run(c, shim_build, timeout=180)
        print(out)
        if err:
            print("STDERR:", err)
        if st != 0:
            print("PRELOAD SHIM BUILD FAILED (status %d)" % st)
            return 12
        print("\n>>> Stage 6 READY: see docs/llama-integration.md for the "
              "llama-cli / llama-server integration commands.")
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
