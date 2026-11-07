#!/usr/bin/env bash
# build_device.sh - compile libggml-cdsp.so on the K20 Pro (aarch64 Linux).
#
# Run from a directory that contains ALL of these (the deploy script assembles
# exactly this layout under /data/local/tmp/ggml-cdsp):
#   ggml-cdsp.cpp        - the ggml backend plugin
#   cdsp_gemm_host.cpp   - FastRPC session wrapper (#ifdef __linux__)
#   cdsp_gemm_stub.c     - QAIC-generated FastRPC stub (cdsp_gemm_open/...)
#   cdsp_gemm_host.h     - host-side interface header
#   ggml-backend-impl.h  - copied verbatim from ggml 0.9.0 src (ABI pin)
#   gemm.h               - QAIC-generated header (cdsp_gemm_URI etc.)
#   AEEStdDef.h AEEStdErr.h
#
# Requires: g++ with C++20, libadsprpc (+ system <remote.h>), and the ggml
# public headers (ggml.h, ggml-backend.h) available at $GGML_INCLUDE.
#
# NOTE: cdsp_gemm_stub.c is C, not valid C++ (it uses C tentative definitions
# and void*->struct* conversions). It is compiled as a plain C object and linked
# into the C++ shared lib. gemm.h wraps its prototypes in `extern "C"`, so the
# C++ callers resolve the unmangled symbols.
set -e

GGML_INCLUDE="${GGML_INCLUDE:-/data/local/tmp/ggml-include}"
# On the K20 Pro the fastrpc dev headers live under /usr/local/include/fastrpc
# and libadsprpc under /usr/local/lib (built from mainline fastrpc). Default to
# those so the build works out of the box.
FASTRPC_INCLUDE="${FASTRPC_INCLUDE:-/usr/local/include/fastrpc}"
ADSPRPC_LIBDIR="${ADSPRPC_LIBDIR:-/usr/local/lib}"

echo "[*] GGML_INCLUDE   = $GGML_INCLUDE"
echo "[*] FASTRPC_INCLUDE= $FASTRPC_INCLUDE"
echo "[*] ADSPRPC_LIBDIR = $ADSPRPC_LIBDIR"

INC="-I. -I$GGML_INCLUDE -I$FASTRPC_INCLUDE"

# QAIC stub: compile as plain C into an object file.
gcc -c -fPIC $INC cdsp_gemm_stub.c -o cdsp_gemm_stub.o

# -DGGML_BACKEND_DL makes GGML_BACKEND_DL_IMPL() emit the ggml_backend_init()
# symbol that ggml_backend_load() dlopen's.
g++ -shared -fPIC -std=c++20 -DGGML_BACKEND_DL $INC \
    ggml-cdsp.cpp cdsp_gemm_host.cpp cdsp_gemm_stub.o \
    -L$ADSPRPC_LIBDIR -ladsprpc -o libggml-cdsp.so

echo "[*] built libggml-cdsp.so"
file libggml-cdsp.so 2>/dev/null || true
