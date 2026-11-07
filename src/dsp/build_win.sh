#!/usr/bin/env bash
# build_win.sh - build the cDSP int8 GEMM skeleton (.so) on Windows using the
# Qualcomm Hexagon SDK 4.3.0.0 toolchain (qaic.exe + hexagon-clang.exe).
# Run from this directory (src/hexagon690/llm).
set -e

SDK=/c/Qualcomm/Hexagon_SDK/4.3.0.0
QAIC="$SDK/ipc/fastrpc/qaic/WinNT/qaic.exe"
CLANG="$SDK/tools/HEXAGON_Tools/8.4.11/Tools/bin/hexagon-clang.exe"

INCS="$SDK/incs"
INC_STDDEF="$SDK/incs/stddef"
QURT_INC="$SDK/rtos/qurt/computev66/include"
QURT_POSIX="$SDK/rtos/qurt/computev66/include/posix"
QURT_LIB="$SDK/rtos/qurt/computev66/lib/pic"   # PIC libs for -fPIC shared obj
HEX_INC="$SDK/tools/HEXAGON_Tools/8.4.11/Tools/target/hexagon/include"  # HVX intrinsics

IDL=gemm.idl
GEN=gen
# QAIC names the skel library after the .idl FILE name (gemm.idl -> libgemm_skel.so),
# and bakes that into cdsp_gemm_URI. So the output .so MUST be named libgemm_skel.so
# or the open() URI won't match.
# Output into the repository's prebuilt/ dir so the deploy script and the
# committed artifact stay in sync. (From src/dsp this is ../../prebuilt.)
SKEL_SO="$(dirname "$0")/../../prebuilt/libgemm_skel.so"

echo "==> QAIC: $QAIC"
echo "==> CLANG: $CLANG"

mkdir -p "$GEN"
"$QAIC" -mdll -o "$GEN" -I"$INCS" -I"$INC_STDDEF" "$IDL"

# HVX build: -mhvx=128 enables 1024-bit vectors and the Q6_* HVX intrinsics.
# The cDSP (Hexagon 690 / V66) has a 128-byte HVX unit; the code locks it via
# qurt_hvx_lock(QURT_HVX_MODE_128B) for the whole invocation.
"$CLANG" -mv66 -mhvx -G0 -fPIC -shared -O3 \
  -I"$GEN" -I"$SDK/inc/std" -I"$SDK/inc" -I"$HEX_INC" \
  -I"$QURT_INC" -I"$QURT_POSIX" -I"$INCS" -I"$INC_STDDEF" \
  -o "$SKEL_SO" gemm_dsp.c "$GEN/cdsp_gemm_skel.c" \
  -L"$QURT_LIB" -lqurt -lposix -Wl,--no-threads

echo "==> built $SKEL_SO"
file "$SKEL_SO" 2>/dev/null || true
