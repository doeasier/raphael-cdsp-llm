//============================================================================
// cdsp_preload.c - LD_PRELOAD shim that registers the cDSP ggml backend into
// a llama.cpp build configured with GGML_BACKEND_DL=OFF.
//
// WHY THIS IS NEEDED
// ------------------
// llama.cpp's llama_backend_init() only calls ggml_backend_load_all() when
// ggml_backend_reg_count() == 0. In a GGML_BACKEND_DL=OFF build the CPU (and
// other built-in) backends are registered STATICALLY in the ggml backend
// registry constructor, so the count is already > 0 and load_all() - and with
// it the GGML_BACKEND_PATH environment hook - is never reached. Our out-of-tree
// libggml-cdsp.so therefore never gets registered and no MUL_MAT is offloaded.
//
// This shim runs as an ELF constructor: it resolves ggml_backend_load() from
// the already-loaded libggml and calls it with our plugin path, appending the
// "cdsp" ACCEL device to the registry. llama.cpp then picks it up in the
// "add ACCEL backends" loop (src/llama-context.cpp) and, with the default
// cparams.op_offload = true, the ggml scheduler routes supported MUL_MAT
// nodes to it. Non-MUL_MAT ops stay on the CPU backend.
//
// BUILD (on device, aarch64 Linux):
//   gcc -shared -fPIC -O2 cdsp_preload.c -o libggml_cdsp_preload.so -ldl
//
// USE:
//   export GGML_CDSP_BACKEND_PATH=/data/local/tmp/ggml-backend/libggml-cdsp.so
//   export LD_PRELOAD=/data/local/tmp/ggml-backend/libggml_cdsp_preload.so
//   export LD_LIBRARY_PATH=/root/llama.cpp-0.9.0/build/bin:/usr/local/lib
//   llama-cli -m model.gguf -p "..." -n 32 -st
//
// Alternative (no shim): rebuild llama.cpp with -DGGML_BACKEND_DL=ON, then
// GGML_BACKEND_PATH works directly because the registry starts empty.
//============================================================================
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef void * (*ggml_backend_load_fn)(const char *);

static const char * default_backend_path(void) {
    return "/data/local/tmp/ggml-backend/libggml-cdsp.so";
}

__attribute__((constructor))
static void cdsp_preload_init(void) {
    const char * path = getenv("GGML_CDSP_BACKEND_PATH");
    if (!path || !*path) path = default_backend_path();

    /* Make sure libggml is loaded and fully initialized before we touch its
     * registry. With LD_PRELOAD our constructor may run before libggml's own
     * initializers, so an explicit dlopen() is the safe ordering guarantee. */
    void * ggml = dlopen("libggml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!ggml) {
        fprintf(stderr, "[cdsp-preload] dlopen(libggml.so) failed: %s\n", dlerror());
        return;
    }

    ggml_backend_load_fn load =
        (ggml_backend_load_fn) dlsym(ggml, "ggml_backend_load");
    if (!load) {
        load = (ggml_backend_load_fn) dlsym(RTLD_DEFAULT, "ggml_backend_load");
    }
    if (!load) {
        fprintf(stderr, "[cdsp-preload] ggml_backend_load() not found\n");
        return;
    }

    void * reg = load(path);
    if (!reg) {
        fprintf(stderr, "[cdsp-preload] FAILED to register cDSP backend from %s\n", path);
    } else {
        fprintf(stderr, "[cdsp-preload] registered cDSP backend from %s\n", path);
    }
}
