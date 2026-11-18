/*
 * Mock CUDA Driver API for testing
 * Provides minimal implementation of cuDriverGetVersion for test environment
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Mock implementation of cuDriverGetVersion
// Returns CUDA 12.8.1 (12081) to enable PC sampling in tests
CUresult cuDriverGetVersion(int *driverVersion) {
    if (driverVersion == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Version format: major * 1000 + minor * 10 + patch
    // 12.8.1 = 12 * 1000 + 8 * 10 + 1 = 12081
    *driverVersion = 12081;

    if (getenv("PARCAGPU_DEBUG") != NULL) {
        fprintf(stderr, "[MOCK_CUDA] cuDriverGetVersion() -> 12.8.1 (12081)\n");
    }

    return CUDA_SUCCESS;
}

// Mock cuCtxSynchronize — no-op in test (no real GPU work to wait for)
CUresult cuCtxSynchronize(void) {
    return CUDA_SUCCESS;
}

// Mock cuCtxGetDevice — return a fixed device id for the parcagpu
// gpu_config probe path.
CUresult cuCtxGetDevice(CUdevice *device) {
    if (device == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

// Mock cuDeviceGetAttribute — return plausible H100-class values for the
// attributes parcagpu queries at context init. Anything not enumerated
// returns 0 (still success), so callers see a non-error but inert value.
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
    (void)dev;
    if (pi == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *pi = 1830000; // 1.83 GHz, H100 boost
        break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *pi = 132; // H100 SXM5
        break;
    default:
        *pi = 0;
        break;
    }
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Driver memory API + cuInit injection, for test/gpu_mem_client.c.
//
// Each memory call fires the CUPTI driver-API callbacks (ENTER, EXIT) that
// the mock CUPTI subscriber has enabled, the way the real driver does, so
// the client exercises parcagpu end to end with no GPU. The mock CUPTI
// library is reached through dlsym so libcuda keeps no link dependency on
// it. Addresses come from a bump allocator and are never reused.
// ---------------------------------------------------------------------------

#include <dlfcn.h>
#include <string.h>
#include <cupti.h>

static CUcontext g_mock_ctx = (CUcontext)0x1000;
static unsigned long long g_next_ptr = 0x7f1000000000ULL;
static uint32_t g_corr = 1;

static CUdeviceptr mock_bump(size_t size) {
    CUdeviceptr p = (CUdeviceptr)g_next_ptr;
    g_next_ptr += (size + 0xFFFFF) & ~0xFFFFFULL; // 1 MiB aligned
    g_next_ptr += 0x100000;                         // guard gap
    return p;
}

static void mock_fire(CUpti_CallbackId cbid, CUpti_ApiCallbackSite site,
                      const char *name, void *params, CUresult *ret,
                      uint32_t corr) {
    typedef int (*IsEnabled)(CUpti_CallbackDomain, CUpti_CallbackId);
    static IsEnabled isEnabled;
    static CUpti_CallbackFunc *cbp;
    if (!isEnabled)
        isEnabled = (IsEnabled)dlsym(RTLD_DEFAULT,
                                     "__mock_cupti_is_callback_enabled");
    if (!cbp)
        cbp = (CUpti_CallbackFunc *)dlsym(RTLD_DEFAULT,
                                          "__cupti_runtime_api_callback");
    if (!isEnabled || !cbp || !*cbp ||
        !isEnabled(CUPTI_CB_DOMAIN_DRIVER_API, cbid))
        return;
    CUpti_CallbackData d;
    memset(&d, 0, sizeof(d));
    d.callbackSite = site;
    d.functionName = name;
    d.functionParams = params;
    d.functionReturnValue = ret;
    d.context = g_mock_ctx;
    d.correlationId = corr;
    (*cbp)(NULL, CUPTI_CB_DOMAIN_DRIVER_API, cbid, &d);
}

// Run BODY between ENTER and EXIT callbacks for one driver call.
#define MOCK_DRIVER_CALL(CBID, NAME, PARAMS, BODY)                             \
    do {                                                                       \
        CUresult _ret = CUDA_SUCCESS;                                          \
        uint32_t _corr = g_corr++;                                             \
        mock_fire(CBID, CUPTI_API_ENTER, NAME, PARAMS, &_ret, _corr);          \
        BODY;                                                                  \
        mock_fire(CBID, CUPTI_API_EXIT, NAME, PARAMS, &_ret, _corr);           \
        return _ret;                                                           \
    } while (0)

// Real CUDA loads CUDA_INJECTION64_PATH and calls InitializeInjection from
// cuInit; do the same.
CUresult cuInit(unsigned int flags) {
    (void)flags;
    static int done;
    if (done)
        return CUDA_SUCCESS;
    done = 1;
    const char *inj = getenv("CUDA_INJECTION64_PATH");
    if (inj && *inj) {
        void *h = dlopen(inj, RTLD_NOW | RTLD_GLOBAL);
        int (*init)(void) = h ? (int (*)(void))dlsym(h, "InitializeInjection")
                              : NULL;
        if (init)
            init();
        else
            fprintf(stderr, "[MOCK_CUDA] injection %s failed: %s\n", inj,
                    dlerror());
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    (void)dev;
    *pctx = g_mock_ctx;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream s) {
    (void)s;
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    cuMemAlloc_v2_params p = {dptr, bytesize};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, "cuMemAlloc_v2",
                     &p, *dptr = mock_bump(bytesize));
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    cuMemFree_v2_params p = {dptr};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, "cuMemFree_v2", &p,
                     (void)0);
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
    cuMemAllocManaged_params p = {dptr, bytesize, flags};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
                     "cuMemAllocManaged", &p, *dptr = mock_bump(bytesize));
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize,
                         CUstream hStream) {
    cuMemAllocAsync_params p = {dptr, bytesize, hStream};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync, "cuMemAllocAsync",
                     &p, *dptr = mock_bump(bytesize));
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    cuMemFreeAsync_params p = {dptr, hStream};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync, "cuMemFreeAsync",
                     &p, (void)0);
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                       const CUmemAllocationProp *prop,
                                       CUmemAllocationGranularity_flags option) {
    (void)prop;
    (void)option;
    *granularity = 2 << 20;
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags) {
    (void)size;
    (void)prop;
    (void)flags;
    static CUmemGenericAllocationHandle next = 1;
    *handle = next++;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    (void)handle;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
    (void)alignment;
    (void)addr;
    (void)flags;
    *ptr = mock_bump(size);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)ptr;
    (void)size;
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
    cuMemMap_params p = {ptr, size, offset, handle, flags};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemMap, "cuMemMap", &p,
                     (void)0);
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    cuMemUnmap_params p = {ptr, size};
    MOCK_DRIVER_CALL(CUPTI_DRIVER_TRACE_CBID_cuMemUnmap, "cuMemUnmap", &p,
                     (void)0);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {
    (void)ptr;
    (void)size;
    (void)desc;
    (void)count;
    return CUDA_SUCCESS;
}
