/*
 * Mock CUDA Driver API for testing
 * Provides minimal implementation of cuDriverGetVersion for test environment
 */

#include <stdio.h>
#include <stdint.h>
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

// Context pushed by cuCtxPushCurrent (one level is all parcagpu needs).
// Mock contexts are (CUcontext)(1 + device index); see mock_cupti.c.
static __thread CUcontext pushed_ctx = NULL;

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    pushed_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext *ctx) {
    if (ctx != NULL)
        *ctx = pushed_ctx;
    pushed_ctx = NULL;
    return CUDA_SUCCESS;
}

// Mock cuCtxGetDevice — the pushed context's device, else device 0 (the
// gpu_config probe path doesn't push).
CUresult cuCtxGetDevice(CUdevice *device) {
    if (device == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = pushed_ctx ? (CUdevice)((uintptr_t)pushed_ctx - 1) : 0;
    return CUDA_SUCCESS;
}

// Mock cuDeviceGetUuid_v2 — bytes 0..14 are 0xA0+i, byte 15 is the ordinal,
// so each device's UUID is distinct and recognizable.
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    if (uuid == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (int i = 0; i < 15; i++)
        uuid->bytes[i] = (char)(0xA0 + i);
    uuid->bytes[15] = (char)dev;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    if (pciBusId == NULL || len < 13) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    snprintf(pciBusId, len, "0000:%02x:00.0", 0x10 + dev);
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
