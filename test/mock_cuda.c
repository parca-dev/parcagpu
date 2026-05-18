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
