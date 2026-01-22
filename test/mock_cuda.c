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
