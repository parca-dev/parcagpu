// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// GPU memory client for test/test-gpu-mem-e2e.sh. Uses only the CUDA driver
// API, so the same source builds against the real libcuda
// (gpu_mem_client_real) and against test/mock_cuda.c (gpu_mem_client), which
// fires the CUPTI callbacks a real driver would.
//
// Usage: gpu_mem_client <go-file> exact|bulk
//
// Before <go-file> exists the client makes one 64 MiB alloc/free, which must
// not be reported: no consumer is attached yet. Then it runs one workload:
//
//   exact  16 allocations of known sizes across all four kinds, half freed:
//            device   8 x 4 MiB  (cuMemAlloc),         4 freed
//            managed  2 x 2 MiB  (cuMemAllocManaged),  1 freed
//            async    4 x 1 MiB  (cuMemAllocAsync),    2 freed
//            vmm      2 x 2 MiB  (cuMemMap),           1 unmapped
//          allocs=16 bytes=46137344, in use: allocs=8 bytes=23068672.
//          Run with PARCAGPU_MEM_SAMPLING_INTERVAL=0 to check exact counts.
//
//   bulk   2000 x 256 KiB cuMemAlloc, each freed right away, for the
//          weighted-total check under the default sampling interval.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call,     \
              (int)_r);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define MiB (1024u * 1024u)

static void wait_for(const char *path) {
  struct timespec ts = {0, 20 * 1000 * 1000};
  for (int i = 0; i < 3000 && access(path, F_OK) != 0; ++i)
    nanosleep(&ts, NULL);
  if (access(path, F_OK) != 0) {
    fprintf(stderr, "timed out waiting for %s\n", path);
    exit(1);
  }
}

static void exact(CUdevice dev) {
  CUdeviceptr d[8], m[2], a[4];
  for (int i = 0; i < 8; ++i)
    CHECK(cuMemAlloc(&d[i], 4 * MiB));
  for (int i = 0; i < 4; ++i)
    CHECK(cuMemFree(d[i]));

  for (int i = 0; i < 2; ++i)
    CHECK(cuMemAllocManaged(&m[i], 2 * MiB, CU_MEM_ATTACH_GLOBAL));
  CHECK(cuMemFree(m[0]));

  for (int i = 0; i < 4; ++i)
    CHECK(cuMemAllocAsync(&a[i], 1 * MiB, 0));
  CHECK(cuMemFreeAsync(a[0], 0));
  CHECK(cuMemFreeAsync(a[1], 0));
  CHECK(cuStreamSynchronize(0));

  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t gran = 0;
  CHECK(cuMemGetAllocationGranularity(&gran, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (gran == 0 || (2 * MiB) % gran != 0) {
    fprintf(stderr, "unexpected VMM granularity %zu\n", gran);
    exit(1);
  }
  CUdeviceptr base;
  CHECK(cuMemAddressReserve(&base, 4 * MiB, 0, 0, 0));
  CUmemGenericAllocationHandle h[2];
  CUmemAccessDesc access;
  memset(&access, 0, sizeof(access));
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  for (int i = 0; i < 2; ++i) {
    CHECK(cuMemCreate(&h[i], 2 * MiB, &prop, 0));
    CHECK(cuMemMap(base + (CUdeviceptr)i * 2 * MiB, 2 * MiB, 0, h[i], 0));
    CHECK(cuMemSetAccess(base + (CUdeviceptr)i * 2 * MiB, 2 * MiB, &access, 1));
  }
  CHECK(cuMemUnmap(base, 2 * MiB));
  CHECK(cuMemRelease(h[0]));
  // The rest stays allocated until exit: it is the expected in-use set.
}

static void bulk(void) {
  for (int i = 0; i < 2000; ++i) {
    CUdeviceptr p;
    CHECK(cuMemAlloc(&p, 256 * 1024));
    CHECK(cuMemFree(p));
  }
}

int main(int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[2], "exact") && strcmp(argv[2], "bulk"))) {
    fprintf(stderr, "usage: %s <go-file> exact|bulk\n", argv[0]);
    return 2;
  }

  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));

  // Before any consumer: must not be reported.
  CUdeviceptr early;
  CHECK(cuMemAlloc(&early, 64 * MiB));
  CHECK(cuMemFree(early));

  printf("ready\n");
  fflush(stdout);
  wait_for(argv[1]);

  if (strcmp(argv[2], "exact") == 0)
    exact(dev);
  else
    bulk();

  printf("done\n");
  fflush(stdout);
  // Give the parser time to drain the ring buffer before we exit.
  sleep(1);
  return 0;
}
