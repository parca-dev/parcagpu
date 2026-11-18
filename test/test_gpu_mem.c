// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// Focused unit test for sampled GPU memory allocation profiling
// (src/gpu_memory.cpp). Loads libparcagpucupti.so against the mock CUDA and
// CUPTI libraries, fires driver-API alloc/free callbacks through the
// subscriber the library registered, and asserts through the test-only
// `parcagpu_gpumem_test_*` accessors.
//
// No GPU, no live USDT consumer: `force_capture` stands in for an attached
// probe.

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cupti.h>

// Mirrors the struct in src/gpu_memory.cpp.
struct parcagpu_gpumem_test_stats {
  uint64_t allocs_seen;
  uint64_t allocs_sampled;
  uint64_t bytes_seen;
  uint64_t weighted_bytes;
  uint64_t frees;
  uint64_t last_ptr;
  uint64_t last_weight;
  uint32_t last_kind;
};

enum { KIND_DEVICE = 1, KIND_MANAGED = 2, KIND_ASYNC = 3, KIND_VMM = 4 };

typedef int (*InitFn_t)(void);
typedef void (*ForceFn_t)(int);
typedef void (*SetU64Fn_t)(uint64_t);
typedef uint64_t (*GetU64Fn_t)(void);
typedef void (*StatsFn_t)(struct parcagpu_gpumem_test_stats *);
typedef void (*ResetFn_t)(void);
typedef int (*IsEnabledFn_t)(CUpti_CallbackDomain, CUpti_CallbackId);

static CUpti_CallbackFunc g_cb;
static IsEnabledFn_t g_isEnabled;
static StatsFn_t g_stats;
static ResetFn_t g_reset;
static SetU64Fn_t g_setInterval;
static SetU64Fn_t g_setSeed;

static void fail(const char *what) {
  fprintf(stderr, "FAIL: %s\n", what);
  exit(1);
}

#define EXPECT(cond, what)                                                     \
  do {                                                                         \
    if (!(cond))                                                               \
      fail(what);                                                              \
  } while (0)

static struct parcagpu_gpumem_test_stats stats(void) {
  struct parcagpu_gpumem_test_stats s;
  memset(&s, 0, sizeof(s));
  g_stats(&s);
  return s;
}

// Dispatch only if subscribed, like real CUPTI.
static void fire(CUpti_CallbackId cbid, CUpti_ApiCallbackSite site,
                 void *params, CUresult ret) {
  if (!g_isEnabled(CUPTI_CB_DOMAIN_DRIVER_API, cbid))
    return;
  CUpti_CallbackData cbdata;
  memset(&cbdata, 0, sizeof(cbdata));
  cbdata.callbackSite = site;
  cbdata.functionName = "cuMem*";
  cbdata.functionParams = params;
  cbdata.functionReturnValue = &ret;
  cbdata.context = (CUcontext)0x1000;
  cbdata.correlationId = 1;
  g_cb(NULL, CUPTI_CB_DOMAIN_DRIVER_API, cbid, &cbdata);
}

// One full alloc call: ENTER (no output yet) then EXIT with the result.
static void alloc_v2(CUdeviceptr ptr, size_t size, CUresult ret) {
  CUdeviceptr out = 0;
  cuMemAlloc_v2_params p = {&out, size};
  fire(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, CUPTI_API_ENTER, &p, ret);
  out = ptr;
  fire(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, CUPTI_API_EXIT, &p, ret);
}

static void free_v2(CUdeviceptr ptr) {
  cuMemFree_v2_params p = {ptr};
  fire(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, CUPTI_API_ENTER, &p, CUDA_SUCCESS);
  fire(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, CUPTI_API_EXIT, &p, CUDA_SUCCESS);
}

// xorshift64 for the test's own size distribution.
static uint64_t g_rng = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return g_rng;
}

int main(void) {
  const char *libpath = getenv("PARCAGPU_TEST_LIB");
  if (!libpath)
    libpath = "./libparcagpucupti.so";

  // Read at init; checked below, then overridden per case.
  setenv("PARCAGPU_MEM_SAMPLING_INTERVAL", "1234567", 1);

  void *h = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
  if (!h)
    fail(dlerror());

  InitFn_t init = (InitFn_t)dlsym(h, "InitializeInjection");
  ForceFn_t force = (ForceFn_t)dlsym(h, "parcagpu_gpumem_test_force_capture");
  g_setSeed = (SetU64Fn_t)dlsym(h, "parcagpu_gpumem_test_set_seed");
  g_setInterval = (SetU64Fn_t)dlsym(h, "parcagpu_gpumem_test_set_interval");
  GetU64Fn_t getInterval =
      (GetU64Fn_t)dlsym(h, "parcagpu_gpumem_test_interval");
  g_stats = (StatsFn_t)dlsym(h, "parcagpu_gpumem_test_stats");
  g_reset = (ResetFn_t)dlsym(h, "parcagpu_gpumem_test_reset");
  g_isEnabled =
      (IsEnabledFn_t)dlsym(RTLD_DEFAULT, "__mock_cupti_is_callback_enabled");
  if (!init || !force || !g_setSeed || !g_setInterval || !getInterval ||
      !g_stats || !g_reset || !g_isEnabled)
    fail("missing library or mock entry points");

  EXPECT(init() == 1, "InitializeInjection != 1");
  CUpti_CallbackFunc *cbp =
      (CUpti_CallbackFunc *)dlsym(RTLD_DEFAULT, "__cupti_runtime_api_callback");
  EXPECT(cbp && *cbp, "no CUPTI subscriber callback registered");
  g_cb = *cbp;

  // -------- Case 1: env var and subscriptions --------
  EXPECT(getInterval() == 1234567,
         "PARCAGPU_MEM_SAMPLING_INTERVAL not applied");
  const CUpti_CallbackId want[] = {
      CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuMemMap,
      CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuMemUnmap,
  };
  for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
    if (!g_isEnabled(CUPTI_CB_DOMAIN_DRIVER_API, want[i])) {
      fprintf(stderr, "cbid %u not subscribed\n", (unsigned)want[i]);
      fail("memory cbid not subscribed");
    }
  }
  EXPECT(!g_isEnabled(CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2),
         "pinned host alloc should not be subscribed");
  printf("ok  subscriptions and env\n");

  // Nothing is reported while no consumer is attached.
  g_setInterval(0);
  g_reset();
  alloc_v2(0x7000000000ULL, 4096, CUDA_SUCCESS);
  free_v2(0x7000000000ULL);
  {
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.allocs_seen == 0 && s.frees == 0,
           "events recorded with no consumer attached");
  }
  printf("ok  no consumer, no work\n");

  force(1);

  // -------- Case 2: interval 0 samples every allocation, weight == size ----
  g_setInterval(0);
  g_reset();
  alloc_v2(0x7f0000000000ULL, 4096, CUDA_SUCCESS);
  {
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.allocs_sampled == 1, "interval 0: alloc not sampled");
    EXPECT(s.last_weight == 4096, "interval 0: weight != size");
    EXPECT(s.last_ptr == 0x7f0000000000ULL, "wrong ptr reported");
    EXPECT(s.last_kind == KIND_DEVICE, "cuMemAlloc_v2 kind != device");
  }

  {
    CUdeviceptr out = 0x7f0000100000ULL;
    size_t pitch = 512;
    cuMemAllocPitch_v2_params p = {&out, &pitch, 500, 100, 4};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2, CUPTI_API_EXIT, &p,
         CUDA_SUCCESS);
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.last_weight == 512 * 100, "pitch alloc size != pitch*height");
    EXPECT(s.last_kind == KIND_DEVICE, "pitch kind != device");
  }
  {
    CUdeviceptr out = 0x7f0000200000ULL;
    cuMemAllocManaged_params p = {&out, 8192, 1};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged, CUPTI_API_EXIT, &p,
         CUDA_SUCCESS);
    EXPECT(stats().last_kind == KIND_MANAGED, "managed kind");
  }
  {
    CUdeviceptr out = 0x7f0000300000ULL;
    cuMemAllocAsync_params p = {&out, 16384, NULL};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync, CUPTI_API_EXIT, &p,
         CUDA_SUCCESS);
    EXPECT(stats().last_kind == KIND_ASYNC, "async kind");
    out = 0x7f0000400000ULL;
    cuMemAllocFromPoolAsync_ptsz_params q = {&out, 16384, NULL, NULL};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz, CUPTI_API_EXIT,
         &q, CUDA_SUCCESS);
    EXPECT(stats().last_ptr == 0x7f0000400000ULL, "pool async ptr");
    EXPECT(stats().last_kind == KIND_ASYNC, "pool async kind");
  }
  {
    cuMemMap_params p = {0x7f0001000000ULL, 2 << 20, 0, 0, 0};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemMap, CUPTI_API_EXIT, &p, CUDA_SUCCESS);
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.last_kind == KIND_VMM, "cuMemMap kind != vmm");
    EXPECT(s.last_weight == (2 << 20), "cuMemMap size");
  }
  {
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.allocs_sampled == 6, "interval 0: expected 6 sampled allocs");
    EXPECT(s.weighted_bytes == s.bytes_seen,
           "interval 0: weighted bytes != bytes");
  }
  printf("ok  interval 0 samples everything, kinds and sizes\n");

  // -------- Case 3: failed allocs and wrong sites report nothing ----------
  g_reset();
  alloc_v2(0x7f0002000000ULL, 4096, CUDA_ERROR_OUT_OF_MEMORY);
  {
    CUdeviceptr out = 0x7f0002000000ULL;
    cuMemAlloc_v2_params p = {&out, 4096};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, CUPTI_API_ENTER, &p,
         CUDA_SUCCESS);
  }
  EXPECT(stats().allocs_sampled == 0, "failed or ENTER-site alloc reported");
  printf("ok  failed allocs ignored\n");

  // -------- Case 4: frees fire once, on ENTER, for every free cbid --------
  g_reset();
  free_v2(0x7f0000000000ULL);
  {
    cuMemFreeAsync_params p = {0x7f0000300000ULL, NULL};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync, CUPTI_API_ENTER, &p,
         CUDA_SUCCESS);
    cuMemFreeAsync_ptsz_params q = {0x7f0000400000ULL, NULL};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz, CUPTI_API_ENTER, &q,
         CUDA_SUCCESS);
    cuMemUnmap_params u = {0x7f0001000000ULL, 2 << 20};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemUnmap, CUPTI_API_ENTER, &u, CUDA_SUCCESS);
    cuMemFree_v2_params z = {0};
    fire(CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, CUPTI_API_ENTER, &z,
         CUDA_SUCCESS);
  }
  EXPECT(stats().frees == 4, "expected 4 frees (null free ignored)");
  printf("ok  frees\n");

  // -------- Case 5: byte sampling is unbiased --------
  // Log-uniform sizes from 256 B to 64 MiB, mean interval 512 KiB.
  const uint64_t interval = 512 * 1024;
  g_setInterval(interval);
  g_setSeed(42);
  g_reset();
  const int n = 200000;
  for (int i = 0; i < n; ++i) {
    double e = 8.0 + (double)(rnd() % 1000000) / 1000000.0 * 18.0;
    size_t size = (size_t)pow(2.0, e);
    alloc_v2(0x100000000ULL + (uint64_t)i * 0x10000000ULL, size, CUDA_SUCCESS);
  }
  {
    struct parcagpu_gpumem_test_stats s = stats();
    double ratio = (double)s.weighted_bytes / (double)s.bytes_seen;
    printf("    n=%d sampled=%llu bytes=%llu weighted=%llu ratio=%.4f\n", n,
           (unsigned long long)s.allocs_sampled,
           (unsigned long long)s.bytes_seen,
           (unsigned long long)s.weighted_bytes, ratio);
    EXPECT(s.allocs_seen == (uint64_t)n, "not every alloc seen");
    EXPECT(s.allocs_sampled > 0 && s.allocs_sampled < (uint64_t)n,
           "sampling did not thin the stream");
    EXPECT(fabs(ratio - 1.0) < 0.03, "weighted bytes off by more than 3%");
  }

  // Small allocations: many below the interval, few sampled, still unbiased.
  g_setSeed(7);
  g_reset();
  for (int i = 0; i < n; ++i)
    alloc_v2(0x200000000ULL + (uint64_t)i * 0x1000ULL, 4096, CUDA_SUCCESS);
  {
    struct parcagpu_gpumem_test_stats s = stats();
    double ratio = (double)s.weighted_bytes / (double)s.bytes_seen;
    double expectSampled = (double)n * -expm1(-4096.0 / (double)interval);
    printf("    small: sampled=%llu (expect ~%.0f) ratio=%.4f\n",
           (unsigned long long)s.allocs_sampled, expectSampled, ratio);
    EXPECT(fabs(ratio - 1.0) < 0.05, "small allocs: weighted off by > 5%");
    EXPECT(fabs((double)s.allocs_sampled - expectSampled) <
               5.0 * sqrt(expectSampled),
           "small allocs: sample count far from expected");
  }

  // Allocations far above the interval are always sampled, at ~their size.
  g_reset();
  for (int i = 0; i < 1000; ++i)
    alloc_v2(0x300000000ULL + (uint64_t)i * 0x10000000ULL, 64 << 20,
             CUDA_SUCCESS);
  {
    struct parcagpu_gpumem_test_stats s = stats();
    EXPECT(s.allocs_sampled == 1000, "64 MiB allocs not always sampled");
    EXPECT(s.weighted_bytes == s.bytes_seen,
           "64 MiB allocs: weight should equal size");
  }
  printf("ok  byte sampling is unbiased\n");

  printf("=== ALL GPU MEMORY TESTS PASSED ===\n");
  return 0;
}
