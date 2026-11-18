// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

// probes.h / otel_gpu_memory_probes.h must come before anything that might
// include <sys/sdt.h>, so that _SDT_HAS_SEMAPHORES is defined first.
#include "otel_gpu_memory_probes.h"
#include "probes.h"

#include "gpu_memory.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>

#include <cuda.h>

#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "pc_sampling.h" // DEBUG_PRINTF, fireError

// ---------------------------------------------------------------------------
// Probe sinks. One call site per probe so each probe has a single location
// in .note.stapsdt (one uprobe per consumer).
//
// The otel_gpu_memory sinks follow opentelemetry-ebpf-profiler #1747, whose
// eBPF reads the arguments straight from the ABI argument registers and
// ignores the SDT descriptors. These functions are noinline, so their
// arguments arrive in exactly those registers; the register-pinned locals
// keep the compiler from moving them before the probe site. The build
// verifies the resulting notes (scripts/check-otel-usdt-regs.sh).
// ---------------------------------------------------------------------------

#if defined(__x86_64__)
#define PARCAGPU_ARG0_REG "rdi"
#define PARCAGPU_ARG1_REG "rsi"
#define PARCAGPU_ARG2_REG "rdx"
#elif defined(__aarch64__)
#define PARCAGPU_ARG0_REG "x0"
#define PARCAGPU_ARG1_REG "x1"
#define PARCAGPU_ARG2_REG "x2"
#else
#error "Unsupported architecture for otel_gpu_memory probes"
#endif

extern "C" {

__attribute__((noinline)) void otelGpuMemAlloc(uint64_t ptr, uint64_t size,
                                               uint64_t weightedBytes) {
  register uint64_t a0 asm(PARCAGPU_ARG0_REG) = ptr;
  register uint64_t a1 asm(PARCAGPU_ARG1_REG) = size;
  register uint64_t a2 asm(PARCAGPU_ARG2_REG) = weightedBytes;
  OTEL_GPU_MEMORY_ALLOC((void *)a0, a1, a2);
}

__attribute__((noinline)) void otelGpuMemFree(uint64_t ptr) {
  register uint64_t a0 asm(PARCAGPU_ARG0_REG) = ptr;
  OTEL_GPU_MEMORY_FREE((void *)a0);
}

__attribute__((noinline)) void parcagpuGpuMemAlloc(uint64_t ptr, uint64_t size,
                                                   uint64_t weightedBytes,
                                                   uint32_t deviceId,
                                                   uint32_t kind) {
  PARCAGPU_GPU_MEM_ALLOC(ptr, size, weightedBytes, deviceId, kind);
}

__attribute__((noinline)) void parcagpuGpuMemFree(uint64_t ptr,
                                                  uint32_t deviceId) {
  PARCAGPU_GPU_MEM_FREE(ptr, deviceId);
}

} // extern "C"

namespace parcagpu {
namespace gpu_memory {

namespace {

// Mean bytes between samples; 0 samples every allocation.
std::atomic<uint64_t> g_interval{kDefaultSamplingInterval};

// Set by configure() when PARCAGPU_MEM_SAMPLING_INTERVAL is valid.
std::atomic<bool> g_enabled{false};

// Test hooks (see the extern "C" block at the bottom).
std::atomic<bool> g_testForceCapture{false};
std::atomic<uint64_t> g_testSeed{0};
std::atomic<uint64_t> g_testAllocsSeen{0};
std::atomic<uint64_t> g_testAllocsSampled{0};
std::atomic<uint64_t> g_testBytesSeen{0};
std::atomic<uint64_t> g_testWeightedBytes{0};
std::atomic<uint64_t> g_testFrees{0};
std::atomic<uint64_t> g_testLastPtr{0};
std::atomic<uint64_t> g_testLastWeight{0};
std::atomic<uint32_t> g_testLastKind{0};

// Per-thread sampler state.
struct SamplerState {
  uint64_t rng = 0;            // xorshift64 state; 0 = unseeded
  uint64_t seedGeneration = 0; // g_testSeed value the state was seeded from
  int64_t bytesUntilSample = 0;
  bool primed = false; // bytesUntilSample has been drawn
};
thread_local SamplerState t_sampler;

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// (Re)seed when the thread first samples or a test changed the seed. A
// reseed also discards the countdown, so a seeded test run is repeatable.
void ensureSeeded(SamplerState &s) {
  uint64_t seed = g_testSeed.load(std::memory_order_relaxed);
  if (s.rng != 0 && s.seedGeneration == seed)
    return;
  s.rng = seed ? seed : (nowNs() ^ (uint64_t)(uintptr_t)&s);
  if (s.rng == 0)
    s.rng = 0x9E3779B97F4A7C15ULL;
  s.seedGeneration = seed;
  s.primed = false;
}

// xorshift64 (Marsaglia). rand_r's 31 bits are too coarse for -log(u) tails.
uint64_t nextRandom(SamplerState &s) {
  uint64_t x = s.rng;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  s.rng = x;
  return x;
}

// Draw the byte gap to the next sample from Exp(mean = interval).
int64_t nextGap(SamplerState &s, uint64_t interval) {
  // u in (0, 1]: never 0, so log(u) stays finite.
  double u = (double)((nextRandom(s) >> 11) + 1) * 0x1.0p-53;
  double gap = -std::log(u) * (double)interval;
  if (gap < 1.0)
    return 1;
  if (gap > 9.0e18)
    return INT64_MAX;
  return (int64_t)gap;
}

// Decide whether an allocation of `size` bytes is sampled. On a hit, return
// the unbiased byte estimate in *weight.
//
// Allocations walk a Poisson process over bytes with rate 1/interval; an
// allocation is sampled when a sample point lands inside it, which happens
// with probability p = 1 - exp(-size/interval). Reporting size/p keeps the
// expected sum equal to the true byte total (opentelemetry-ebpf-profiler
// ADR 00003, the same estimator tcmalloc and jemalloc use).
bool sample(uint64_t size, uint64_t *weight) {
  uint64_t interval = g_interval.load(std::memory_order_relaxed);
  if (interval == 0) {
    *weight = size;
    return true;
  }
  if (size == 0)
    return false;

  auto &s = t_sampler;
  ensureSeeded(s);
  if (!s.primed) {
    s.bytesUntilSample = nextGap(s, interval);
    s.primed = true;
  }

  if (size < (uint64_t)s.bytesUntilSample) {
    s.bytesUntilSample -= (int64_t)size;
    return false;
  }
  s.bytesUntilSample = nextGap(s, interval);

  double p = -std::expm1(-(double)size / (double)interval);
  double w = p > 0.0 ? (double)size / p : (double)size;
  *weight = w >= 1.8e19 ? UINT64_MAX : (uint64_t)std::llround(w);
  if (*weight < size)
    *weight = size;
  return true;
}

bool otelAllocEnabled() { return OTEL_GPU_MEMORY_ALLOC_ENABLED(); }
bool otelFreeEnabled() { return OTEL_GPU_MEMORY_FREE_ENABLED(); }
bool nativeAllocEnabled() { return PARCAGPU_GPU_MEM_ALLOC_ENABLED(); }
bool nativeFreeEnabled() { return PARCAGPU_GPU_MEM_FREE_ENABLED(); }
bool forced() { return g_testForceCapture.load(std::memory_order_relaxed); }

// Device of the current context, cached per thread by context.
uint32_t currentDevice(CUcontext ctx) {
  thread_local CUcontext cachedCtx = nullptr;
  thread_local uint32_t cachedDev = 0;
  if (ctx != nullptr && ctx == cachedCtx)
    return cachedDev;
  CUdevice dev = 0;
  if (proton::cuda::ctxGetDevice<false>(&dev) != CUDA_SUCCESS)
    return 0;
  cachedCtx = ctx;
  cachedDev = (uint32_t)dev;
  return cachedDev;
}

void emitAlloc(const CUpti_CallbackData *cbdata, uint64_t ptr, uint64_t size,
               Kind kind) {
  const bool otel = otelAllocEnabled();
  const bool native = nativeAllocEnabled();
  const bool force = forced();
  if (!otel && !native && !force)
    return;
  if (ptr == 0)
    return;

  if (force) {
    g_testAllocsSeen.fetch_add(1, std::memory_order_relaxed);
    g_testBytesSeen.fetch_add(size, std::memory_order_relaxed);
  }

  uint64_t weight = 0;
  if (!sample(size, &weight))
    return;

  DEBUG_PRINTF("[PARCAGPU] GPU alloc sampled: ptr=0x%lx size=%lu weight=%lu "
               "kind=%u\n",
               (unsigned long)ptr, (unsigned long)size, (unsigned long)weight,
               (unsigned)kind);

  if (otel)
    otelGpuMemAlloc(ptr, size, weight);
  if (native || force) {
    uint32_t dev = currentDevice(cbdata->context);
    parcagpuGpuMemAlloc(ptr, size, weight, dev, kind);
  }

  if (force) {
    g_testAllocsSampled.fetch_add(1, std::memory_order_relaxed);
    g_testWeightedBytes.fetch_add(weight, std::memory_order_relaxed);
    g_testLastPtr.store(ptr, std::memory_order_relaxed);
    g_testLastWeight.store(weight, std::memory_order_relaxed);
    g_testLastKind.store(kind, std::memory_order_relaxed);
  }
}

void emitFree(const CUpti_CallbackData *cbdata, uint64_t ptr) {
  if (ptr == 0)
    return;
  const bool force = forced();
  if (otelFreeEnabled())
    otelGpuMemFree(ptr);
  if (nativeFreeEnabled() || force)
    parcagpuGpuMemFree(ptr, currentDevice(cbdata->context));
  if (force)
    g_testFrees.fetch_add(1, std::memory_order_relaxed);
}

// The cbids we subscribe to. Runtime cudaMalloc/cudaFree/cudaMallocAsync
// all go through these driver calls, so each allocation is seen once.
constexpr CUpti_CallbackId kAllocCbids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemMap,
};
constexpr CUpti_CallbackId kFreeCbids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuMemUnmap,
};

} // namespace

void configure() {
  const char *env = std::getenv("PARCAGPU_MEM_SAMPLING_INTERVAL");
  if (env == nullptr || *env == '\0')
    return;
  char *end = nullptr;
  unsigned long long v = std::strtoull(env, &end, 10);
  if (end == env || *end != '\0' || env[0] == '-')
    return; // validateEnvVars() reports it
  g_interval.store((uint64_t)v, std::memory_order_relaxed);
  g_enabled.store(true, std::memory_order_relaxed);
}

bool enabled() { return g_enabled.load(std::memory_order_relaxed); }

bool isMemCbid(CUpti_CallbackId cbid) {
  for (auto c : kAllocCbids)
    if (c == cbid)
      return true;
  for (auto c : kFreeCbids)
    if (c == cbid)
      return true;
  return false;
}

void setCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
  for (auto c : kAllocCbids)
    proton::cupti::enableCallback<false>(enable ? 1 : 0, subscriber,
                                         CUPTI_CB_DOMAIN_DRIVER_API, c);
  for (auto c : kFreeCbids)
    proton::cupti::enableCallback<false>(enable ? 1 : 0, subscriber,
                                         CUPTI_CB_DOMAIN_DRIVER_API, c);
}

void onCallback(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata) {
  const void *params = cbdata->functionParams;
  if (params == nullptr)
    return;

  // Frees fire on ENTER: once the call returns, another thread can be handed
  // the same address, and its alloc must not land before this free.
  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
      emitFree(cbdata, static_cast<const cuMemFree_v2_params *>(params)->dptr);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync:
      emitFree(cbdata,
               static_cast<const cuMemFreeAsync_params *>(params)->dptr);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz:
      emitFree(cbdata,
               static_cast<const cuMemFreeAsync_ptsz_params *>(params)->dptr);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemUnmap:
      emitFree(cbdata, static_cast<const cuMemUnmap_params *>(params)->ptr);
      break;
    default:
      break;
    }
    return;
  }

  if (cbdata->callbackSite != CUPTI_API_EXIT)
    return;
  // Allocations fire on EXIT, and only when they succeeded: that is when
  // the output pointer is valid.
  const auto *ret = static_cast<const CUresult *>(cbdata->functionReturnValue);
  if (ret == nullptr || *ret != CUDA_SUCCESS)
    return;

  switch (cbid) {
  case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2: {
    auto *p = static_cast<const cuMemAlloc_v2_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindDevice);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2: {
    auto *p = static_cast<const cuMemAllocPitch_v2_params *>(params);
    if (p->dptr && p->pPitch)
      emitAlloc(cbdata, *p->dptr, (uint64_t)*p->pPitch * p->Height,
                kKindDevice);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged: {
    auto *p = static_cast<const cuMemAllocManaged_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindManaged);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync: {
    auto *p = static_cast<const cuMemAllocAsync_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindAsync);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz: {
    auto *p = static_cast<const cuMemAllocAsync_ptsz_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindAsync);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync: {
    auto *p = static_cast<const cuMemAllocFromPoolAsync_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindAsync);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz: {
    auto *p = static_cast<const cuMemAllocFromPoolAsync_ptsz_params *>(params);
    if (p->dptr)
      emitAlloc(cbdata, *p->dptr, p->bytesize, kKindAsync);
    break;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuMemMap: {
    auto *p = static_cast<const cuMemMap_params *>(params);
    emitAlloc(cbdata, p->ptr, p->size, kKindVmm);
    break;
  }
  default:
    break;
  }
}

} // namespace gpu_memory
} // namespace parcagpu

// ---- Test-only ABI -------------------------------------------------------
// Resolved via dlsym by the unit tests; not part of the public API.
extern "C" {

void parcagpu_gpumem_test_force_capture(int on) {
  parcagpu::gpu_memory::g_testForceCapture.store(on != 0,
                                                 std::memory_order_relaxed);
}

// Seed the per-thread RNGs (0 restores time-based seeding) and reset the
// calling thread's countdown.
void parcagpu_gpumem_test_set_seed(uint64_t seed) {
  parcagpu::gpu_memory::g_testSeed.store(seed, std::memory_order_relaxed);
}

void parcagpu_gpumem_test_set_interval(uint64_t interval) {
  parcagpu::gpu_memory::g_interval.store(interval, std::memory_order_relaxed);
}

uint64_t parcagpu_gpumem_test_interval(void) {
  return parcagpu::gpu_memory::g_interval.load(std::memory_order_relaxed);
}

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

void parcagpu_gpumem_test_stats(struct parcagpu_gpumem_test_stats *out) {
  using namespace parcagpu::gpu_memory;
  if (!out)
    return;
  out->allocs_seen = g_testAllocsSeen.load(std::memory_order_relaxed);
  out->allocs_sampled = g_testAllocsSampled.load(std::memory_order_relaxed);
  out->bytes_seen = g_testBytesSeen.load(std::memory_order_relaxed);
  out->weighted_bytes = g_testWeightedBytes.load(std::memory_order_relaxed);
  out->frees = g_testFrees.load(std::memory_order_relaxed);
  out->last_ptr = g_testLastPtr.load(std::memory_order_relaxed);
  out->last_weight = g_testLastWeight.load(std::memory_order_relaxed);
  out->last_kind = g_testLastKind.load(std::memory_order_relaxed);
}

void parcagpu_gpumem_test_reset(void) {
  using namespace parcagpu::gpu_memory;
  g_testAllocsSeen.store(0, std::memory_order_relaxed);
  g_testAllocsSampled.store(0, std::memory_order_relaxed);
  g_testBytesSeen.store(0, std::memory_order_relaxed);
  g_testWeightedBytes.store(0, std::memory_order_relaxed);
  g_testFrees.store(0, std::memory_order_relaxed);
  g_testLastPtr.store(0, std::memory_order_relaxed);
  g_testLastWeight.store(0, std::memory_order_relaxed);
  g_testLastKind.store(0, std::memory_order_relaxed);
}

} // extern "C"
