// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstring>
#include <ctime>
#include <mutex>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

// probes.h pulls in <sys/sdt.h> with semaphore support; must precede any
// other header that might pull in sdt.h without semaphores.
#include "probes.h"

#include <dlfcn.h>

#include "Driver/GPU/CudaApi.h"

#include "nvtx.h"
#include "nvtx_collector.h"
#include "pc_sampling.h" // DEBUG_PRINTF, parcagpu::debug_enabled

// CMake defines this at configure time. Default here is a clangd/IDE
// convenience only; real builds get the value from CMakeLists.txt.
#ifndef PARCAGPU_NVTX_BATCH_SIZE
#define PARCAGPU_NVTX_BATCH_SIZE 256
#endif

// Locked wire format with parcagpu/ebpf/cupti_bpf.h `struct cupti_nvtx_event`.
// If you change struct nvtx_event in nvtx.h, update the BPF mirror too.
static_assert(sizeof(struct nvtx_event) == 56,
              "struct nvtx_event size must match cupti_nvtx_event in BPF mirror");

namespace parcagpu {

namespace {

constexpr uint32_t kBatchSize = PARCAGPU_NVTX_BATCH_SIZE;

// Longest message copied per event, including the NUL; longer messages are
// truncated. Most NVTX messages (PyTorch op names, NCCL phase tags) are
// short, but vLLM's layerwise tracing emits dict strings of several hundred
// bytes. The BPF reader uses the same cap (NVTX_MAX_MSG_LEN).
constexpr size_t kMsgCap = 1024;

// Messages of the pending batch are packed into a per-thread arena, so short
// messages take only their own length and there is no allocation per event.
// A batch fires early when the arena might not fit another kMsgCap message.
constexpr size_t kArenaSize = 64 * 1024;
static_assert(kArenaSize >= kMsgCap, "arena must hold at least one message");

// A batch also fires once its oldest event is this old, checked as events
// arrive, so threads that annotate a few times a second (e.g. one range per
// inference step) still report promptly. Bounds probe fires to ~10/s/thread.
constexpr uint64_t kMaxBatchAgeNs = 100ULL * 1'000'000ULL;

struct ThreadState {
  uint32_t count;
  uint32_t arenaUsed; // bytes of arena holding the pending batch's messages
  uint64_t firstNs;   // timestamp of events[0] in the pending batch
  struct nvtx_event events[kBatchSize];
  const void *eventPtrs[kBatchSize];
  char arena[kArenaSize];
};

pthread_key_t g_tlsKey;
std::once_flag g_tlsKeyOnce;
std::atomic<bool> g_tlsKeyCreated{false};

// ---- Test-only instrumentation -------------------------------------------
// Production cost is one relaxed atomic increment per event/batch/name fire,
// which is negligible next to the rest of captureEvent. The
// `force-capture` flag lets unit tests exercise the collector without an
// attached USDT consumer (no probe semaphore is hot, so the normal early
// exits would skip everything). Accessed exclusively via the extern "C"
// `parcagpu_nvtx_test_*` accessors at the bottom of this file.
std::atomic<uint64_t> g_testEventsCaptured{0};
std::atomic<uint64_t> g_testBatchesFired{0};
std::atomic<uint64_t> g_testResourceNamesFired{0};
std::atomic<bool>     g_testForceCapture{false};
// Copy of the most recently captured event (and its message), for tests that
// check field values rather than counts. Only written when force-capture is on.
struct nvtx_event g_testLastEvent;
char g_testLastMessage[kMsgCap];

// Set once CUDA has called InitializeInjection. Before that, NVTX events can
// arrive (NVTX clients initialize at their first call, often at import time),
// and resolving the device would mean loading libcuda ourselves.
std::atomic<bool> g_cudaReady{false};

} // namespace

namespace nvtx {

bool captureEnabled() {
  return PARCAGPU_NVTX_EVENT_BATCH_ENABLED() ||
         g_testForceCapture.load(std::memory_order_relaxed) ||
         debug_enabled;
}
bool resourceNameEnabled() {
  return PARCAGPU_NVTX_RESOURCE_NAME_ENABLED() ||
         g_testForceCapture.load(std::memory_order_relaxed) ||
         debug_enabled;
}

} // namespace nvtx

namespace {

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

uint32_t currentDeviceId() {
  if (!g_cudaReady.load(std::memory_order_acquire))
    return UINT32_MAX;
  // ctxGetCurrent/ctxGetDevice are cheap and don't touch sampling state or
  // any CUPTI buffers. Note that proton::cuda::ctxGetCurrent<false> still
  // throws if libcuda.so.1 itself can't be loaded or doesn't expose the
  // symbol (the <false> only suppresses CUDA-return-value checks). Wrap the whole thing in a
  // catch-all: NVTX device-id resolution is best-effort, and a UINT32_MAX
  // sentinel is fine for downstream consumers.
  try {
    CUcontext ctx = nullptr;
    if (proton::cuda::ctxGetCurrent<false>(&ctx) != CUDA_SUCCESS || !ctx) {
      return UINT32_MAX;
    }
    CUdevice dev = 0;
    if (proton::cuda::ctxGetDevice<false>(&dev) != CUDA_SUCCESS) {
      return UINT32_MAX;
    }
    return static_cast<uint32_t>(dev);
  } catch (...) {
    return UINT32_MAX;
  }
}

__attribute__((noinline)) void fireNvtxEventBatch(const void **ptrs,
                                                  uint32_t count) {
  PARCAGPU_NVTX_EVENT_BATCH(ptrs, count);
}

__attribute__((noinline)) void fireNvtxResourceName(uint32_t kind, uint64_t id,
                                                    const char *name) {
  PARCAGPU_NVTX_RESOURCE_NAME(kind, id, name);
}

void flushBatchLocked(ThreadState *s) {
  if (s->count == 0)
    return;
  DEBUG_PRINTF("[PARCAGPU] nvtx batch fire: count=%u\n", s->count);
  fireNvtxEventBatch(s->eventPtrs, s->count);
  g_testBatchesFired.fetch_add(1, std::memory_order_relaxed);
  s->count = 0;
  s->arenaUsed = 0;
}

void destroyThreadState(void *p) {
  auto *s = static_cast<ThreadState *>(p);
  if (!s)
    return;
  flushBatchLocked(s);
  std::free(s);
}

ThreadState *getThreadState() {
  std::call_once(g_tlsKeyOnce, [] {
    if (pthread_key_create(&g_tlsKey, destroyThreadState) == 0)
      g_tlsKeyCreated.store(true, std::memory_order_release);
  });
  if (!g_tlsKeyCreated.load(std::memory_order_acquire))
    return nullptr;
  auto *s = static_cast<ThreadState *>(pthread_getspecific(g_tlsKey));
  if (!s) {
    s = static_cast<ThreadState *>(std::calloc(1, sizeof(ThreadState)));
    if (s) {
      pthread_setspecific(g_tlsKey, s);
    }
  }
  return s;
}

// Copies src (truncated to kMsgCap - 1 bytes) and returns the bytes written,
// including the NUL.
size_t copyMessage(char *dst, const char *src) {
  size_t i = 0;
  if (src) {
    while (i + 1 < kMsgCap && src[i] != '\0') {
      dst[i] = src[i];
      ++i;
    }
  }
  dst[i] = '\0';
  return i + 1;
}

} // namespace

namespace nvtx {

void captureEvent(uint16_t kind, uint32_t domainId, uint64_t rangeId,
                  uint32_t color, uint64_t payload, const char *message) {
  if (!captureEnabled())
    return;
  auto *s = getThreadState();
  if (!s)
    return;
  if (kArenaSize - s->arenaUsed < kMsgCap)
    flushBatchLocked(s);
  const uint32_t i = s->count;
  struct nvtx_event *ev = &s->events[i];
  ev->timestamp_ns   = nowNs();
  ev->tid            = static_cast<uint32_t>(syscall(SYS_gettid));
  ev->correlation_id = currentRuntimeCorrelationId();
  ev->domain_id      = domainId;
  ev->kind           = kind;
  ev->flags          = 0;
  ev->color          = color;
  ev->device_id      = currentDeviceId();
  ev->payload        = payload;
  ev->range_id       = rangeId;
  char *msg          = s->arena + s->arenaUsed;
  s->arenaUsed      += static_cast<uint32_t>(copyMessage(msg, message));
  ev->message        = msg;
  s->eventPtrs[i]    = ev;
  s->count           = i + 1;
  if (i == 0)
    s->firstNs = ev->timestamp_ns;
  if (g_testForceCapture.load(std::memory_order_relaxed)) {
    g_testLastEvent = *ev;
    copyMessage(g_testLastMessage, msg);
  }
  g_testEventsCaptured.fetch_add(1, std::memory_order_relaxed);
  DEBUG_PRINTF("[PARCAGPU] nvtx capture: kind=%u domain=%u tid=%u corr=%u dev=%u range_id=%llu msg=%s\n",
               (unsigned)kind, domainId, ev->tid, ev->correlation_id,
               ev->device_id, (unsigned long long)ev->range_id,
               msg);
  if (s->count == kBatchSize ||
      ev->timestamp_ns - s->firstNs >= kMaxBatchAgeNs) {
    flushBatchLocked(s);
  }
}

void emitResourceName(uint32_t kind, uint64_t id, const char *name) {
  if (!resourceNameEnabled())
    return;
  if (!name)
    name = "";
  DEBUG_PRINTF("[PARCAGPU] nvtx name: kind=%u id=0x%llx name=%s\n",
               kind, (unsigned long long)id, name);
  fireNvtxResourceName(kind, id, name);
  g_testResourceNamesFired.fetch_add(1, std::memory_order_relaxed);
}

void onCudaInjected() {
  g_cudaReady.store(true, std::memory_order_release);

  // Point NVTX clients that haven't initialized yet at this library. Clients
  // that already made their first NVTX call before cuInit only see us if the
  // launcher set NVTX_INJECTION64_PATH; that's the recommended setup. Never
  // override an existing value (e.g. nsys's own injection).
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&onCudaInjected), &info) &&
      info.dli_fname) {
    setenv("NVTX_INJECTION64_PATH", info.dli_fname, /*overwrite=*/0);
  }
  const char *injectPath = getenv("NVTX_INJECTION64_PATH");
  DEBUG_PRINTF("[PARCAGPU] NVTX_INJECTION64_PATH=%s\n",
               injectPath ? injectPath : "(unset)");
}

void flushCurrentThread() {
  if (!g_tlsKeyCreated.load(std::memory_order_acquire))
    return;
  auto *s = static_cast<ThreadState *>(pthread_getspecific(g_tlsKey));
  if (s)
    flushBatchLocked(s);
}

} // namespace nvtx

} // namespace parcagpu

// ---- Test-only ABI -------------------------------------------------------
// Exposed via dlsym from the unit-test harness. Not part of the public API;
// not referenced by the production reporter pipeline. Wrapped in extern "C"
// so callers can resolve them by unmangled name.
extern "C" {

void parcagpu_nvtx_test_force_capture(int on) {
  parcagpu::g_testForceCapture.store(on != 0, std::memory_order_relaxed);
}

uint64_t parcagpu_nvtx_test_events_captured(void) {
  return parcagpu::g_testEventsCaptured.load(std::memory_order_relaxed);
}

uint64_t parcagpu_nvtx_test_batches_fired(void) {
  return parcagpu::g_testBatchesFired.load(std::memory_order_relaxed);
}

uint64_t parcagpu_nvtx_test_resource_names_fired(void) {
  return parcagpu::g_testResourceNamesFired.load(std::memory_order_relaxed);
}

// Copies the most recently captured event; returns its message (valid until
// the next capture).
const char *parcagpu_nvtx_test_last_event(struct nvtx_event *out) {
  if (out)
    *out = parcagpu::g_testLastEvent;
  return parcagpu::g_testLastMessage;
}

void parcagpu_nvtx_test_reset(void) {
  parcagpu::g_testEventsCaptured.store(0, std::memory_order_relaxed);
  parcagpu::g_testBatchesFired.store(0, std::memory_order_relaxed);
  parcagpu::g_testResourceNamesFired.store(0, std::memory_order_relaxed);
}

} // extern "C"
