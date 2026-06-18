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

#include <cupti.h>
// cupti.h transitively includes cupti_nvtx_cbid.h on newer CUDA versions;
// don't include it explicitly to avoid the double-typedef error.

#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Driver/GPU/NvtxApi.h"

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

// Per-event message copy length. NVTX messages in real workloads
// (PyTorch op names, NCCL phase tags) sit well under this. We truncate
// rather than spill into a variable-size pool — the BPF reader uses the
// same cap with bpf_probe_read_user_str.
constexpr size_t kMsgCap = 240;

// Mirror of nvtxEventAttributes_v2 (nvtx3/nvToolsExt.h). We don't include the
// NVTX header to avoid pulling its inline definitions into the build.
// Layout must match exactly; verified against CUDA 12.x.
struct EventAttribs {
  uint16_t version;
  uint16_t size;
  uint32_t category;
  int32_t  colorType;     // NVTX_COLOR_ARGB == 1
  uint32_t color;
  int32_t  payloadType;
  int32_t  reserved0;
  union {
    uint64_t ull;
    int64_t  ll;
    double   d;
    uint32_t ui;
    int32_t  i;
    float    f;
  } payload;
  int32_t  messageType;   // NVTX_MESSAGE_TYPE_ASCII == 1
  union {
    const char *ascii;
    const void *wide;      // wchar_t* or registered string handle
  } message;
};

constexpr int32_t kColorArgb       = 1;   // NVTX_COLOR_ARGB
constexpr int32_t kMessageAscii    = 1;   // NVTX_MESSAGE_TYPE_ASCII
constexpr int32_t kMessageRegistered = 3; // NVTX_MESSAGE_TYPE_REGISTERED

// nvtxRangeId_t is uint64_t (per nvToolsExt.h).
using NvtxRangeId = uint64_t;

struct ThreadState {
  uint32_t count;
  struct nvtx_event events[kBatchSize];
  const void *eventPtrs[kBatchSize];
  char messagePool[kBatchSize][kMsgCap];
};

pthread_key_t g_tlsKey;
std::once_flag g_tlsKeyOnce;

// ---- Test-only instrumentation -------------------------------------------
// Production cost is one relaxed atomic increment per event/batch/name fire,
// which is negligible next to the cost of the CUPTI callback itself. The
// `force-capture` flag lets unit tests exercise the collector without an
// attached USDT consumer (no probe semaphore is hot, so the normal early
// exits would skip everything). Accessed exclusively via the extern "C"
// `parcagpu_nvtx_test_*` accessors at the bottom of this file.
std::atomic<uint64_t> g_testEventsCaptured{0};
std::atomic<uint64_t> g_testBatchesFired{0};
std::atomic<uint64_t> g_testResourceNamesFired{0};
std::atomic<bool>     g_testForceCapture{false};

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

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

uint32_t currentDeviceId() {
  // ctxGetCurrent/ctxGetDevice are cheap and safe inside CUPTI callbacks:
  // they don't touch sampling state or any CUPTI buffers. Note that
  // proton::cuda::ctxGetCurrent<false> still throws if libcuda.so.1 itself
  // can't be loaded or doesn't expose the symbol (the <false> only
  // suppresses CUDA-return-value checks). Wrap the whole thing in a
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
}

void destroyThreadState(void *p) {
  auto *s = static_cast<ThreadState *>(p);
  if (!s)
    return;
  flushBatchLocked(s);
  std::free(s);
}

ThreadState *getThreadState() {
  std::call_once(g_tlsKeyOnce,
                 [] { pthread_key_create(&g_tlsKey, destroyThreadState); });
  auto *s = static_cast<ThreadState *>(pthread_getspecific(g_tlsKey));
  if (!s) {
    s = static_cast<ThreadState *>(std::calloc(1, sizeof(ThreadState)));
    if (s) {
      pthread_setspecific(g_tlsKey, s);
    }
  }
  return s;
}

void copyMessage(char *dst, const char *src) {
  if (!src) {
    dst[0] = '\0';
    return;
  }
  size_t i = 0;
  while (i + 1 < kMsgCap && src[i] != '\0') {
    dst[i] = src[i];
    ++i;
  }
  dst[i] = '\0';
}

void captureEvent(uint16_t kind, uint64_t rangeId, uint32_t color,
                  uint64_t payload, const char *message,
                  uint32_t correlationId) {
  DEBUG_PRINTF("[PARCAGPU] nvtx in: kind=%u corr=%u range_id=%llu msg=%s\n",
               (unsigned)kind, correlationId,
               (unsigned long long)rangeId, message ? message : "(null)");
  if (!captureEnabled())
    return;
  auto *s = getThreadState();
  if (!s)
    return;
  const uint32_t i = s->count;
  struct nvtx_event *ev = &s->events[i];
  ev->timestamp_ns   = nowNs();
  ev->tid            = static_cast<uint32_t>(syscall(SYS_gettid));
  ev->correlation_id = correlationId;
  ev->domain_id      = 0;
  ev->kind           = kind;
  ev->flags          = 0;
  ev->color          = color;
  ev->device_id      = currentDeviceId();
  ev->payload        = payload;
  ev->range_id       = rangeId;
  copyMessage(s->messagePool[i], message);
  ev->message        = s->messagePool[i];
  s->eventPtrs[i]    = ev;
  s->count           = i + 1;
  g_testEventsCaptured.fetch_add(1, std::memory_order_relaxed);
  DEBUG_PRINTF("[PARCAGPU] nvtx capture: kind=%u tid=%u corr=%u dev=%u range_id=%llu msg=%s\n",
               (unsigned)kind, ev->tid, ev->correlation_id, ev->device_id,
               (unsigned long long)ev->range_id, s->messagePool[i]);
  if (s->count == kBatchSize) {
    flushBatchLocked(s);
  }
}

void captureFromEx(uint16_t kind, uint64_t rangeId, const void *attrPtr,
                   uint32_t correlationId) {
  uint32_t color = 0;
  uint64_t payload = 0;
  const char *message = nullptr;
  if (attrPtr) {
    const auto *a = static_cast<const EventAttribs *>(attrPtr);
    if (a->colorType == kColorArgb)
      color = a->color;
    payload = a->payload.ull;
    // Registered-string and wide-string messages are not decoded in v1;
    // they show up in the log as empty body, which is honest about what
    // we know without false data.
    if (a->messageType == kMessageAscii)
      message = a->message.ascii;
  }
  captureEvent(kind, rangeId, color, payload, message, correlationId);
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

constexpr CUpti_CallbackId kEnabledCbids[] = {
    CUPTI_CBID_NVTX_nvtxMarkA,
    CUPTI_CBID_NVTX_nvtxMarkEx,
    CUPTI_CBID_NVTX_nvtxRangeStartA,
    CUPTI_CBID_NVTX_nvtxRangeStartEx,
    CUPTI_CBID_NVTX_nvtxRangeEnd,
    CUPTI_CBID_NVTX_nvtxRangePushA,
    CUPTI_CBID_NVTX_nvtxRangePushEx,
    CUPTI_CBID_NVTX_nvtxRangePop,
    CUPTI_CBID_NVTX_nvtxNameOsThreadA,
    CUPTI_CBID_NVTX_nvtxNameCuStreamA,
    CUPTI_CBID_NVTX_nvtxNameCudaStreamA,
    CUPTI_CBID_NVTX_nvtxNameCuDeviceA,
    CUPTI_CBID_NVTX_nvtxNameCudaDeviceA,
};

} // namespace

bool NvtxCollector::enable(CUpti_SubscriberHandle subscriber) {
  DEBUG_PRINTF("[PARCAGPU] NvtxCollector::enable() starting\n");
  proton::nvtx::enable();
  const char *injectPath = getenv("NVTX_INJECTION64_PATH");
  DEBUG_PRINTF("[PARCAGPU] NVTX_INJECTION64_PATH=%s\n",
               injectPath ? injectPath : "(unset)");
  bool ok = true;
  for (auto cbid : kEnabledCbids) {
    if (proton::cupti::enableCallback<false>(
            /*enable=*/1, subscriber, CUPTI_CB_DOMAIN_NVTX, cbid) !=
        CUPTI_SUCCESS) {
      ok = false;
    }
  }
  DEBUG_PRINTF("[PARCAGPU] NvtxCollector::enable() done ok=%d cbids=%zu\n",
               (int)ok, sizeof(kEnabledCbids) / sizeof(kEnabledCbids[0]));
  return ok;
}

void NvtxCollector::disable(CUpti_SubscriberHandle subscriber) {
  for (auto cbid : kEnabledCbids) {
    proton::cupti::enableCallback<false>(/*enable=*/0, subscriber,
                                         CUPTI_CB_DOMAIN_NVTX, cbid);
  }
  proton::nvtx::disable();
}

void NvtxCollector::flushCurrentThread() {
  auto *s = static_cast<ThreadState *>(pthread_getspecific(g_tlsKey));
  if (s)
    flushBatchLocked(s);
}

void NvtxCollector::onCallback(CUpti_CallbackId cbid, const void *cbdata_void,
                               uint32_t correlationId) {
  // Both semaphores cold? Nothing to do.
  if (!captureEnabled() && !resourceNameEnabled())
    return;

  const auto *nvtxData = static_cast<const CUpti_NvtxData *>(cbdata_void);
  if (!nvtxData)
    return;
  const void *params = nvtxData->functionParams;
  if (!params)
    return;

  switch (cbid) {
  // ---- Marks ------------------------------------------------------------
  case CUPTI_CBID_NVTX_nvtxMarkA: {
    struct P { const char *message; };
    const auto *p = static_cast<const P *>(params);
    captureEvent(NVTX_KIND_MARK, 0, 0, 0, p->message, correlationId);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxMarkEx: {
    struct P { const void *eventAttrib; };
    const auto *p = static_cast<const P *>(params);
    captureFromEx(NVTX_KIND_MARK, 0, p->eventAttrib, correlationId);
    break;
  }

  // ---- Stacked range push/pop ------------------------------------------
  case CUPTI_CBID_NVTX_nvtxRangePushA: {
    struct P { const char *message; };
    const auto *p = static_cast<const P *>(params);
    captureEvent(NVTX_KIND_RANGE_PUSH, 0, 0, 0, p->message, correlationId);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxRangePushEx: {
    struct P { const void *eventAttrib; };
    const auto *p = static_cast<const P *>(params);
    captureFromEx(NVTX_KIND_RANGE_PUSH, 0, p->eventAttrib, correlationId);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxRangePop: {
    captureEvent(NVTX_KIND_RANGE_POP, 0, 0, 0, nullptr, correlationId);
    break;
  }

  // ---- Non-stacked range start/end -------------------------------------
  case CUPTI_CBID_NVTX_nvtxRangeStartA: {
    struct P { const char *message; };
    const auto *p = static_cast<const P *>(params);
    NvtxRangeId id = 0;
    if (nvtxData->functionReturnValue)
      id = *static_cast<const NvtxRangeId *>(nvtxData->functionReturnValue);
    captureEvent(NVTX_KIND_RANGE_START, id, 0, 0, p->message, correlationId);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxRangeStartEx: {
    struct P { const void *eventAttrib; };
    const auto *p = static_cast<const P *>(params);
    NvtxRangeId id = 0;
    if (nvtxData->functionReturnValue)
      id = *static_cast<const NvtxRangeId *>(nvtxData->functionReturnValue);
    captureFromEx(NVTX_KIND_RANGE_START, id, p->eventAttrib, correlationId);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxRangeEnd: {
    struct P { NvtxRangeId id; };
    const auto *p = static_cast<const P *>(params);
    captureEvent(NVTX_KIND_RANGE_END, p->id, 0, 0, nullptr, correlationId);
    break;
  }

  // ---- Resource naming (one-shot, no batching) -------------------------
  case CUPTI_CBID_NVTX_nvtxNameOsThreadA: {
    struct P { uint32_t threadId; const char *name; };
    const auto *p = static_cast<const P *>(params);
    emitResourceName(NVTX_RESOURCE_OS_THREAD, p->threadId, p->name);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxNameCuStreamA: {
    struct P { void *stream; const char *name; };
    const auto *p = static_cast<const P *>(params);
    emitResourceName(NVTX_RESOURCE_CUDA_STREAM,
                     reinterpret_cast<uint64_t>(p->stream), p->name);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxNameCudaStreamA: {
    struct P { void *stream; const char *name; };
    const auto *p = static_cast<const P *>(params);
    emitResourceName(NVTX_RESOURCE_CUDA_STREAM,
                     reinterpret_cast<uint64_t>(p->stream), p->name);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxNameCuDeviceA: {
    struct P { int device; const char *name; };
    const auto *p = static_cast<const P *>(params);
    emitResourceName(NVTX_RESOURCE_CUDA_DEVICE,
                     static_cast<uint64_t>(p->device), p->name);
    break;
  }
  case CUPTI_CBID_NVTX_nvtxNameCudaDeviceA: {
    struct P { int device; const char *name; };
    const auto *p = static_cast<const P *>(params);
    emitResourceName(NVTX_RESOURCE_CUDA_DEVICE,
                     static_cast<uint64_t>(p->device), p->name);
    break;
  }

  default:
    break;
  }
}

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

void parcagpu_nvtx_test_reset(void) {
  parcagpu::g_testEventsCaptured.store(0, std::memory_order_relaxed);
  parcagpu::g_testBatchesFired.store(0, std::memory_order_relaxed);
  parcagpu::g_testResourceNamesFired.store(0, std::memory_order_relaxed);
}

} // extern "C"
