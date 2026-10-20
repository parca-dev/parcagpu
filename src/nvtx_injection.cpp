// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// NVTX injection: libparcagpucupti.so doubles as an NVTX tool library. NVTX
// v3 is header-only, so every library in the process that uses it (torch,
// the Python nvtx package, cuDNN, TensorRT, ...) carries its own client copy.
// At its first NVTX call, each copy dlopens NVTX_INJECTION64_PATH and calls
// InitializeInjectionNvtx2 with an export table we use to install our handlers.
// Copies that find no injection turn into no-ops for the life of the process,
// which is why launchers should set NVTX_INJECTION64_PATH up front: many
// clients make their first call at import time, long before cuInit loads us
// through CUDA_INJECTION64_PATH.
//
// Being the injection (rather than subscribing to CUPTI's NVTX callback
// domain) means we own domain and registered-string handles, so resolving
// them is a pointer cast, and we see every call from the first one.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <mutex>
#include <string>
#include <unordered_map>

#define NVTX_NO_IMPL
#include <nvtx3/nvToolsExt.h>

#include "nvtx.h"
#include "nvtx_collector.h"
#include "pc_sampling.h" // DEBUG_PRINTF

namespace parcagpu {
namespace nvtx {
namespace {

// Domain handles we hand out. nvtxDomainHandle_t is an opaque pointer type
// owned by the injection; ours point at a Domain. id 0 is the default domain
// (a null handle in the domain APIs, and every non-domain API).
struct Domain {
  uint32_t id;
  const char *name;
  std::atomic<uint64_t> lastAnnouncedNs{0};
};

// How often a domain's name is re-announced on nvtx_resource_name while
// events for it keep arriving. Domains are created once, often before a
// consumer attaches, so a one-shot announcement would usually be missed.
constexpr uint64_t kDomainReannounceNs = 30ULL * 1'000'000'000ULL;

std::mutex g_registryMu;
// Domains and registered strings live for the life of the process: handles
// may be used from any thread at any time, and clients never release
// registered strings. Both maps dedup by name, so repeated registration
// (e.g. the same domain from several NVTX client copies) doesn't grow them.
std::unordered_map<std::string, Domain *> *g_domains;
std::unordered_map<std::string, const char *> *g_strings;
uint32_t g_nextDomainId = 1;

std::atomic<uint64_t> g_nextRangeId{1};
thread_local int t_pushDepth = 0;

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

void announceDomain(Domain *d, uint64_t now) {
  d->lastAnnouncedNs.store(now, std::memory_order_relaxed);
  emitResourceName(NVTX_RESOURCE_DOMAIN, d->id, d->name);
}

uint32_t domainId(nvtxDomainHandle_t h) {
  if (!h)
    return 0;
  auto *d = reinterpret_cast<Domain *>(h);
  if (resourceNameEnabled()) {
    const uint64_t now = nowNs();
    if (now - d->lastAnnouncedNs.load(std::memory_order_relaxed) >
        kDomainReannounceNs) {
      announceDomain(d, now);
    }
  }
  return d->id;
}

// Unpack nvtxEventAttributes_t. A registered message's handle is a pointer
// to our interned copy of the string. Wide strings are not decoded.
void captureAttribs(uint16_t kind, uint32_t domain, uint64_t rangeId,
                    const nvtxEventAttributes_t *a) {
  uint32_t color = 0;
  uint64_t payload = 0;
  const char *message = nullptr;
  if (a) {
    if (a->colorType == NVTX_COLOR_ARGB)
      color = a->color;
    payload = a->payload.ullValue;
    if (a->messageType == NVTX_MESSAGE_TYPE_ASCII)
      message = a->message.ascii;
    else if (a->messageType == NVTX_MESSAGE_TYPE_REGISTERED)
      message = reinterpret_cast<const char *>(a->message.registered);
  }
  captureEvent(kind, domain, rangeId, color, payload, message);
}

// ---- Core (non-domain) API ----------------------------------------------

void markEx(const nvtxEventAttributes_t *a) {
  if (captureEnabled())
    captureAttribs(NVTX_KIND_MARK, 0, 0, a);
}

void markA(const char *message) {
  captureEvent(NVTX_KIND_MARK, 0, 0, 0, 0, message);
}

nvtxRangeId_t rangeStartEx(const nvtxEventAttributes_t *a) {
  const nvtxRangeId_t id = g_nextRangeId.fetch_add(1, std::memory_order_relaxed);
  if (captureEnabled())
    captureAttribs(NVTX_KIND_RANGE_START, 0, id, a);
  return id;
}

nvtxRangeId_t rangeStartA(const char *message) {
  const nvtxRangeId_t id = g_nextRangeId.fetch_add(1, std::memory_order_relaxed);
  captureEvent(NVTX_KIND_RANGE_START, 0, id, 0, 0, message);
  return id;
}

void rangeEnd(nvtxRangeId_t id) {
  captureEvent(NVTX_KIND_RANGE_END, 0, id, 0, 0, nullptr);
}

// Push returns the 0-based level of the range being pushed, pop the level of
// the range being popped (negative on underflow). One counter per thread,
// shared across domains.
int rangePushEx(const nvtxEventAttributes_t *a) {
  if (captureEnabled())
    captureAttribs(NVTX_KIND_RANGE_PUSH, 0, 0, a);
  return t_pushDepth++;
}

int rangePushA(const char *message) {
  captureEvent(NVTX_KIND_RANGE_PUSH, 0, 0, 0, 0, message);
  return t_pushDepth++;
}

int rangePop() {
  captureEvent(NVTX_KIND_RANGE_POP, 0, 0, 0, 0, nullptr);
  return --t_pushDepth;
}

void nameOsThreadA(uint32_t threadId, const char *name) {
  emitResourceName(NVTX_RESOURCE_OS_THREAD, threadId, name);
}

// ---- Domain API -----------------------------------------------------------

void domainMarkEx(nvtxDomainHandle_t d, const nvtxEventAttributes_t *a) {
  if (captureEnabled())
    captureAttribs(NVTX_KIND_MARK, domainId(d), 0, a);
}

nvtxRangeId_t domainRangeStartEx(nvtxDomainHandle_t d,
                                 const nvtxEventAttributes_t *a) {
  const nvtxRangeId_t id = g_nextRangeId.fetch_add(1, std::memory_order_relaxed);
  if (captureEnabled())
    captureAttribs(NVTX_KIND_RANGE_START, domainId(d), id, a);
  return id;
}

void domainRangeEnd(nvtxDomainHandle_t d, nvtxRangeId_t id) {
  if (captureEnabled())
    captureEvent(NVTX_KIND_RANGE_END, domainId(d), id, 0, 0, nullptr);
}

int domainRangePushEx(nvtxDomainHandle_t d, const nvtxEventAttributes_t *a) {
  if (captureEnabled())
    captureAttribs(NVTX_KIND_RANGE_PUSH, domainId(d), 0, a);
  return t_pushDepth++;
}

int domainRangePop(nvtxDomainHandle_t d) {
  if (captureEnabled())
    captureEvent(NVTX_KIND_RANGE_POP, domainId(d), 0, 0, 0, nullptr);
  return --t_pushDepth;
}

nvtxStringHandle_t domainRegisterStringA(nvtxDomainHandle_t, const char *s) {
  if (!s)
    return nullptr;
  try {
    std::lock_guard<std::mutex> lock(g_registryMu);
    if (!g_strings)
      g_strings = new std::unordered_map<std::string, const char *>();
    auto it = g_strings->find(s);
    if (it != g_strings->end())
      return reinterpret_cast<nvtxStringHandle_t>(const_cast<char *>(it->second));
    const char *copy = strdup(s);
    if (!copy)
      return nullptr;
    g_strings->emplace(s, copy);
    return reinterpret_cast<nvtxStringHandle_t>(const_cast<char *>(copy));
  } catch (...) {
    return nullptr;
  }
}

nvtxDomainHandle_t domainCreateA(const char *name) {
  if (!name)
    name = "";
  Domain *d = nullptr;
  try {
    std::lock_guard<std::mutex> lock(g_registryMu);
    if (!g_domains)
      g_domains = new std::unordered_map<std::string, Domain *>();
    auto it = g_domains->find(name);
    if (it != g_domains->end())
      return reinterpret_cast<nvtxDomainHandle_t>(it->second);
    const char *copy = strdup(name);
    if (!copy)
      return nullptr;
    d = new Domain{g_nextDomainId++, copy};
    g_domains->emplace(name, d);
  } catch (...) {
    return nullptr;
  }
  DEBUG_PRINTF("[PARCAGPU] nvtx domain created: id=%u name=%s\n", d->id,
               d->name);
  if (resourceNameEnabled())
    announceDomain(d, nowNs());
  return reinterpret_cast<nvtxDomainHandle_t>(d);
}

// Domains are kept (see the registry comment); destroy is a no-op.
void domainDestroy(nvtxDomainHandle_t) {}

void initialize(const void *) {}

// ---- CUDA driver / runtime resource naming ---------------------------------

void nameCuDeviceA(int device, const char *name) {
  emitResourceName(NVTX_RESOURCE_CUDA_DEVICE, static_cast<uint64_t>(device),
                   name);
}

void nameCuStreamA(void *stream, const char *name) {
  emitResourceName(NVTX_RESOURCE_CUDA_STREAM,
                   reinterpret_cast<uint64_t>(stream), name);
}

// ---- Export table plumbing --------------------------------------------------

template <typename Fn>
NvtxFunctionPointer fp(Fn fn) {
  return reinterpret_cast<NvtxFunctionPointer>(fn);
}

struct Slot {
  unsigned int cbid;
  NvtxFunctionPointer fn;
};

// Install handlers for one module. Slots the client doesn't have (older
// NVTX) are skipped; anything we don't assign the client turns into a no-op.
bool assignModule(const NvtxExportTableCallbacks *etbl, NvtxCallbackModule m,
                  const Slot *slots, size_t n) {
  NvtxFunctionTable table = nullptr;
  unsigned int size = 0;
  if (!etbl->GetModuleFunctionTable(m, &table, &size) || !table)
    return false;
  for (size_t i = 0; i < n; ++i) {
    if (slots[i].cbid < size && table[slots[i].cbid])
      *table[slots[i].cbid] = slots[i].fn;
  }
  return true;
}

} // namespace
} // namespace nvtx
} // namespace parcagpu

// Called by each NVTX client copy at its first NVTX call. Must not throw: the
// caller is C. Returns 1 so the client keeps our handlers; 0 makes it fall
// back to no-ops.
extern "C" int InitializeInjectionNvtx2(NvtxGetExportTableFunc_t getExportTable) {
  using namespace parcagpu::nvtx;
  if (!getExportTable)
    return 0;
  const auto *etbl = static_cast<const NvtxExportTableCallbacks *>(
      getExportTable(NVTX_ETID_CALLBACKS));
  if (!etbl || etbl->struct_size < sizeof(NvtxExportTableCallbacks) ||
      !etbl->GetModuleFunctionTable)
    return 0;

  static const Slot core[] = {
      {NVTX_CBID_CORE_MarkEx, fp(&markEx)},
      {NVTX_CBID_CORE_MarkA, fp(&markA)},
      {NVTX_CBID_CORE_RangeStartEx, fp(&rangeStartEx)},
      {NVTX_CBID_CORE_RangeStartA, fp(&rangeStartA)},
      {NVTX_CBID_CORE_RangeEnd, fp(&rangeEnd)},
      {NVTX_CBID_CORE_RangePushEx, fp(&rangePushEx)},
      {NVTX_CBID_CORE_RangePushA, fp(&rangePushA)},
      {NVTX_CBID_CORE_RangePop, fp(&rangePop)},
      {NVTX_CBID_CORE_NameOsThreadA, fp(&nameOsThreadA)},
  };
  static const Slot core2[] = {
      {NVTX_CBID_CORE2_DomainMarkEx, fp(&domainMarkEx)},
      {NVTX_CBID_CORE2_DomainRangeStartEx, fp(&domainRangeStartEx)},
      {NVTX_CBID_CORE2_DomainRangeEnd, fp(&domainRangeEnd)},
      {NVTX_CBID_CORE2_DomainRangePushEx, fp(&domainRangePushEx)},
      {NVTX_CBID_CORE2_DomainRangePop, fp(&domainRangePop)},
      {NVTX_CBID_CORE2_DomainRegisterStringA, fp(&domainRegisterStringA)},
      {NVTX_CBID_CORE2_DomainCreateA, fp(&domainCreateA)},
      {NVTX_CBID_CORE2_DomainDestroy, fp(&domainDestroy)},
      {NVTX_CBID_CORE2_Initialize, fp(&initialize)},
  };
  static const Slot cuda[] = {
      {NVTX_CBID_CUDA_NameCuDeviceA, fp(&nameCuDeviceA)},
      {NVTX_CBID_CUDA_NameCuStreamA, fp(&nameCuStreamA)},
  };
  static const Slot cudart[] = {
      {NVTX_CBID_CUDART_NameCudaDeviceA, fp(&nameCuDeviceA)},
      {NVTX_CBID_CUDART_NameCudaStreamA, fp(&nameCuStreamA)},
  };

  // CORE is the one module every client has; the rest are optional.
  if (!assignModule(etbl, NVTX_CB_MODULE_CORE, core, std::size(core)))
    return 0;
  assignModule(etbl, NVTX_CB_MODULE_CORE2, core2, std::size(core2));
  assignModule(etbl, NVTX_CB_MODULE_CUDA, cuda, std::size(cuda));
  assignModule(etbl, NVTX_CB_MODULE_CUDART, cudart, std::size(cudart));

  // pthread_key destructors don't run for the main thread, and a process
  // that never initializes CUDA never registers the profiler's atexit
  // cleanup, so flush the exiting thread's batch here.
  static std::once_flag atexitOnce;
  std::call_once(atexitOnce, [] { atexit([] { flushCurrentThread(); }); });

  const auto *ver = static_cast<const NvtxExportTableVersionInfo *>(
      getExportTable(NVTX_ETID_VERSIONINFO));
  if (ver && ver->struct_size >= sizeof(NvtxExportTableVersionInfo) &&
      ver->SetInjectionNvtxVersion)
    ver->SetInjectionNvtxVersion(NVTX_VERSION);

  DEBUG_PRINTF("[PARCAGPU] InitializeInjectionNvtx2: handlers installed\n");
  return 1;
}
