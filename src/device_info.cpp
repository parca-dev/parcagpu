// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include "device_info.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <time.h>
#include <vector>

#include "Driver/Dispatch.h"
#include "log.h"
#include "probes.h"

namespace {
constexpr const char *kLogComponent = "device_info";
} // namespace

// Proton's CUDA dispatch table lacks these driver calls, and its ExternLibCuda
// is private to CudaApi.cpp; declare an equivalent lazily-dlopen'd handle here
// rather than patching the submodule.
namespace proton::cuda {
struct ExternLibCudaDeviceInfo : public ExternLibBase {
  using RetType = CUresult;
  static constexpr const char *name = "libcuda.so.1";
  static constexpr RetType success = CUDA_SUCCESS;
  static inline void *lib = nullptr;
};
template <bool C> CUresult ctxPushCurrent(CUcontext);
template <bool C> CUresult ctxPopCurrent(CUcontext *);
template <bool C> CUresult ctxGetDeviceDI(CUdevice *);
template <bool C> CUresult deviceGetUuid(CUuuid *, CUdevice);
template <bool C> CUresult deviceGetPCIBusId(char *, int, CUdevice);
DEFINE_DISPATCH(ExternLibCudaDeviceInfo, ctxPushCurrent, cuCtxPushCurrent_v2,
                CUcontext)
DEFINE_DISPATCH(ExternLibCudaDeviceInfo, ctxPopCurrent, cuCtxPopCurrent_v2,
                CUcontext *)
DEFINE_DISPATCH(ExternLibCudaDeviceInfo, ctxGetDeviceDI, cuCtxGetDevice,
                CUdevice *)
DEFINE_DISPATCH(ExternLibCudaDeviceInfo, deviceGetUuid, cuDeviceGetUuid_v2,
                CUuuid *, CUdevice)
DEFINE_DISPATCH(ExternLibCudaDeviceInfo, deviceGetPCIBusId,
                cuDeviceGetPCIBusId, char *, int, CUdevice)
} // namespace proton::cuda

// Single call site for the probe (see the fireX wrappers in pc_sampling.cpp).
__attribute__((noinline)) void fireDeviceInfo(uint32_t dev,
                                              const uint8_t *uuid,
                                              const char *pciBusId) {
  PARCAGPU_DEVICE_INFO(dev, uuid, pciBusId);
}

namespace parcagpu::device_info {
namespace {

constexpr uint64_t kRefreshPeriodNs = 30ULL * 1000 * 1000 * 1000;

struct Device {
  uint32_t ordinal;
  uint8_t uuid[16];
  char pciBusId[32];
  uint64_t lastEmittedNs;
};

std::mutex g_mutex;
std::vector<Device> g_devices; // guarded by g_mutex
std::atomic<uint16_t> g_prevSem{0};
std::atomic<uint64_t> g_lastRefreshNs{0};

uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

// The USDT semaphore counts attached uprobe consumers; the kernel updates it
// from outside this address space.
uint16_t readSem() {
  return __atomic_load_n(
      reinterpret_cast<uint16_t *>(&parcagpu_device_info_semaphore),
      __ATOMIC_RELAXED);
}

} // namespace

void onContextCreated(CUcontext ctx) {
  // Best effort: a missing symbol throws from Dispatch, and a failed lookup
  // only costs the UUID join, so never let either break the application.
  try {
    // The new context isn't necessarily current in the resource callback.
    CUdevice dev = 0;
    if (proton::cuda::ctxPushCurrent<false>(ctx) != CUDA_SUCCESS)
      return;
    const CUresult res = proton::cuda::ctxGetDeviceDI<false>(&dev);
    CUcontext popped = nullptr;
    proton::cuda::ctxPopCurrent<false>(&popped);
    if (res != CUDA_SUCCESS)
      return;

    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto &d : g_devices) {
      if (d.ordinal == (uint32_t)dev)
        return; // another context on a known device
    }
    Device d{};
    d.ordinal = (uint32_t)dev;
    CUuuid uuid{};
    if (proton::cuda::deviceGetUuid<false>(&uuid, dev) != CUDA_SUCCESS)
      return;
    static_assert(sizeof(uuid.bytes) == sizeof(d.uuid));
    std::memcpy(d.uuid, uuid.bytes, sizeof(d.uuid));
    if (proton::cuda::deviceGetPCIBusId<false>(d.pciBusId, sizeof(d.pciBusId),
                                               dev) != CUDA_SUCCESS)
      d.pciBusId[0] = '\0';
    d.lastEmittedNs = nowNs();
    PLOG_INFO(kLogComponent, "device %u: PCI %s", d.ordinal, d.pciBusId);
    fireDeviceInfo(d.ordinal, d.uuid, d.pciBusId);
    g_devices.push_back(d);
  } catch (...) {
  }
}

void emit() {
  // Re-fire on a consumer join (count increase, so 1→2 also fires), plus a
  // periodic refresh bounding staleness from detach/attach cycles that net
  // the same count between reads. Same pattern as PCSampling::emitMetadata.
  const uint16_t sem = readSem();
  const uint16_t prev = g_prevSem.exchange(sem, std::memory_order_acq_rel);
  if (sem == 0)
    return;
  const bool join = sem > prev;
  const uint64_t now = nowNs();
  const bool refresh =
      now - g_lastRefreshNs.load(std::memory_order_relaxed) > kRefreshPeriodNs;
  if (!join && !refresh)
    return;
  if (refresh)
    g_lastRefreshNs.store(now, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(g_mutex);
  for (auto &d : g_devices) {
    if (join || now - d.lastEmittedNs > kRefreshPeriodNs) {
      fireDeviceInfo(d.ordinal, d.uuid, d.pciBusId);
      d.lastEmittedNs = now;
    }
  }
}

} // namespace parcagpu::device_info
