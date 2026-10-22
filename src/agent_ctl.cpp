// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#define PROCCTL_IMPLEMENTATION
#include "procctl.h"

#include "agent_ctl.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <time.h>

#include "Driver/Dispatch.h"
#include "Driver/GPU/CudaApi.h"
#include "parcagpu_ctl.h"
#include "pc_sampling.h"

// Driver calls Proton's dispatch table lacks, declared the same way (lazily
// dlsym'd from the already-loaded libcuda) rather than patching the submodule.
namespace proton::cuda {
struct ExternLibCudaCtl : public ExternLibBase {
  using RetType = CUresult;
  static constexpr const char *name = "libcuda.so.1";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = CUDA_SUCCESS;
  static inline void *lib = nullptr;
};
template <bool CheckSuccess> CUresult deviceGetUuid(CUuuid *uuid, CUdevice dev);
DEFINE_DISPATCH(ExternLibCudaCtl, deviceGetUuid, cuDeviceGetUuid_v2, CUuuid *,
                CUdevice)
template <bool CheckSuccess> CUresult ctxPushCurrent(CUcontext ctx);
DEFINE_DISPATCH(ExternLibCudaCtl, ctxPushCurrent, cuCtxPushCurrent_v2,
                CUcontext)
template <bool CheckSuccess> CUresult ctxPopCurrent(CUcontext *ctx);
DEFINE_DISPATCH(ExternLibCudaCtl, ctxPopCurrent, cuCtxPopCurrent_v2,
                CUcontext *)
} // namespace proton::cuda

namespace parcagpu {

namespace {

constexpr uint64_t kLeaseCheckNs = 250'000'000ULL;
constexpr uint32_t kMaxWindowMs = 10'000;

procctl_t g_ctx;
alignas(8) unsigned char g_stateStorage[PROCCTL_STATE_SIZE(
    sizeof(parcagpu_ctl_state))];
std::atomic<bool> g_ready{false};

// The active configuration, packed so the hot path reads it with one load:
// bit 63 active, bits 0-15 devices, 16-21 factor, 22-35 window ms.
std::atomic<uint64_t> g_config{0};
std::atomic<uint64_t> g_nextLeaseCheckNs{0};
std::atomic<bool> g_leaseExpired{true};

// Device table, reported as consumer state. ordinals[i] is state.devices[i]'s
// CUdevice in this process.
std::mutex g_devMutex;
parcagpu_ctl_state g_state{};
CUdevice g_ordinals[PARCAGPU_CTL_MAX_DEVICES];

uint64_t pack(const parcagpu_ctl_config &c) {
  return (1ULL << 63) | (uint64_t)c.pc_sampling_devices |
         ((uint64_t)c.pc_sampling_factor << 16) |
         ((uint64_t)c.pc_sampling_window_ms << 22);
}

uint64_t coarseNowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void publishStateLocked() {
  procctl_state_set(&g_ctx, &g_state, sizeof g_state);
}

// A forked child can't use its parent's CUDA context, and its copy of the
// state block would report the parent's devices: drop the channel. A child
// that execs gets its own.
void atforkChild() {
  if (g_ready.exchange(false)) {
    procctl_close(&g_ctx);
  }
  g_config.store(0, std::memory_order_relaxed);
}

bool valid(const parcagpu_ctl_config &c, uint32_t numDevices) {
  if (c.pc_sampling_factor != 0 &&
      (c.pc_sampling_factor < 5 || c.pc_sampling_factor > 31))
    return false;
  if (c.pc_sampling_window_ms > kMaxWindowMs)
    return false;
  return (c.pc_sampling_devices >> numDevices) == 0;
}

// Applies a new generation and acks it. An invalid one is not acked. Nor is
// one that arrives with the lease expired (a dead agent's): procctl delivers
// it again when the lease recovers.
void apply(int n, const parcagpu_ctl_config &c, uint64_t gen) {
  if (procctl_lease_expired(&g_ctx, 0))
    return;
  if (n == -ENODATA) {
    g_config.store(0, std::memory_order_release);
    DEBUG_PRINTF("[PARCAGPU] agent control: no configuration (gen %lu)\n", gen);
    procctl_ack(&g_ctx, gen);
    return;
  }
  uint32_t numDevices;
  {
    std::lock_guard<std::mutex> lock(g_devMutex);
    numDevices = g_state.num_devices;
  }
  if (n != (int)sizeof c || !valid(c, numDevices)) {
    DEBUG_PRINTF("[PARCAGPU] agent control: rejected gen %lu (len %d, "
                 "devices 0x%x, factor %u, window %u ms)\n",
                 gen, n, c.pc_sampling_devices, c.pc_sampling_factor,
                 c.pc_sampling_window_ms);
    fireError(EINVAL, "invalid agent configuration", "agent_ctl");
    return;
  }
  g_config.store(pack(c), std::memory_order_release);
  DEBUG_PRINTF("[PARCAGPU] agent control: gen %lu devices 0x%x factor %u "
               "window %u ms\n",
               gen, c.pc_sampling_devices, c.pc_sampling_factor,
               c.pc_sampling_window_ms);
  procctl_ack(&g_ctx, gen);
}

void checkLease() {
  uint64_t now = coarseNowNs();
  uint64_t next = g_nextLeaseCheckNs.load(std::memory_order_relaxed);
  if (now < next || !g_nextLeaseCheckNs.compare_exchange_strong(
                        next, now + kLeaseCheckNs, std::memory_order_relaxed))
    return;
  // Expired: revert to the environment. On recovery the next poll delivers
  // the current configuration again.
  bool expired = procctl_lease_expired(&g_ctx, 0);
  if (expired)
    g_config.store(0, std::memory_order_release);
  if (g_leaseExpired.exchange(expired) != expired)
    DEBUG_PRINTF("[PARCAGPU] agent control: lease %s\n",
                 expired ? "expired" : "recovered");
}

} // namespace

void agentCtlInit() {
  procctl_config_t cfg = {};
  cfg.consumer = PARCAGPU_CTL_CONSUMER;
  cfg.schema_version = PARCAGPU_CTL_SCHEMA_VERSION;
  cfg.payload_capacity = sizeof(parcagpu_ctl_config);
  cfg.state_capacity = sizeof(parcagpu_ctl_state);
  cfg.state_storage = g_stateStorage;
  int r = procctl_init(&g_ctx, &cfg);
  if (r != 0) {
    DEBUG_PRINTF("[PARCAGPU] agent control unavailable: procctl_init %d\n", r);
    fireError(r, "agent control unavailable (procctl_init failed)",
              "agent_ctl");
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_devMutex);
    publishStateLocked();
  }
  pthread_atfork(nullptr, nullptr, atforkChild);
  g_ready.store(true, std::memory_order_release);
  DEBUG_PRINTF("[PARCAGPU] agent control ready\n");
}

void agentCtlPoll() {
  if (!g_ready.load(std::memory_order_acquire))
    return;
  parcagpu_ctl_config c{};
  uint64_t gen = 0;
  int n = procctl_poll(&g_ctx, &c, sizeof c, &gen);
  if (n > 0 || n == -ENODATA)
    apply(n, c, gen);
  else if (n == -E2BIG)
    fireError(E2BIG, "agent configuration too large", "agent_ctl");
  checkLease();
}

void agentCtlAddContext(CUcontext ctx) {
  if (!g_ready.load(std::memory_order_acquire))
    return;
  // The context isn't necessarily current in CONTEXT_CREATED.
  CUdevice dev = 0;
  if (proton::cuda::ctxPushCurrent<false>(ctx) != CUDA_SUCCESS)
    return;
  CUresult r = proton::cuda::ctxGetDevice<false>(&dev);
  CUcontext popped = nullptr;
  proton::cuda::ctxPopCurrent<false>(&popped);
  if (r != CUDA_SUCCESS)
    return;

  std::lock_guard<std::mutex> lock(g_devMutex);
  uint32_t n = g_state.num_devices;
  for (uint32_t i = 0; i < n; i++) {
    if (g_ordinals[i] == dev)
      return;
  }
  if (n == PARCAGPU_CTL_MAX_DEVICES) {
    fireError((int32_t)dev, "too many devices for agent control", "agent_ctl");
    return;
  }
  auto &d = g_state.devices[n];
  CUuuid uuid{};
  int clockKHz = 0, smCount = 0;
  proton::cuda::deviceGetUuid<false>(&uuid, dev);
  proton::cuda::deviceGetAttribute<false>(&clockKHz,
                                          CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
  proton::cuda::deviceGetAttribute<false>(
      &smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  static_assert(sizeof d.uuid == sizeof uuid.bytes, "uuid size");
  memcpy(d.uuid, uuid.bytes, sizeof d.uuid);
  d.clock_khz = (uint32_t)clockKHz;
  d.sm_count = (uint32_t)smCount;
  d.pc_state = PARCAGPU_PC_OFF;
  g_ordinals[n] = dev;
  g_state.num_devices = n + 1;
  publishStateLocked();
  DEBUG_PRINTF("[PARCAGPU] agent control: device %d is slot %u\n", (int)dev, n);
}

AgentConfig agentCtlConfig() {
  uint64_t v = g_config.load(std::memory_order_acquire);
  AgentConfig c;
  c.active = (v >> 63) != 0;
  c.devices = (uint32_t)(v & 0xffff);
  c.factor = (uint32_t)((v >> 16) & 0x3f);
  c.windowMs = (uint32_t)((v >> 22) & 0x3fff);
  return c;
}

} // namespace parcagpu
