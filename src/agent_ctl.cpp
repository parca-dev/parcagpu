// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#define PROCCTL_IMPLEMENTATION
#include "procctl.h"

#include "agent_ctl.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <time.h>
#include <utility>
#include <vector>

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
constexpr uint32_t kMaxGapMs = 60'000;
constexpr long kReleaseTickNs = 100'000'000L;

procctl_t g_ctx;
alignas(8) unsigned char g_stateStorage[PROCCTL_STATE_SIZE(
    sizeof(parcagpu_ctl_state))];
std::atomic<bool> g_ready{false};

// The current configuration, packed so the hot path reads it with one load:
// bit 63 active, bits 0-15 devices, 16-21 factor, 22-35 window ms, 36-51 gap
// ms.
std::atomic<uint64_t> g_config{0};
std::atomic<uint64_t> g_configGen{0};
std::atomic<uint64_t> g_nextLeaseCheckNs{0};
std::atomic<bool> g_leaseExpired{true};

// Serializes applying changes, so they take effect in order.
std::mutex g_applyMutex;
std::atomic<AgentApplyFn> g_apply{nullptr};
uint64_t g_appliedGen = 0; // under g_applyMutex
thread_local bool t_applying = false;

std::atomic<bool> g_threadStop{false};
bool g_threadRunning = false;
pthread_t g_thread;

// Device table, reported as consumer state. g_ordinals[i] is
// state.devices[i]'s CUdevice in this process; g_contexts maps contexts to
// slots.
std::mutex g_devMutex;
parcagpu_ctl_state g_state{};
CUdevice g_ordinals[PARCAGPU_CTL_MAX_DEVICES];
std::vector<std::pair<CUcontext, uint32_t>> g_contexts;

uint64_t pack(const parcagpu_ctl_config &c) {
  return (1ULL << 63) | (uint64_t)c.pc_sampling_devices |
         ((uint64_t)c.pc_sampling_factor << 16) |
         ((uint64_t)c.pc_sampling_window_ms << 22) |
         ((uint64_t)c.pc_sampling_gap_ms << 36);
}

uint64_t coarseNowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void publishStateLocked() {
  procctl_state_set(&g_ctx, &g_state, sizeof g_state);
}

int slotLocked(CUcontext ctx) {
  for (auto &[c, slot] : g_contexts) {
    if (c == ctx)
      return (int)slot;
  }
  return -1;
}

// A forked child can't use its parent's CUDA context, and its copy of the
// state block would report the parent's devices: drop the channel. A child
// that execs gets its own. The release thread doesn't exist in the child.
void atforkChild() {
  g_threadRunning = false;
  if (g_ready.exchange(false)) {
    procctl_close(&g_ctx);
  }
  g_config.store(0, std::memory_order_relaxed);
}

bool valid(const parcagpu_ctl_config &c, uint32_t numDevices) {
  if (c.pc_sampling_factor != 0 &&
      (c.pc_sampling_factor < 5 || c.pc_sampling_factor > 31))
    return false;
  if (c.pc_sampling_window_ms > kMaxWindowMs ||
      c.pc_sampling_gap_ms > kMaxGapMs)
    return false;
  return (c.pc_sampling_devices >> numDevices) == 0;
}

// A change to apply: a new generation (ack it after), or the lease expiring.
struct Update {
  uint64_t packed = 0;
  uint64_t gen = 0;
  bool ack = false;
};

// Decodes a new generation. Rejects an invalid one, which then isn't acked,
// and one that arrives with the lease expired (a dead agent's): procctl
// delivers it again when the lease recovers.
bool decode(int n, const parcagpu_ctl_config &c, uint64_t gen, Update *u) {
  if (procctl_lease_expired(&g_ctx, 0))
    return false;
  if (n != -ENODATA) {
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
      return false;
    }
  }
  u->packed = n == -ENODATA ? 0 : pack(c);
  u->gen = gen;
  u->ack = true;
  return true;
}

// Lease expiry turns the configuration off. On recovery the next poll
// delivers the current configuration again.
bool checkLease(Update *u) {
  uint64_t now = coarseNowNs();
  uint64_t next = g_nextLeaseCheckNs.load(std::memory_order_relaxed);
  if (now < next || !g_nextLeaseCheckNs.compare_exchange_strong(
                        next, now + kLeaseCheckNs, std::memory_order_relaxed))
    return false;
  bool expired = procctl_lease_expired(&g_ctx, 0);
  if (g_leaseExpired.exchange(expired) != expired)
    DEBUG_PRINTF("[PARCAGPU] agent control: lease %s\n",
                 expired ? "expired" : "recovered");
  if (!expired || g_config.load(std::memory_order_relaxed) == 0)
    return false;
  *u = Update{};
  return true;
}

// Reports pc_state for the configuration just applied: ON where it grants
// sampling, OFF elsewhere (sampling there has been stopped).
void reportLocked(uint64_t packed) {
  bool active = (packed >> 63) != 0;
  for (uint32_t i = 0; i < g_state.num_devices; i++) {
    g_state.devices[i].pc_state =
        active && ((packed >> i) & 1) ? PARCAGPU_PC_ON : PARCAGPU_PC_OFF;
  }
  publishStateLocked();
}

void apply(const Update &u) {
  std::lock_guard<std::mutex> lock(g_applyMutex);
  if (u.ack && u.gen < g_appliedGen)
    return; // another thread already applied a newer generation
  t_applying = true;
  g_apply.load(std::memory_order_acquire)([&] {
    g_config.store(u.packed, std::memory_order_release);
    g_configGen.fetch_add(1, std::memory_order_release);
  });
  t_applying = false;
  {
    std::lock_guard<std::mutex> lock2(g_devMutex);
    reportLocked(u.packed);
  }
  if (u.ack) {
    g_appliedGen = u.gen;
    procctl_ack(&g_ctx, u.gen);
  }
  DEBUG_PRINTF("[PARCAGPU] agent control: applied %s: devices 0x%x factor "
               "%u window %u ms\n",
               u.ack ? "configuration" : "lease expiry",
               (uint32_t)(u.packed & 0xffff),
               (uint32_t)((u.packed >> 16) & 0x3f),
               (uint32_t)((u.packed >> 22) & 0x3fff));
}

void *releaseThread(void *) {
  while (!g_threadStop.load(std::memory_order_acquire)) {
    struct timespec ts = {0, kReleaseTickNs};
    nanosleep(&ts, nullptr);
    try {
      agentCtlPoll();
    } catch (const std::exception &e) {
      DEBUG_PRINTF("[PARCAGPU] agent control thread caught: %s\n", e.what());
    } catch (...) {
    }
  }
  return nullptr;
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

bool agentCtlReady() { return g_ready.load(std::memory_order_acquire); }

void agentCtlStart(AgentApplyFn apply) {
  if (!agentCtlReady())
    return;
  g_apply.store(apply, std::memory_order_release);
  // Block signals in the thread: they belong to the application's threads.
  sigset_t all, old;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &old);
  g_threadRunning =
      pthread_create(&g_thread, nullptr, releaseThread, nullptr) == 0;
  pthread_sigmask(SIG_SETMASK, &old, nullptr);
  if (!g_threadRunning)
    fireError(0, "agent control: release thread not started", "agent_ctl");
}

void agentCtlShutdown() {
  if (!g_threadRunning)
    return;
  g_threadStop.store(true, std::memory_order_release);
  pthread_join(g_thread, nullptr);
  g_threadRunning = false;
}

void agentCtlPoll() {
  if (!g_ready.load(std::memory_order_acquire) ||
      g_apply.load(std::memory_order_acquire) == nullptr || t_applying)
    return;
  parcagpu_ctl_config c{};
  uint64_t gen = 0;
  Update u;
  int n = procctl_poll(&g_ctx, &c, sizeof c, &gen);
  if ((n > 0 || n == -ENODATA) && decode(n, c, gen, &u))
    apply(u);
  else if (n == -E2BIG)
    fireError(E2BIG, "agent configuration too large", "agent_ctl");
  if (checkLease(&u))
    apply(u);
}

void agentCtlAddContext(CUcontext ctx) {
  if (!agentCtlReady())
    return;
  // The context isn't necessarily current in CONTEXT_CREATED.
  CUdevice dev = 0;
  if (!agentCtlPushContext(ctx))
    return;
  CUresult r = proton::cuda::ctxGetDevice<false>(&dev);
  agentCtlPopContext();
  if (r != CUDA_SUCCESS)
    return;

  std::lock_guard<std::mutex> lock(g_devMutex);
  uint32_t n = g_state.num_devices, slot = n;
  for (uint32_t i = 0; i < n; i++) {
    if (g_ordinals[i] == dev)
      slot = i;
  }
  if (slot == n) {
    if (n == PARCAGPU_CTL_MAX_DEVICES) {
      fireError((int32_t)dev, "too many devices for agent control",
                "agent_ctl");
      return;
    }
    auto &d = g_state.devices[n];
    CUuuid uuid{};
    int clockKHz = 0, smCount = 0;
    proton::cuda::deviceGetUuid<false>(&uuid, dev);
    proton::cuda::deviceGetAttribute<false>(
        &clockKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
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
    DEBUG_PRINTF("[PARCAGPU] agent control: device %d is slot %u\n", (int)dev,
                 n);
  }
  // A destroyed context's address can be reused: the latest mapping wins.
  for (auto &e : g_contexts) {
    if (e.first == ctx) {
      e.second = slot;
      return;
    }
  }
  g_contexts.emplace_back(ctx, slot);
}

AgentConfig agentCtlConfig() {
  uint64_t v = g_config.load(std::memory_order_acquire);
  AgentConfig c;
  c.active = (v >> 63) != 0;
  c.devices = (uint32_t)(v & 0xffff);
  c.factor = (uint32_t)((v >> 16) & 0x3f);
  c.windowMs = (uint32_t)((v >> 22) & 0x3fff);
  c.gapMs = (uint32_t)((v >> 36) & 0xffff);
  return c;
}

uint64_t agentCtlGeneration() {
  return g_configGen.load(std::memory_order_acquire);
}

bool agentCtlAllows(CUcontext ctx, bool envDefault) {
  uint64_t v = g_config.load(std::memory_order_acquire);
  if ((v >> 63) == 0)
    return envDefault;
  std::lock_guard<std::mutex> lock(g_devMutex);
  int slot = slotLocked(ctx);
  return slot >= 0 && ((v >> slot) & 1);
}

void agentCtlSetCapable(bool capable) {
  if (!agentCtlReady())
    return;
  std::lock_guard<std::mutex> lock(g_devMutex);
  g_state.pc_capable = capable;
  publishStateLocked();
}

void agentCtlSetPCState(CUcontext ctx, uint32_t state) {
  if (!agentCtlReady())
    return;
  std::lock_guard<std::mutex> lock(g_devMutex);
  int slot = slotLocked(ctx);
  if (slot < 0 || g_state.devices[slot].pc_state == state)
    return;
  g_state.devices[slot].pc_state = state;
  publishStateLocked();
}

void agentCtlSetFactor(CUcontext ctx, uint32_t factor) {
  if (!agentCtlReady())
    return;
  std::lock_guard<std::mutex> lock(g_devMutex);
  int slot = slotLocked(ctx);
  if (slot < 0 || g_state.devices[slot].sampling_factor == factor)
    return;
  g_state.devices[slot].sampling_factor = factor;
  publishStateLocked();
}

bool agentCtlPushContext(CUcontext ctx) {
  return proton::cuda::ctxPushCurrent<false>(ctx) == CUDA_SUCCESS;
}

void agentCtlPopContext() {
  CUcontext popped = nullptr;
  proton::cuda::ctxPopCurrent<false>(&popped);
}

} // namespace parcagpu
