// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <unistd.h>

// USDT probes — must come before any header that might include <sys/sdt.h>,
// so that _SDT_HAS_SEMAPHORES is defined first.
#include "probes.h"

// Include proton headers
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiCallbacks.h"
#include "Utility/Singleton.h"
#include "agent_ctl.h"
#include "parcagpu_ctl.h"
#include "pc_gate.h"
#include "correlation_filter.h"
#include "device_info.h"
#include "env_config.h"
#include "kernel_stats.h"
#include "machine_gpus.h"
#include "nvtx_collector.h"
#include "pc_sampling.h"
#include "token_bucket.h"

// Proton's CUPTI dispatch table lacks cuptiActivityFlushPeriod.
#include "Driver/Dispatch.h"
namespace proton::cupti {
template <bool C> CUptiResult activityFlushPeriod(uint32_t);
DEFINE_DISPATCH(ExternLibCupti, activityFlushPeriod, cuptiActivityFlushPeriod,
                uint32_t)
} // namespace proton::cupti

namespace parcagpu {

// Debug logging control
bool debug_enabled = false;

// Global correlation tracking instances
static CorrelationFilter g_correlationFilter;
static GraphCorrelationMap g_graphCorrelationMap;
static std::atomic<uint32_t> g_bufferCycle{0};

// Thread-local tracking: store correlation ID from runtime ENTER
// so we can skip driver EXIT probe when it matches (driver calls happen under
// runtime calls)
thread_local uint32_t runtimeEnterCorrelationId = 0;

// A resource MODULE_LOADED callback can run inside cuModuleLoad. CUPTI's
// resource record has no filename, so carry that optional hint across the
// synchronous call. Canonicalize in the application's cwd/mount namespace.
// Data/DataEx, JIT linking and deferred loads deliberately have no path hint.
thread_local std::vector<std::string> moduleLoadPaths;

uint32_t currentRuntimeCorrelationId() { return runtimeEnterCorrelationId; }

// Configured probe rates (tokens/sec), read from env once in init_debug; the
// agent's, when it sets them, take precedence. Globals because the buckets
// below are thread_local: setRate from init_debug would only reach the single
// thread that ran it. Without PARCAGPU_RATE_LIMIT / PARCAGPU_GRAPH_RATE_LIMIT,
// each default is split across the machine's GPUs (splitRateLimit).
static constexpr double kDefaultLaunchRateLimit = 100.0;
static constexpr double kDefaultGraphRateLimit = 20.0;
std::atomic<double> g_eagerRateLimit{kDefaultLaunchRateLimit};
std::atomic<double> g_graphRateLimit{kDefaultGraphRateLimit};

// Thread-local rate limiters. Graph launches get a separate, lower bucket
// (PARCAGPU_GRAPH_RATE_LIMIT) so graph replays don't flood cupti_events and
// eager bursts can't starve graph sampling.
thread_local TokenBucket callbackLimiter(100.0);
thread_local TokenBucket graphCallbackLimiter(20.0);

// Copy the configured rates into this thread's buckets: once per thread, and
// again whenever the agent's configuration changes.
static inline void applyRateLimits() {
  thread_local uint64_t appliedGen = UINT64_MAX;
  const uint64_t gen = parcagpu::agentCtlGeneration();
  if (appliedGen == gen)
    return;
  appliedGen = gen;
  const parcagpu::AgentConfig cfg = parcagpu::agentCtlConfig();
  callbackLimiter.setRate(
      cfg.rateLimit ? cfg.rateLimit
                    : g_eagerRateLimit.load(std::memory_order_relaxed));
  graphCallbackLimiter.setRate(
      cfg.graphRateLimit ? cfg.graphRateLimit
                         : g_graphRateLimit.load(std::memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// PC sampling probabilistic control.
//
// Sampling is gated by a per-thread interval + dice-roll mechanism: at most
// once per kPCSamplingIntervalNs, roll against probability; if it hits, open
// a sampling window that stays active until the next interval boundary.
//
// The user-facing knob is PARCAGPU_PC_SAMPLING_RATE (samples/sec); a
// process-wide controller adjusts the dice-roll probability over time so the
// observed sample rate converges on the target. Internally the controller
// reads `samplesTotal` (incremented from pc_sampling.cpp on every batch) and
// recalibrates probability every kPCControlPeriodNs based on the rate
// observed since the last update.
// ---------------------------------------------------------------------------

// 30 ms — short enough to give ~33 dice rolls/sec (tighter rate variance,
// faster response to workload phase changes), long enough that CUPTI
// start/stop cost (~25 us measured) is amortized to <0.1% of wall time.
static constexpr uint64_t kPCSamplingIntervalNs = 30'000'000ULL;
// How often the controller recalibrates probability based on observed rate.
static constexpr uint64_t kPCControlPeriodNs = 5'000'000'000ULL;
// Don't react to <25% rate error (avoids oscillating on noise).
static constexpr double kPCControlTolerance = 0.25;
// Symmetric step clamp at sqrt(2). Larger steps (2x or 4x) amplify
// single-window sampling noise into multi-update oscillations: one
// 30ms window happening to land in an idle phase shows few samples,
// the controller over-corrects, then the next busy phase shows many.
// sqrt(2) keeps each adjustment small enough that the eventual
// equilibrium sits within the tolerance band even under bursty signal.
static constexpr double kPCControlStepShrink = 1.41421356;
static constexpr double kPCControlStepGrow = 1.41421356;
static constexpr double kPCProbMin = 0.001;
static constexpr double kPCProbMax = 1.0;
// Initial probability *at the default target rate*. Higher values waste
// samples on kernel-dense workloads (FNS-class: 5K launches/sec); lower
// values starve kernel-sparse workloads (a few launches/sec) of dice
// rolls until the controller grows it. 0.02 is a tested compromise at
// targetRate=100. Real initial probability scales with targetRate (see
// init_debug) so callers asking for high rates start sampling
// immediately instead of waiting many control periods to climb.
static constexpr double kPCInitialProbabilityAtDefaultRate = 0.02;
// Default target rate when PARCAGPU_PC_SAMPLING_RATE is unset.
static constexpr double kPCDefaultTargetRate = 100.0;

struct PCRateController {
  double targetRate = kPCDefaultTargetRate;       // immutable after init
  std::atomic<double> probability{kPCInitialProbabilityAtDefaultRate};
  std::atomic<uint64_t> samplesTotal{0};
  std::atomic<uint64_t> lastCheckNs{0};
  std::atomic<uint64_t> lastCheckTotal{0};
};
static PCRateController g_pcController;

// Stream captures and PC sampling state changes keep out of each other's
// way. Opening a window synchronizes the context and starts PC sampling;
// closing one stops it; setting a context up or letting it go enables or
// disables PC sampling on it. A capture forbids all of these, on the
// capturing thread or, during a global-mode capture, on any thread; the call
// fails and invalidates the capture. So a capture counts from its
// BeginCapture ENTER (g_streamCaptures), and a state change from before it
// starts (g_pcTransitions, PCTransition): a change gives up if a capture is
// counted (checked again with the gate closed, PCSampling::captureFree), and
// a BeginCapture waits for changes in progress, then closes the open windows
// (beforeCapture). Both counters are seq_cst: of a change and a capture
// that race, at least one sees the other.
static std::atomic<int> g_streamCaptures{0};
static std::atomic<int> g_pcTransitions{0};

// How long a BeginCapture waits for a state change (a window opening's
// synchronize waits for the GPU); then it goes ahead.
static constexpr uint64_t kCaptureWaitNs = 1'000'000'000ULL;

class PCTransition {
public:
  PCTransition() { g_pcTransitions.fetch_add(1); }
  ~PCTransition() { g_pcTransitions.fetch_sub(1); }
  PCTransition(const PCTransition &) = delete;
  PCTransition &operator=(const PCTransition &) = delete;
  static bool captureFree() { return g_streamCaptures.load() == 0; }
};

static bool isBeginCapture(CUpti_CallbackId cbid) {
  switch (cbid) {
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz:
    return true;
  default:
    return false;
  }
}

static void trackStreamCapture(CUpti_CallbackId cbid,
                               const CUpti_CallbackData *cbdata) {
  switch (cbid) {
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz:
    // Counted at ENTER (beforeCapture); uncount one that didn't begin.
    if (*static_cast<const CUresult *>(cbdata->functionReturnValue) !=
        CUDA_SUCCESS)
      g_streamCaptures.fetch_sub(1);
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture:
  case CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz: {
    // Ends the capture even when it fails (an invalidated capture).
    int n = g_streamCaptures.load();
    while (n > 0 && !g_streamCaptures.compare_exchange_weak(n, n - 1))
      ;
    break;
  }
  default:
    break;
  }
}

// Per-thread state for opening windows. A window itself belongs to its
// context (PCSampling): any thread launching there drains and closes it, so
// it lasts its length even if the thread that opened it goes quiet.
struct PCSamplingState {
  uint64_t lastCheckNs = 0; // Last time this thread tried to open a window
  unsigned int rngSeed = 0; // Thread-local RNG state
};
thread_local PCSamplingState g_pcSamplingState;

static uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Seed the per-thread RNG lazily.
static void ensureRngSeeded(PCSamplingState &s) {
  if (s.rngSeed == 0) {
    uint64_t t = nowNs();
    s.rngSeed = (unsigned int)(t ^ (uintptr_t)&s);
    if (s.rngSeed == 0)
      s.rngSeed = 1;
  }
}

static double threadRandom(PCSamplingState &s) {
  ensureRngSeeded(s);
  return (double)rand_r(&s.rngSeed) / RAND_MAX;
}

// ---------------------------------------------------------------------------
// Host blocking-call sampling (cuda_blocking_call probe).
//
// CUDA API calls that can block the calling host thread are timed from ENTER
// to EXIT on that thread: GPU waits (synchronize, sync memcpy, free, query),
// allocation and page pinning, VMM mapping, module/library loading and graph
// instantiation. The probe fires inside the API call, so the agent sees the
// application's stack; the signed cbid tells it which call (and category).
// Calls lasting at least the threshold T are always reported with
// weight = duration; shorter calls are reported with probability duration/T
// and weight = T, so summed weights are an unbiased estimate of total blocked
// time and the emitted event rate is bounded by blocked time / T.
// ---------------------------------------------------------------------------

// Default T: 10 us. Override with PARCAGPU_BLOCKING_THRESHOLD_US (0 = all).
static constexpr uint64_t kBlockingDefaultThresholdNs = 10'000;
std::atomic<uint64_t> g_blockingThresholdNs{kBlockingDefaultThresholdNs};

// Blocking-class callbacks: the single source for both the enable list and
// the isBlockingCb() switch. Proton's setRuntimeCallbacks/setLaunchCallbacks
// only cover launches, so these are enabled explicitly in init(). Names are
// checked against the CUDA 12.9 cupti_{runtime,driver}_cbid.h.
// Async enqueues (cudaMemcpyAsync, cudaMallocAsync, launches) are deliberately
// excluded: they are high-frequency and their cost is GPU-side.
#define PARCAGPU_BLOCKING_RUNTIME_CBIDS(X)                                     \
  /* GPU waits: device / stream / event synchronize and query */               \
  X(cudaDeviceSynchronize_v3020)                                               \
  X(cudaThreadSynchronize_v3020)                                               \
  X(cudaStreamSynchronize_v3020)                                               \
  X(cudaStreamSynchronize_ptsz_v7000)                                          \
  X(cudaStreamQuery_v3020)                                                     \
  X(cudaStreamQuery_ptsz_v7000)                                                \
  X(cudaEventSynchronize_v3020)                                                \
  X(cudaEventQuery_v3020)                                                      \
  /* Synchronous memcpy */                                                     \
  X(cudaMemcpy_v3020)                                                          \
  X(cudaMemcpy_ptds_v7000)                                                     \
  X(cudaMemcpy2D_v3020)                                                        \
  X(cudaMemcpy2D_ptds_v7000)                                                   \
  X(cudaMemcpy3D_v3020)                                                        \
  X(cudaMemcpy3D_ptds_v7000)                                                   \
  X(cudaMemcpyToSymbol_v3020)                                                  \
  X(cudaMemcpyToSymbol_ptds_v7000)                                             \
  X(cudaMemcpyFromSymbol_v3020)                                                \
  X(cudaMemcpyFromSymbol_ptds_v7000)                                           \
  X(cudaMemcpyPeer_v4000)                                                      \
  /* Device allocation / free (free implicitly synchronizes the device) */     \
  X(cudaMalloc_v3020)                                                          \
  X(cudaMallocPitch_v3020)                                                     \
  X(cudaMalloc3D_v3020)                                                        \
  X(cudaMallocManaged_v6000)                                                   \
  X(cudaFree_v3020)                                                            \
  X(cudaMemPoolTrimTo_v11020)                                                  \
  /* Pinned host memory */                                                     \
  X(cudaMallocHost_v3020)                                                      \
  X(cudaHostAlloc_v3020)                                                       \
  X(cudaFreeHost_v3020)                                                        \
  X(cudaHostRegister_v4000)                                                    \
  X(cudaHostUnregister_v4000)                                                  \
  /* Graph instantiation / update / upload */                                  \
  X(cudaGraphInstantiate_v10000)                                               \
  X(cudaGraphInstantiate_v12000)                                               \
  X(cudaGraphInstantiateWithFlags_v11040)                                      \
  X(cudaGraphInstantiateWithParams_v12000)                                     \
  X(cudaGraphInstantiateWithParams_ptsz_v12000)                                \
  X(cudaGraphExecUpdate_v10020)                                                \
  X(cudaGraphUpload_v10000)                                                    \
  X(cudaGraphUpload_ptsz_v10000)

// Legacy pre-_v2 driver cbids are kept for old binaries; current headers map
// cuMemcpyXtoY / cuMemAlloc etc. to the _v2 entry points.
#define PARCAGPU_BLOCKING_DRIVER_CBIDS(X)                                      \
  /* GPU waits */                                                              \
  X(cuCtxSynchronize)                                                          \
  X(cuStreamSynchronize)                                                       \
  X(cuStreamSynchronize_ptsz)                                                  \
  X(cuStreamQuery)                                                             \
  X(cuStreamQuery_ptsz)                                                        \
  X(cuEventSynchronize)                                                        \
  X(cuEventQuery)                                                              \
  /* Synchronous memcpy */                                                     \
  X(cuMemcpy)                                                                  \
  X(cuMemcpy_ptds)                                                             \
  X(cuMemcpyHtoD)                                                              \
  X(cuMemcpyHtoD_v2)                                                           \
  X(cuMemcpyHtoD_v2_ptds)                                                      \
  X(cuMemcpyDtoH)                                                              \
  X(cuMemcpyDtoH_v2)                                                           \
  X(cuMemcpyDtoH_v2_ptds)                                                      \
  X(cuMemcpyDtoD)                                                              \
  X(cuMemcpyDtoD_v2)                                                           \
  X(cuMemcpyDtoD_v2_ptds)                                                      \
  X(cuMemcpy2D)                                                                \
  X(cuMemcpy2D_v2)                                                             \
  X(cuMemcpy2D_v2_ptds)                                                        \
  X(cuMemcpy3D)                                                                \
  X(cuMemcpy3D_v2)                                                             \
  X(cuMemcpy3D_v2_ptds)                                                        \
  X(cuMemcpyPeer)                                                              \
  X(cuMemcpyPeer_ptds)                                                         \
  /* Device allocation / free */                                               \
  X(cuMemAlloc)                                                                \
  X(cuMemAlloc_v2)                                                             \
  X(cuMemAllocPitch)                                                           \
  X(cuMemAllocPitch_v2)                                                        \
  X(cuMemAllocManaged)                                                         \
  X(cuMemFree)                                                                 \
  X(cuMemFree_v2)                                                              \
  X(cuMemPoolTrimTo)                                                           \
  /* Pinned host memory */                                                     \
  X(cuMemAllocHost)                                                            \
  X(cuMemAllocHost_v2)                                                         \
  X(cuMemHostAlloc)                                                            \
  X(cuMemHostAlloc_v2)                                                         \
  X(cuMemFreeHost)                                                             \
  X(cuMemHostRegister)                                                         \
  X(cuMemHostRegister_v2)                                                      \
  X(cuMemHostUnregister)                                                       \
  /* Virtual memory management (e.g. PyTorch expandable segments) */           \
  X(cuMemCreate)                                                               \
  X(cuMemRelease)                                                              \
  X(cuMemMap)                                                                  \
  X(cuMemUnmap)                                                                \
  X(cuMemSetAccess)                                                            \
  X(cuMemAddressReserve)                                                       \
  X(cuMemAddressFree)                                                          \
  /* Module / library loading and linking (includes PTX JIT) */                \
  X(cuModuleLoad)                                                              \
  X(cuModuleLoadData)                                                          \
  X(cuModuleLoadDataEx)                                                        \
  X(cuModuleLoadFatBinary)                                                     \
  X(cuModuleUnload)                                                            \
  X(cuLibraryLoadData)                                                         \
  X(cuLibraryLoadFromFile)                                                     \
  X(cuLibraryUnload)                                                           \
  X(cuLinkComplete)                                                            \
  /* Graph instantiation / update / upload */                                  \
  X(cuGraphInstantiate)                                                        \
  X(cuGraphInstantiate_v2)                                                     \
  X(cuGraphInstantiateWithFlags)                                               \
  X(cuGraphInstantiateWithParams)                                              \
  X(cuGraphInstantiateWithParams_ptsz)                                         \
  X(cuGraphExecUpdate)                                                         \
  X(cuGraphExecUpdate_v2)                                                      \
  X(cuGraphUpload)                                                             \
  X(cuGraphUpload_ptsz)

struct BlockingCallback {
  CUpti_CallbackDomain domain;
  CUpti_CallbackId cbid;
};

#define PARCAGPU_RUNTIME_ENTRY(name)                                           \
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_##name},
#define PARCAGPU_DRIVER_ENTRY(name)                                            \
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_##name},
static constexpr BlockingCallback kBlockingCallbacks[] = {
    PARCAGPU_BLOCKING_RUNTIME_CBIDS(PARCAGPU_RUNTIME_ENTRY)
        PARCAGPU_BLOCKING_DRIVER_CBIDS(PARCAGPU_DRIVER_ENTRY)};
#undef PARCAGPU_RUNTIME_ENTRY
#undef PARCAGPU_DRIVER_ENTRY

// Runs on every runtime/driver callback, including kernel launches, so it is
// a switch rather than a scan of kBlockingCallbacks.
static bool isBlockingCb(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
#define PARCAGPU_RUNTIME_CASE(name) case CUPTI_RUNTIME_TRACE_CBID_##name:
#define PARCAGPU_DRIVER_CASE(name) case CUPTI_DRIVER_TRACE_CBID_##name:
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    switch (cbid) {
      PARCAGPU_BLOCKING_RUNTIME_CBIDS(PARCAGPU_RUNTIME_CASE)
      return true;
    default:
      return false;
    }
  }
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    switch (cbid) {
      PARCAGPU_BLOCKING_DRIVER_CBIDS(PARCAGPU_DRIVER_CASE)
      return true;
    default:
      return false;
    }
  }
  return false;
#undef PARCAGPU_RUNTIME_CASE
#undef PARCAGPU_DRIVER_CASE
}

// Per-thread timing of the in-flight blocking call. A runtime call usually
// wraps a driver call with the same correlation ID; only the runtime
// call is reported (same convention as cuda_correlation).
struct BlockingState {
  bool runtimeActive = false;
  uint32_t runtimeCorrelationId = 0;
  uint64_t runtimeStartNs = 0; // 0 = not timed (probe was detached at ENTER)
  uint64_t driverStartNs = 0;
};
thread_local BlockingState g_blockingState;

static bool isNestedDriverBlocking(CUpti_CallbackDomain domain,
                               uint32_t correlationId) {
  const auto &ss = g_blockingState;
  return domain == CUPTI_CB_DOMAIN_DRIVER_API && ss.runtimeActive &&
         ss.runtimeCorrelationId == correlationId;
}

static void blockingEnter(CUpti_CallbackDomain domain, uint32_t correlationId) {
  auto &ss = g_blockingState;
  // Track runtime nesting even when detached, so attaching mid-call
  // can't report the nested driver call on its own.
  uint64_t now = PARCAGPU_CUDA_BLOCKING_CALL_ENABLED() ? nowNs() : 0;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    ss.runtimeActive = true;
    ss.runtimeCorrelationId = correlationId;
    ss.runtimeStartNs = now;
  } else if (!isNestedDriverBlocking(domain, correlationId)) {
    ss.driverStartNs = now;
  }
}

// Device of the context the API ran on. cuCtxGetDevice reads the current
// context, which CUPTI callbacks run with; cached per thread by context.
static uint32_t blockingDeviceId(CUcontext ctx) {
  thread_local CUcontext cachedCtx = nullptr;
  thread_local uint32_t cachedDev = UINT32_MAX;
  if (ctx != nullptr && ctx == cachedCtx)
    return cachedDev;
  try {
    CUdevice dev = 0;
    if (proton::cuda::ctxGetDevice<false>(&dev) != CUDA_SUCCESS)
      return UINT32_MAX;
    if (ctx != nullptr) {
      cachedCtx = ctx;
      cachedDev = (uint32_t)dev;
    }
    return (uint32_t)dev;
  } catch (...) {
    return UINT32_MAX;
  }
}

// endNs is taken at the top of the EXIT callback, before any PC-sampling
// drain, so our own work is not counted as blocked time.
static void blockingExit(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                     uint32_t correlationId, CUcontext ctx, uint64_t endNs) {
  auto &ss = g_blockingState;
  uint64_t startNs;
  int signedCbid;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    startNs = ss.runtimeStartNs;
    ss.runtimeActive = false;
    ss.runtimeStartNs = 0;
    signedCbid = (int)cbid;
  } else {
    if (isNestedDriverBlocking(domain, correlationId))
      return; // the enclosing runtime call reports it
    startNs = ss.driverStartNs;
    ss.driverStartNs = 0;
    signedCbid = -(int)cbid;
  }
  if (startNs == 0 || endNs == 0 || endNs < startNs)
    return;

  const uint64_t threshold = g_blockingThresholdNs.load(std::memory_order_relaxed);
  const uint64_t duration = endNs - startNs;
  uint64_t weight = duration;
  if (duration < threshold) {
    // Keep with probability duration/T and weight T: E[weight] = duration.
    if (threadRandom(g_pcSamplingState) >= (double)duration / threshold)
      return;
    weight = threshold;
  }

  uint32_t deviceId = blockingDeviceId(ctx);
  DEBUG_PRINTF("[PARCAGPU] cuda_blocking_call: cbid=%d dev=%u duration=%luns "
               "weight=%luns\n",
               signedCbid, deviceId, duration, weight);
  // Keep signedCbid in a register at the probe: the compiler may otherwise
  // pass it as a bare stack slot ([sp]), which bpftrace misreads on arm64.
  asm volatile("" : "+r"(signedCbid));
  PARCAGPU_CUDA_BLOCKING_CALL(startNs, endNs, signedCbid, deviceId, weight);
}

// Called from pc_sampling.cpp:processPCSamplingData() once per CUPTI batch
// with the sum of distinct (PC, stallReason) pairs that had non-zero samples
// — the unit the agent emits as a gpu_pc record on the receive side, and the
// right rate to steer the controller with.
void recordPCSamples(uint64_t n) {
  g_pcController.samplesTotal.fetch_add(n, std::memory_order_relaxed);
}

// Adjust controller.probability to converge on targetRate. Cheap to call —
// returns immediately unless kPCControlPeriodNs has elapsed since the last
// adjustment. Safe under concurrent calls (CAS on lastCheckNs).
static void controllerMaybeUpdate() {
  if (g_pcController.targetRate <= 0.0)
    return;
  uint64_t now = nowNs();
  uint64_t last = g_pcController.lastCheckNs.load(std::memory_order_relaxed);
  if (now - last < kPCControlPeriodNs)
    return;
  // Only one thread per period actually performs the update.
  if (!g_pcController.lastCheckNs.compare_exchange_strong(
          last, now, std::memory_order_acq_rel))
    return;
  uint64_t total =
      g_pcController.samplesTotal.load(std::memory_order_relaxed);
  uint64_t lastTotal = g_pcController.lastCheckTotal.exchange(
      total, std::memory_order_acq_rel);
  uint64_t delta = total - lastTotal;
  uint64_t elapsed = now - last;
  if (elapsed == 0)
    return;
  double observedRate = (double)delta * 1e9 / (double)elapsed;
  double err =
      (observedRate - g_pcController.targetRate) / g_pcController.targetRate;
  if (std::abs(err) <= kPCControlTolerance)
    return;
  double ratio =
      g_pcController.targetRate / std::max(observedRate, 1e-3);
  ratio = std::clamp(ratio, 1.0 / kPCControlStepShrink, kPCControlStepGrow);
  double oldP = g_pcController.probability.load(std::memory_order_relaxed);
  double newP = std::clamp(oldP * ratio, kPCProbMin, kPCProbMax);
  g_pcController.probability.store(newP, std::memory_order_relaxed);
  DEBUG_PRINTF("[PARCAGPU] PC rate controller: observed=%.2f target=%.2f "
               "old_p=%.5f new_p=%.5f\n",
               observedRate, g_pcController.targetRate, oldP, newP);
}

void init_debug() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    debug_enabled = getenv("PARCAGPU_DEBUG") != nullptr;
    // An unset (or invalid) limit is its default split across the
    // machine's GPUs.
    const uint32_t gpus = countMachineGpus();
    auto rateLimit = [gpus](const char *name, double base) {
      const char *env = getenv(name);
      double rate = env ? atof(env) : 0;
      if (rate > 0)
        return rate;
      rate = splitRateLimit(base, gpus);
      DEBUG_PRINTF("[PARCAGPU] %s: %.2f/s per thread (%.0f/s across %u GPUs "
                   "on the machine)\n",
                   name, rate, base, gpus);
      return rate;
    };
    g_eagerRateLimit.store(
        rateLimit("PARCAGPU_RATE_LIMIT", kDefaultLaunchRateLimit),
        std::memory_order_relaxed);
    g_graphRateLimit.store(
        rateLimit("PARCAGPU_GRAPH_RATE_LIMIT", kDefaultGraphRateLimit),
        std::memory_order_relaxed);

    const char *blockingThresholdEnv = getenv("PARCAGPU_BLOCKING_THRESHOLD_US");
    if (blockingThresholdEnv != nullptr) {
      double us = atof(blockingThresholdEnv);
      if (us >= 0.0) {
        g_blockingThresholdNs.store((uint64_t)(us * 1000.0),
                                std::memory_order_relaxed);
      }
    }

    const char *targetRateEnv = getenv("PARCAGPU_PC_SAMPLING_RATE");
    if (targetRateEnv) {
      double r = atof(targetRateEnv);
      if (r > 0.0)
        g_pcController.targetRate = r;
    }
    // Scale initial probability with target rate so high-target requests
    // start sampling immediately. The controller fires only when a
    // sampling window closes; if the workload launches kernels rarely
    // (vortex-class: ~1 launch/sec) and the initial probability is too
    // low for a window to open, the controller starves and can never
    // climb. Scaling avoids that for the common "user wants lots of
    // samples on a sparse workload" case.
    {
      double scaled = kPCInitialProbabilityAtDefaultRate *
                      (g_pcController.targetRate / kPCDefaultTargetRate);
      double initP = std::clamp(scaled, kPCProbMin, kPCProbMax);
      g_pcController.probability.store(initP, std::memory_order_relaxed);
    }
    g_pcController.lastCheckNs.store(nowNs(), std::memory_order_relaxed);

    validateEnvVars();
    initialized = true;
  }
}

// Out-of-line USDT probe site for activity batches.
// Single call site ensures one probe location in the ELF .note.stapsdt section.
static constexpr int ACTIVITY_BATCH_SIZE = 128;

} // namespace parcagpu

__attribute__((noinline)) void parcagpuActivityBatch(const void **ptrs,
                                                     uint32_t count) {
  PARCAGPU_ACTIVITY_BATCH(ptrs, count);
}

namespace parcagpu {

// Simplified profiler using Proton's patterns
class CuptiProfiler : public proton::Singleton<CuptiProfiler> {
public:
  CuptiProfiler() {
    DEBUG_PRINTF("[PARCAGPU] Initializing ParcaGPUProfiler\n");
  }

  ~CuptiProfiler() { cleanup(); }

  bool initialize() {
    if (initialized.exchange(true)) {
      return true; // Already initialized
    }

    DEBUG_PRINTF("[PARCAGPU] Starting initialization\n");

    // cuda_blocking_call is opt-in: CUPTI has no probe-attach hook, so its
    // callbacks would otherwise cost every blocking call (including polling
    // queries) even with no agent attached.
    if (const char *v = getenv("PARCAGPU_BLOCKING_CALLS");
        v != nullptr && strcmp(v, "0") != 0 && *v != '\0') {
      blockingCallsEnabled = true;
      DEBUG_PRINTF("[PARCAGPU] Blocking-call sampling enabled\n");
    }

    parcagpu::agentCtlInit();

    // Check if PC sampling is supported: asked for by the environment, or
    // possible under agent control (contexts are set up for it, but sample
    // only when the agent says so).
    pcSamplingEnabled =
        parcagpu::PCSampling::isSupported(parcagpu::agentCtlReady());
    pcSampling.envSampling = parcagpu::PCSampling::envRequested();
    parcagpu::agentCtlSetCapable(pcSamplingEnabled);
    init_debug(); // reads the environment's rate limits
    // Reported as whole events/s (the split defaults are fractional), at
    // least 1: the state's 0 would read as "not set".
    auto reported = [](const std::atomic<double> &r) {
      double v = r.load(std::memory_order_relaxed);
      return v < 1.0 ? 1u : (uint32_t)(v + 0.5);
    };
    parcagpu::agentCtlSetEnvRateLimits(reported(g_eagerRateLimit),
                                       reported(g_graphRateLimit));
    parcagpu::agentCtlSetEnvFeatures(
        PARCAGPU_FEATURE_KERNEL_STATS |
        (blockingCallsEnabled ? PARCAGPU_FEATURE_BLOCKING_CALLS : 0));
    if (pcSamplingEnabled) {
      DEBUG_PRINTF("[PARCAGPU] PC sampling enabled (serialized mode)\n");
    } else {
      DEBUG_PRINTF(
          "[PARCAGPU] PC sampling disabled, using kernel activity only\n");
    }

    // Subscribe to callbacks
    auto result =
        proton::cupti::subscribe<true>(&subscriber, callbackHandler, nullptr);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] Failed to subscribe to callbacks: error %d\n",
                   result);
      return false;
    }

    // Enable runtime and driver API callbacks (using Proton's utilities)
    proton::setRuntimeCallbacks(subscriber, /*enable=*/true);
    proton::setLaunchCallbacks(subscriber, /*enable=*/true);
    // Where PC sampling may run, every driver API call passes through the
    // gate (pc_gate.h), so that it can change state with no other thread in
    // the driver. The gate counts from before the first callback.
    if (pcSamplingEnabled) {
      parcagpu::gate::setOverdueHandler(closeOverdueWindow);
      pcSampling.captureFree = PCTransition::captureFree;
      parcagpu::gate::enable();
      proton::cupti::enableDomain<true>(1, subscriber,
                                        CUPTI_CB_DOMAIN_DRIVER_API);
    }

    // Driver-API graph launches (cuGraphLaunch/_ptsz); not covered by setLaunch/
    // setRuntimeCallbacks. Only these two cbids -- not setGraphCallbacks(), which
    // also subscribes capture cbids that emit uncorrelated events.
    proton::cupti::enableCallback<true>(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
    proton::cupti::enableCallback<true>(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);

    // Blocking-class callbacks for the cuda_blocking_call probe, when
    // PARCAGPU_BLOCKING_CALLS is set; under an agent, once it turns them on.
    syncBlockingCallbacks();

    // Context creation always: it drives the device_info probe and feeds
    // the device table reported to the agent. The remaining resource
    // callbacks only feed PC sampling.
    if (pcSamplingEnabled) {
      proton::setResourceCallbacks(subscriber, /*enable=*/true);
      proton::cupti::enableCallback<true>(1, subscriber,
                                          CUPTI_CB_DOMAIN_DRIVER_API,
                                          CUPTI_DRIVER_TRACE_CBID_cuModuleLoad);
    } else {
      proton::cupti::enableCallback<false>(
          1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
          CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
    }

    // NVTX goes through our own injection (nvtx_injection.cpp), not CUPTI.
    // Point NVTX clients that haven't initialized yet at this library.
    parcagpu::nvtx::onCudaInjected();

    // Register activity buffer callbacks (using Proton's pattern)
    result = proton::cupti::activityRegisterCallbacks<true>(allocBuffer,
                                                            completeBuffer);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF(
          "[PARCAGPU] Failed to register activity callbacks: error %d\n",
          result);
      return false;
    }

    // Enable kernel activity recording
    result = proton::cupti::activityEnable<true>(
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF(
          "[PARCAGPU] Failed to enable concurrent kernel activity: error %d\n",
          result);
    } else {
      DEBUG_PRINTF("[PARCAGPU] Enabled CONCURRENT_KERNEL activity\n");
    }

    syncFlushPeriod();

    parcagpu::agentCtlStart(applyAgentConfig);

    DEBUG_PRINTF("[PARCAGPU] Successfully initialized CUPTI callbacks\n");
    return true;
  }

  void cleanup() {
    if (!initialized.exchange(false)) {
      return; // Already cleaned up
    }

    DEBUG_PRINTF("[PARCAGPU] Cleanup started\n");

    parcagpu::agentCtlShutdown();

    // PC sampling data is drained in finalize() during CONTEXT_DESTROY_STARTING
    // when the CUDA context is still valid. By the time cleanup() runs, the
    // context may already be dead, so we don't drain here.

    // Disable all callbacks
    if (subscriber) {
      proton::setRuntimeCallbacks(subscriber, /*enable=*/false);
      proton::setLaunchCallbacks(subscriber, /*enable=*/false);
      if (pcSamplingEnabled)
        proton::cupti::enableDomain<false>(0, subscriber,
                                           CUPTI_CB_DOMAIN_DRIVER_API);
      // Mirror the graph-launch callbacks enabled in init().
      proton::cupti::enableCallback<false>(0, subscriber,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
      proton::cupti::enableCallback<false>(
          0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
          CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
      // Mirror the blocking callbacks enabled by syncBlockingCallbacks().
      if (blockingCallbacksOn.load(std::memory_order_acquire)) {
        for (const auto &cb : kBlockingCallbacks)
          proton::cupti::enableCallback<false>(0, subscriber, cb.domain,
                                               cb.cbid);
      }
      if (pcSamplingEnabled) {
        proton::setResourceCallbacks(subscriber, /*enable=*/false);
        proton::cupti::enableCallback<false>(
            0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
            CUPTI_DRIVER_TRACE_CBID_cuModuleLoad);
      } else {
        proton::cupti::enableCallback<false>(
            0, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
            CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
      }
    }
    // Flush this thread's pending NVTX batch; other live threads drain
    // via the pthread_key destructor on their own exit.
    parcagpu::nvtx::flushCurrentThread();

    // Cleanup runs from atexit and may also reach us via a libcupti callback
    // whose caller wasn't built with C++ EH; throwing from here aborts the
    // process. Use <false> + log on the cleanup-path calls.
    if (auto r = proton::cupti::activityFlushAll<false>(
            CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        r != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] activityFlushAll failed: %d\n", r);
    }
    // The forced flush ran completeBuffer for the last records; report them.
    kernel_stats::maybeEmit(/*force=*/true);

    if (auto r = proton::cupti::activityDisable<false>(
            CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        r != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] activityDisable failed: %d\n", r);
    }

    if (subscriber) {
      if (auto r = proton::cupti::unsubscribe<false>(subscriber);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[PARCAGPU] unsubscribe failed: %d\n", r);
      }
      subscriber = nullptr;
    }

    DEBUG_PRINTF("[PARCAGPU] Cleanup completed\n");
  }

private:
  std::atomic<bool> initialized{false};
  bool pcSamplingEnabled = false;
  bool blockingCallsEnabled = false; // PARCAGPU_BLOCKING_CALLS
  // The blocking-class callbacks are enabled. Once on they stay on: under PC
  // sampling the driver domain is enabled as a whole for the gate, and
  // disabling one of its cbids would take that call out of the gate. While
  // blocking calls are off, their callbacks return at once.
  std::atomic<bool> blockingCallbacksOn{false};

  // Whether cuda_blocking_call is on: the agent's feature bit, else
  // PARCAGPU_BLOCKING_CALLS.
  bool blockingCallsOn() const {
    return parcagpu::agentCtlFeature(PARCAGPU_FEATURE_BLOCKING_CALLS,
                                     blockingCallsEnabled);
  }

  // Enables the blocking-class callbacks the first time blocking calls are
  // on. <false>: an older CUPTI rejecting one cbid must not fail injection.
  static void syncBlockingCallbacks() {
    auto &p = instance();
    if (p.blockingCallbacksOn.load(std::memory_order_acquire) ||
        !p.blockingCallsOn())
      return;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (p.blockingCallbacksOn.load(std::memory_order_relaxed))
      return;
    for (const auto &cb : kBlockingCallbacks) {
      if (auto r = proton::cupti::enableCallback<false>(1, p.subscriber,
                                                        cb.domain, cb.cbid);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[PARCAGPU] enableCallback(domain=%d, cbid=%u) failed: "
                     "%d\n",
                     cb.domain, cb.cbid, r);
      }
    }
    p.blockingCallbacksOn.store(true, std::memory_order_release);
    DEBUG_PRINTF("[PARCAGPU] Blocking-call callbacks enabled\n");
  }
  CUpti_SubscriberHandle subscriber = nullptr;

  // PC sampling state — owned by this profiler, destroyed with it.
  parcagpu::PCSampling pcSampling;

  // Outstanding event counter for flushing
  size_t outstandingEvents = 0;

  // False (retried) while a stream capture is in progress.
  static bool applyAgentConfig(const std::function<void()> &commit) {
    bool done;
    {
      PCTransition transition;
      done = CuptiProfiler::instance().pcSampling.applyAgentConfig(commit);
    }
    if (done) {
      syncFlushPeriod();
      syncBlockingCallbacks();
    }
    return done;
  }

  // While kernel stats are on, have CUPTI's worker hand back partly filled
  // buffers every second, so kernel_stats (and kernel_executed) stay current
  // on workloads that launch too few kernels to fill a buffer; while they
  // are off, leave CUPTI's default. Best effort: Dispatch throws if this
  // libcupti lacks the symbol.
  static void syncFlushPeriod() {
    static std::mutex mu;
    static int applied = -1; // flush period in ms; -1 = never set
    std::lock_guard<std::mutex> lock(mu);
    const int want = kernel_stats::enabled() ? 1000 : 0;
    if (want == applied)
      return;
    try {
      if (auto r = proton::cupti::activityFlushPeriod<false>(want);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[PARCAGPU] activityFlushPeriod(%d) failed: %d\n", want,
                     r);
        return;
      }
      applied = want;
      DEBUG_PRINTF("[PARCAGPU] activityFlushPeriod(%d)\n", want);
    } catch (const std::exception &e) {
      applied = want; // unavailable: don't retry
      DEBUG_PRINTF("[PARCAGPU] activityFlushPeriod unavailable: %s\n",
                   e.what());
    }
  }

  // A stream capture is beginning on this thread (its BeginCapture ENTER,
  // not yet in the gate): count it, wait for PC sampling state changes in
  // progress, then close the open windows, regardless of the gate's backoff.
  // No new one opens until the capture ends.
  static void beforeCapture() {
    g_streamCaptures.fetch_add(1);
    auto &profiler = instance();
    if (!profiler.pcSamplingEnabled)
      return;
    const uint64_t start = nowNs();
    while (g_pcTransitions.load() > 0 && nowNs() - start < kCaptureWaitNs) {
      struct timespec ts = {0, 1'000'000};
      nanosleep(&ts, nullptr);
    }
    InternalScope internal;
    try {
      for (CUcontext ctx : profiler.pcSampling.samplingContextsNow()) {
        bool pushed = parcagpu::agentCtlPushContext(ctx);
        profiler.pcSampling.stop(ctx, parcagpu::gate::kCloseTimeoutNs,
                                 /*backoff=*/false);
        if (pushed)
          parcagpu::agentCtlPopContext();
      }
    } catch (...) {
    }
  }

  // The gate's overdue handler: a thread waiting at the gate closes a window
  // whose owner went quiet (pc_gate.h). It's at its call's ENTER, holding
  // none of our locks; the context needn't be its current one.
  static bool closeOverdueWindow(CUcontext ctx) {
    InternalScope internal;
    try {
      bool pushed = parcagpu::agentCtlPushContext(ctx);
      bool stopped = CuptiProfiler::instance().pcSampling.stop(
          ctx, parcagpu::gate::kOverdueCloseTimeoutNs, /*backoff=*/false);
      if (pushed)
        parcagpu::agentCtlPopContext();
      return stopped;
    } catch (...) {
      return false;
    }
  }

  // How long a PC sampling window stays open: the agent's length if it set
  // one. Dice rolls stay kPCSamplingIntervalNs apart.
  static uint64_t pcWindowNs(const AgentConfig &cfg) {
    return cfg.active && cfg.windowMs ? cfg.windowMs * 1'000'000ULL
                                      : kPCSamplingIntervalNs;
  }

  // Buffer management - using Proton's pattern (static methods)
  // A kernel activity is around 224 bytes so a 128kb buffer
  // will hold ~500 activities, we want to flush regularly since
  // we are a continuous profiler so we don't need a huge buffer
  // like most CUPTI profilers.  Also a small size avoids malloc
  // just going to mmap every time so the allocator should cache
  // and re-use these for us.
  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 128 * 1024;

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords) {
    if (!PARCAGPU_CUDA_CORRELATION_ENABLED()) {
      *buffer = nullptr;
      return;
    }
    *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
    if (*buffer == nullptr) {
      DEBUG_PRINTF("[PARCAGPU] ERROR: aligned_alloc failed\n");
      return;
    }
    *bufferSize = BufferSize;
    *maxNumRecords = 0;
    DEBUG_PRINTF("[PARCAGPU:allocBuffer] Allocated buffer at %p size %zu\n",
                 *buffer, *bufferSize);
  }

  static void completeBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                             size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    int recordCount = 0;
    int filteredCount = 0;

    // Batch probe: collect pointers to activity records and pass them to
    // BPF/USDT every ACTIVITY_BATCH_SIZE records. Stack-allocated array
    // of pointers — no heap allocation, no copying, version-independent.
    const void *batchPtrs[ACTIVITY_BATCH_SIZE];
    uint32_t batchCount = 0;

    DEBUG_PRINTF(
        "[PARCAGPU] completeBuffer called: ctx=%p buffer=%p validSize=%zu\n",
        ctx, buffer, validSize);

    // Start a new buffer cycle for graph correlation tracking
    uint32_t cycle = g_bufferCycle.fetch_add(1);
    g_graphCorrelationMap.cycle_start(cycle);

    while (true) {
      CUptiResult result = proton::cupti::activityGetNextRecord<false>(
          buffer, validSize, &record);
      if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else if (result != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[PARCAGPU] Error reading activity record: error %d\n",
                     result);
        break;
      }

      recordCount++;
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        auto *k = reinterpret_cast<CUpti_ActivityKernel5 *>(record);

        // Count every execution, sampled or not, before the filter below
        // drops the launches the rate limiter skipped.
        kernel_stats::record(k);

        // Check correlation filter - only emit probe if this kernel was sampled
        bool shouldEmit = false;
        if (k->graphId != 0) {
          // Graph kernel - check graph correlation map
          shouldEmit = g_graphCorrelationMap.check_and_mark_seen(
              k->correlationId, cycle);
        } else {
          // Regular kernel - check and remove from correlation filter
          shouldEmit = g_correlationFilter.check_and_remove(k->correlationId);
        }

        if (!shouldEmit) {
          filteredCount++;
          DEBUG_PRINTF("[PARCAGPU] Filtered kernel activity: correlationId=%u "
                       "graphId=%u (not in filter)\n",
                       k->correlationId, k->graphId);
          // Skip both KERNEL_EXECUTED and activity_batch push. Without this,
          // the eBPF activity_batch consumer would emit a kernel_event for
          // every rate-limited launch, producing orphan entries in parca-agent's
          // timesAwaitingTraces map (no cuda_correlation USDT was emitted for
          // these correlation IDs, so no trace will ever arrive to match them).
          break;
        }

        DEBUG_PRINTF("[PARCAGPU] Kernel activity: graphId=%u graphNodeId=%lu "
                     "name=%s, correlationId=%u, deviceId=%u, "
                     "streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
                     k->graphId, k->graphNodeId, k->name, k->correlationId,
                     k->deviceId, k->streamId, k->start, k->end,
                     k->end - k->start);

        // Emit USDT probe for kernel execution
        PARCAGPU_KERNEL_EXECUTED(k->start, k->end, k->correlationId,
                                 k->deviceId, k->streamId, k->graphId,
                                 k->graphNodeId, k->name);

        // Collect pointer for batch probe — only for kernel records that
        // passed the correlation filter. eBPF's activity_batch handler only
        // emits kernel_events for KERNEL/CONCURRENT_KERNEL kinds anyway
        // (cuda.ebpf.c kind check), so excluding non-kernel kinds here saves
        // ringbuf bandwidth without losing any consumed data.
        batchPtrs[batchCount++] = record;
        if (batchCount >= ACTIVITY_BATCH_SIZE) {
          parcagpuActivityBatch(batchPtrs, batchCount);
          batchCount = 0;
        }
        break;
      }
      default:
        DEBUG_PRINTF("[PARCAGPU] Activity record %d: kind=%d\n", recordCount,
                     record->kind);
        break;
      }
    }

    // Flush remaining batch
    if (batchCount > 0) {
      parcagpuActivityBatch(batchPtrs, batchCount);
    }

    // End cycle - cleanup completed graph entries
    g_graphCorrelationMap.cycle_end();

    kernel_stats::maybeEmit(/*force=*/false);

    DEBUG_PRINTF("[PARCAGPU] Processed %d activity records (%d filtered) from "
                 "buffer %p\n",
                 recordCount, filteredCount, buffer);

    // Reset to 0 rather than decrement - one API callback can produce N
    // activities so decrementing by recordCount can cause underflow
    CuptiProfiler::instance().outstandingEvents = 0;

    // Free the buffer (Proton's pattern)
    std::free(buffer);
  }

  static void callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const void *cbdata_void) {
    // libcupti invokes us from contexts whose callers weren't built with
    // C++ EH support, so any uncaught throw aborts the process. Treat the
    // whole body as untrusted and swallow exceptions at the boundary.
    // Every driver API call passes the gate: in at ENTER, out at the end
    // of its EXIT callback (whatever that does, it does still inside).
    struct GateExit {
      CUpti_CallbackId cbid;
      bool active;
      ~GateExit() {
        if (active)
          parcagpu::gate::exit(cbid);
      }
    } gateExit{cbid, false};
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
      const auto *cbdata = static_cast<const CUpti_CallbackData *>(cbdata_void);
      if (cbdata->callbackSite == CUPTI_API_ENTER) {
        if (isBeginCapture(cbid) && !InternalScope::active())
          CuptiProfiler::beforeCapture(); // before the gate counts it
        parcagpu::gate::enter(cbid, InternalScope::active());
      } else {
        gateExit.active = true;
        trackStreamCapture(cbid, cbdata);
      }
    }
    // A driver call parcagpu made itself, from this handler or the release
    // thread: nothing more to do, and the caller may hold our locks.
    if (InternalScope::active())
      return;
    InternalScope internal;
    try {
    auto &profiler = CuptiProfiler::instance();

    // Not from resource callbacks: they run inside the driver, where
    // applying a change can't close the gate.
    if (domain != CUPTI_CB_DOMAIN_RESOURCE)
      parcagpu::agentCtlPoll();

    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
      const CUpti_ResourceData *resData =
          static_cast<const CUpti_ResourceData *>(cbdata_void);
      if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
        device_info::onContextCreated(resData->context);
        parcagpu::agentCtlAddContext(resData->context);
      }

      // The rest is for PC sampling (only if enabled)
      if (!profiler.pcSamplingEnabled) {
        return;
      }

      switch (cbid) {
      case CUPTI_CBID_RESOURCE_MODULE_LOADED: {
        const CUpti_ModuleResourceData *modData =
            static_cast<const CUpti_ModuleResourceData *>(
                resData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
          DEBUG_PRINTF("[PARCAGPU] Module loaded: cubin=%p size=%zu\n",
                       modData->pCubin, modData->cubinSize);
          profiler.pcSampling.loadModule(
              modData->pCubin, modData->cubinSize,
              moduleLoadPaths.empty() ? "" : moduleLoadPaths.back());
        }
        break;
      }
      case CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING: {
        const CUpti_ModuleResourceData *modData =
            static_cast<const CUpti_ModuleResourceData *>(
                resData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
          DEBUG_PRINTF("[PARCAGPU] Module unloading: cubin=%p size=%zu\n",
                       modData->pCubin, modData->cubinSize);
          profiler.pcSampling.unloadModule(modData->pCubin, modData->cubinSize);
        }
        break;
      }
      case CUPTI_CBID_RESOURCE_CONTEXT_CREATED: {
        CUcontext ctx = resData->context;
        DEBUG_PRINTF("[PARCAGPU] Context created: %p\n", ctx);
        // Under agent control a context is set up for PC sampling only once
        // its device is granted (PCSampling::prepare), as on a GB10 the first
        // process to enable it owns the GPU's sampler.
        if (profiler.pcSampling.envSampling && !agentCtlConfig().active) {
          // Inside the driver, with the new context not yet in use.
          parcagpu::gate::Bypass bypass;
          profiler.pcSampling.initialize(ctx, 0);
        }
        break;
      }
      case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING: {
        CUcontext ctx = resData->context;
        DEBUG_PRINTF("[PARCAGPU] Context destroying: %p\n", ctx);
        // Inside the driver, with the context no longer in use.
        parcagpu::gate::Bypass bypass;
        profiler.pcSampling.finalize(ctx);
        break;
      }
      default:
        break;
      }
    } else {
      // Handle both Runtime and Driver API callbacks
      const CUpti_CallbackData *cbdata =
          static_cast<const CUpti_CallbackData *>(cbdata_void);
      if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
          cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoad) {
        if (cbdata->callbackSite == CUPTI_API_ENTER) {
          const auto *params =
              static_cast<const cuModuleLoad_params *>(cbdata->functionParams);
          char *path = params && params->fname
                           ? realpath(params->fname, nullptr)
                           : nullptr;
          moduleLoadPaths.emplace_back(path ? path : "");
          free(path);
        } else if (!moduleLoadPaths.empty()) {
          moduleLoadPaths.pop_back();
        }
        return;
      }
      uint32_t correlationId = cbdata->correlationId;

      // PC sampling windows are aligned to kernel-launch boundaries:
      // start on a launch ENTER, stop on a launch EXIT. The decision to
      // open a window is still time + probability gated; the launch CBIDs
      // just determine when the boundaries fire.
      const bool isKernelLaunchCb =
          domain == CUPTI_CB_DOMAIN_DRIVER_API && proton::isLaunch(cbid);
      // Blocking-class calls (cuda_blocking_call probe) never take the launch
      // correlation / rate-limiter path below.
      // A blocking-class callback while blocking calls are off does nothing
      // more than the gate's.
      const bool blockingCb = isBlockingCb(domain, cbid);
      const bool isBlocking = blockingCb && profiler.blockingCallsOn();
      // Other driver calls are subscribed for the gate only (and drains).
      const bool gateOnlyCb =
          (domain == CUPTI_CB_DOMAIN_DRIVER_API && !isKernelLaunchCb &&
           !isBlocking) ||
          (blockingCb && !isBlocking);

      // ENTER: open a sampling window on launch boundary if interval+prob hits.
      if (cbdata->callbackSite == CUPTI_API_ENTER) {
        if (gateOnlyCb)
          return;
        // Blocking calls keep their own nesting state so they don't overwrite
        // the launch correlation ID that NVTX events read.
        if (isBlocking)
          blockingEnter(domain, correlationId);
        else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
          runtimeEnterCorrelationId = correlationId;

        if (profiler.pcSamplingEnabled) {
          // Sample if the agent's configuration grants this context's
          // device, or with no agent configuration if the environment asks.
          const AgentConfig cfg = agentCtlConfig();
          if (isKernelLaunchCb &&
              (cfg.active || profiler.pcSampling.envSampling) &&
              g_streamCaptures.load(std::memory_order_relaxed) == 0) {
            auto &st = g_pcSamplingState;
            uint64_t now = nowNs();

            if ((now - st.lastCheckNs >= kPCSamplingIntervalNs) &&
                agentCtlAllows(cbdata->context,
                               profiler.pcSampling.envSampling)) {
              st.lastCheckNs = now;
              PCTransition transition;
              if (!PCTransition::captureFree())
                return;
              // First set the context up, bring it to the agent's factor, or
              // re-arm it after a failure (skipped while it samples). The
              // context is current here.
              profiler.pcSampling.prepare(cbdata->context,
                                          cfg.active ? cfg.factor : 0,
                                          agentCtlGeneration(), cfg.active);
              // Under the agent, its window and gap set the duty cycle (per
              // context, in start()); else the dice roll, steered by the
              // rate controller.
              bool open = cfg.active ||
                          threadRandom(st) < g_pcController.probability.load(
                                                 std::memory_order_relaxed);
              // start() is false if this context is already sampling or
              // can't sample.
              if (open)
                profiler.pcSampling.start(cbdata->context,
                                          cfg.active ? cfg.gapMs * 1'000'000ULL
                                                     : 0,
                                          pcWindowNs(cfg));
            }
          }

          profiler.pcSampling.emitMetadata();
        }
        device_info::emit();
        return;
      }

      // Process on EXIT to avoid adding latency to GPU launch
      if (cbdata->callbackSite != CUPTI_API_EXIT) {
        return;
      }

      // Take the blocking-call end time before the PC-sampling drain below.
      const uint64_t blockingEndNs =
          isBlocking && PARCAGPU_CUDA_BLOCKING_CALL_ENABLED() ? nowNs() : 0;

      // EXIT: while this callback's context has a window open, drain
      // CUPTI's host staging buffer on every CUDA API EXIT — not just launch
      // EXITs. Empty drains are cheap (single API call returning 0 PCs) and
      // missed drains lose samples (CUPTI_ERROR_OUT_OF_MEMORY when staging
      // fills). Window close is still kernel-launch aligned: only check
      // elapsed-time on a launch EXIT, so the window starts and ends at
      // kernel boundaries. Any thread does this, not just the one that
      // opened the window; while the gate holds the others back, one
      // waiting for an overdue window closes it (closeOverdueWindow).
      if (profiler.pcSamplingEnabled) {
        CUcontext ctx = cbdata->context;
        if (ctx && profiler.pcSampling.isSampling(ctx)) {
          const AgentConfig cfg = agentCtlConfig();
          if (isKernelLaunchCb &&
              profiler.pcSampling.windowDue(ctx, pcWindowNs(cfg))) {
            if (profiler.pcSampling.stop(ctx) && !cfg.active)
              controllerMaybeUpdate();
          } else {
            profiler.pcSampling.collectData(ctx);
          }
        }

        if (!gateOnlyCb)
          profiler.pcSampling.emitMetadata();
      }
      if (gateOnlyCb)
        return;
      device_info::emit();

      if (isBlocking) {
        blockingExit(domain, cbid, correlationId, cbdata->context, blockingEndNs);
        return;
      }

      // Skip correlation/rate-limiter work when no profiler is attached.
      if (!PARCAGPU_CUDA_CORRELATION_ENABLED())
        return;

      const char *name =
          cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
      int signedCbid;

      if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        // Skip if this driver call is under a runtime call (same correlation
        // ID)
        if (correlationId == runtimeEnterCorrelationId) {
          DEBUG_PRINTF("[PARCAGPU] Skipping driver EXIT correlationId=%u - "
                       "runtime will handle\n",
                       correlationId);
          return;
        }
        // Pure driver call (no runtime wrapper) - use negative cbid
        signedCbid = -(int)cbid;
        DEBUG_PRINTF("[PARCAGPU] Driver API callback: cbid=%d, "
                     "correlationId=%u, func=%s\n",
                     cbid, correlationId, name);
      } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        signedCbid = (int)cbid;
        runtimeEnterCorrelationId = 0; // Clear after use
        DEBUG_PRINTF("[PARCAGPU] Runtime API callback: cbid=%d, "
                     "correlationId=%u, func=%s\n",
                     cbid, correlationId, name);
      } else {
        return;
      }

      // Check if this is a graph launch (rate-limited via a separate bucket).
      bool isGraphLaunch = false;
      if (signedCbid < 0) {
        // Driver API: cuGraphLaunch = 514, cuGraphLaunch_ptsz = 515
        int driverCbid = -signedCbid;
        isGraphLaunch =
            (driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
             driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
      } else {
        // Runtime API: cudaGraphLaunch = 311, cudaGraphLaunch_ptsz = 312
        isGraphLaunch =
            (signedCbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
             signedCbid ==
                 CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
      }

      // Rate limit via per-class buckets (eager vs graph). Skip when this
      // launch's context is PC sampling, whichever thread opened the window:
      // every kernel needs its correlation callback to match PC samples to
      // CPU stacks on the agent side.
      if (!profiler.pcSampling.isSampling(cbdata->context)) {
        applyRateLimits();
        TokenBucket &limiter =
            isGraphLaunch ? graphCallbackLimiter : callbackLimiter;
        if (!limiter.tryAcquire()) {
          DEBUG_PRINTF("[PARCAGPU] Rate limited: skipping probe for "
                       "correlationId=%u (graph=%d)\n",
                       correlationId, isGraphLaunch);
          return;
        }
      }

      profiler.outstandingEvents++;
      // Emit USDT probe with signed cbid (negative for driver, positive for
      // runtime)
      PARCAGPU_CUDA_CORRELATION(correlationId, signedCbid, name);

      // Insert into correlation filter so we can match kernel activities later
      if (isGraphLaunch) {
        g_graphCorrelationMap.insert(correlationId);
        DEBUG_PRINTF("[PARCAGPU] Inserted correlationId=%u into graph map\n",
                     correlationId);
      } else {
        g_correlationFilter.insert(correlationId);
        DEBUG_PRINTF(
            "[PARCAGPU] Inserted correlationId=%u into correlation filter\n",
            correlationId);
      }

      // Flush if too many events pile up
      if (profiler.outstandingEvents > 3000) {
        DEBUG_PRINTF("[PARCAGPU] Flushing: outstandingEvents=%zu\n",
                     profiler.outstandingEvents);
        if (auto r = proton::cupti::activityFlushAll<false>(0);
            r != CUPTI_SUCCESS) {
          DEBUG_PRINTF("[PARCAGPU] activityFlushAll failed: %d\n", r);
        }
        profiler.outstandingEvents = 0;
      }
    }
    } catch (const std::exception &e) {
      fprintf(stderr, "[PARCAGPU] callbackHandler caught: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[PARCAGPU] callbackHandler caught unknown exception\n");
    }
  }
};

} // namespace parcagpu

// CUPTI initialization function required for CUDA_INJECTION64_PATH.
// Called from the CUDA driver, which wasn't built with C++ EH; a throw out
// of here would abort the host process. Catch everything at the boundary.
extern "C" int InitializeInjection(void) {
  DEBUG_PRINTF("[PARCAGPU] InitializeInjection called\n");
  try {
    auto &profiler = parcagpu::CuptiProfiler::instance();
    if (!profiler.initialize()) {
      return 0; // Return 0 on failure, but don't break injection
    }
    atexit([]() {
      try {
        parcagpu::CuptiProfiler::instance().cleanup();
      } catch (const std::exception &e) {
        fprintf(stderr, "[PARCAGPU] cleanup caught: %s\n", e.what());
      } catch (...) {
        fprintf(stderr, "[PARCAGPU] cleanup caught unknown exception\n");
      }
    });
    return 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "[PARCAGPU] InitializeInjection caught: %s\n", e.what());
    return 0;
  } catch (...) {
    fprintf(stderr, "[PARCAGPU] InitializeInjection caught unknown exception\n");
    return 0;
  }
}
