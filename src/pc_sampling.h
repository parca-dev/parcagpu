// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PC_SAMPLING_H_
#define PC_SAMPLING_H_

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "stall_reason_map.h"

#include <cuda.h>
#include <cupti.h>

#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Utility/Map.h"
#include "Utility/Set.h"

#define DEBUG_PRINTF(...)                                                      \
  do {                                                                         \
    parcagpu::init_debug();                                                    \
    if (parcagpu::debug_enabled) {                                             \
      struct timespec ts;                                                      \
      clock_gettime(CLOCK_REALTIME, &ts);                                      \
      fprintf(stderr, "[%ld.%09ld] ", ts.tv_sec, ts.tv_nsec);                  \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

namespace parcagpu {

// Debug logging control — defined in cupti.cpp.
extern bool debug_enabled;
extern void init_debug();

// Notify the PC rate controller of a batch of samples just drained from CUPTI.
extern void recordPCSamples(uint64_t n);

// Use Proton's CubinData directly
using proton::CubinData;

// ConfigureData for PARCAGPU (based on Proton's but standalone)
// We don't inherit to avoid linking Proton's profiler dependencies
struct ConfigureData {
  ConfigureData() = default;

  ~ConfigureData() {
    if (stallReasonNames) {
      for (size_t i = 0; i < numStallReasons; i++) {
        if (stallReasonNames[i])
          std::free(stallReasonNames[i]);
      }
      std::free(stallReasonNames);
    }
    if (stallReasonIndices)
      std::free(stallReasonIndices);
    if (pcSamplingData.pPcData) {
      for (size_t i = 0; i < numValidStallReasons; ++i) {
        std::free(pcSamplingData.pPcData[i].stallReason);
      }
      std::free(pcSamplingData.pPcData);
    }
  }

  // factor: sampling period (2^factor cycles); 0 = PARCAGPU_SAMPLING_FACTOR
  // or CUPTI's default.
  void initialize(CUcontext context, uint32_t factor);

  CUpti_PCSamplingConfigurationInfo configureStallReasons();
  CUpti_PCSamplingConfigurationInfo configureSamplingPeriod(uint32_t factor);
  CUpti_PCSamplingConfigurationInfo configureSamplingBuffer();
  CUpti_PCSamplingConfigurationInfo configureScratchBuffer();
  CUpti_PCSamplingConfigurationInfo configureHardwareBufferSize();
  CUpti_PCSamplingConfigurationInfo configureCollectionMode();
  CUpti_PCSamplingConfigurationInfo configureStartStopControl();

  // Buffer size constants (from Proton)
  static constexpr size_t HardwareBufferSize = 128 * 1024 * 1024;
  static constexpr size_t ScratchBufferSize = 16 * 1024 * 1024;
  static constexpr size_t DataBufferPCCount = 1024;

  CUcontext context{};
  uint32_t contextId;
  // Device this context is bound to for its lifetime (see cuCtxCreate /
  // cuCtxGetDevice). Resolved once in PCSampling::initialize() and reused
  // by every pc_sample_batch fired from this context.
  uint32_t device{};
  uint32_t requestedFactor{}; // the factor initialize() was given
  uint32_t numStallReasons{};
  uint32_t numValidStallReasons{};
  int hardwareBufferMB{}; // PARCAGPU_PC_HW_BUFFER_MB; 0 = CUPTI default
  char **stallReasonNames{};
  uint32_t *stallReasonIndices{};
  std::map<size_t, size_t> stallReasonIndexToMetricIndex{};
  std::set<size_t> notIssuedStallReasonIndices{};
  CUpti_PCSamplingData pcSamplingData{};       // registered with CUPTI config
  CUpti_PCSamplingData outputData{};           // separate buffer for getData calls
  std::vector<CUpti_PCSamplingConfigurationInfo> configurationInfos;
};

// PC Sampling class (adapted from Proton's CuptiPCSampling)
// Owned by CuptiProfiler — not a standalone singleton, so lifetime
// is tied to the profiler and there are no static destruction order issues.
class PCSampling {
public:
  PCSampling() = default;
  ~PCSampling() = default;

  // True when PARCAGPU_PC_SAMPLING_RATE asks for PC sampling.
  static bool envRequested();

  // Check if PC sampling is supported (CUPTI >= 12.8.1). Off unless asked
  // for by PARCAGPU_PC_SAMPLING_RATE or possible through agent control.
  static bool isSupported(bool agentControl);

  // Whether contexts sample when the agent doesn't say (envRequested()).
  bool envSampling = false;

  // Enables and configures PC sampling on a new context (not started). factor
  // as for ConfigureData::initialize.
  void initialize(CUcontext context, uint32_t factor);

  // Start PC sampling on `context` — its kernels become serialized until
  // stop(context). Contexts are independent: CUPTI supports one sampled
  // context per GPU, so a process driving several GPUs can sample them all
  // at once. Returns true only if this call started sampling; false if the
  // context is already sampling, not initialized, unavailable (another
  // process owns the GPU's sampler), not allowed (agent control), or CUPTI
  // refused. Under agent control, gapNs is the least time since the
  // context's last window ended. Thread-safe.
  bool start(CUcontext context, uint64_t gapNs = 0);

  // Before sampling on `context` (current, on a launch) where it may sample:
  // enables and configures it if it isn't yet (under agent control, contexts
  // are set up only once granted), re-arms it if it was unavailable or failed
  // (once per configuration gen; only when `agent`, as a grant), and
  // re-configures it if `factor` changed. Skipped while it samples.
  void prepare(CUcontext context, uint32_t factor, uint64_t gen, bool agent);

  // Applies an agent configuration change: commit() under the sampling lock,
  // then stops (and drains) every context it no longer allows, and disables
  // PC sampling on them. On a GB10 the GPU's sampler belongs to the first
  // process that enabled PC sampling, until it disables it: a process must
  // let go of it entirely for another to sample.
  void applyAgentConfig(const std::function<void()> &commit);
  // Stops what the agent no longer allows; returns those contexts.
  std::vector<CUcontext> stopDisallowedLocked();

  // Stop PC sampling on `context` and drain its data. Kernels resume
  // concurrent execution. No-op (false) unless start(context) succeeded
  // earlier.
  bool stop(CUcontext context);

  // True when `context`'s window has been open for windowNs. A window
  // belongs to its context, not the thread that opened it: any thread's
  // launch can close it.
  bool windowDue(CUcontext context, uint64_t windowNs);

  // True while `context` is sampling. Cheap when nothing is sampling.
  bool isSampling(CUcontext context);

  // Emit stall reason map and replay cubin probes for late-attaching tracers.
  // Call periodically regardless of sampling state.
  void emitMetadata();

  void collectData(CUcontext context);
  void collectAllData();
  void finalize(CUcontext context);
  void loadModule(const char *cubin, size_t cubinSize);
  void unloadModule(const char *cubin, size_t cubinSize);

private:
  ConfigureData *getConfigureData(uint32_t contextId);
  CubinData *getCubinData(uint64_t cubinCrc);
  void processPCSamplingData(ConfigureData *configureData);

  proton::ThreadSafeMap<uint32_t, ConfigureData> contextIdToConfigureData;
  // CUPTI context ids, recorded by initialize(), for start() and prepare():
  // on a GB10 (driver 580), cuptiGetContextId from a launch callback that
  // isn't followed by a cuptiPCSamplingStart corrupts the heap when another
  // process on the GPU has PC sampling enabled. Those run on every launch
  // that may open a window, mostly without starting one.
  proton::ThreadSafeMap<CUcontext, uint32_t> contextIds;
  bool cachedContextId(CUcontext context, uint32_t *contextId);
  proton::ThreadSafeMap<size_t, std::pair<CubinData, size_t>>
      cubinCrcToCubinData;
  proton::ThreadSafeSet<uint32_t> contextInitialized;
  proton::ThreadSafeSet<uint32_t> contextFailed; // contexts where enable failed

  // Plain vector of initialized context IDs for iteration in collectAllData.
  // Protected by contextMutex.
  std::vector<uint32_t> initializedContextIds;

  // Contexts currently sampling (between a successful start and its stop).
  // Protected by pcSamplingMutex; numSampling mirrors its size so the
  // per-callback isSampling() check skips the lock when nothing samples.
  std::set<CUcontext> samplingContexts;
  // When each context's last window ended, for start()'s gap, and when its
  // current one started. Protected by pcSamplingMutex.
  std::map<CUcontext, uint64_t> lastStopNs;
  std::map<CUcontext, uint64_t> windowStartNs;
  // Serializes draining a context (collectData, finalize) across threads:
  // any thread may drain a window, and the release thread may stop one.
  // Recursive: collectData's recovery calls finalize. Lock order:
  // drainMutex, contextMutex, pcSamplingMutex.
  std::recursive_mutex drainMutex;
  std::atomic<int> numSampling{0};
  // Contexts whose GPU sampler is owned by another process. CUPTI reports
  // success from enable/start there and fails only GetData/Stop with
  // CUPTI_ERROR_UNKNOWN; we stop and never restart those contexts.
  proton::ThreadSafeSet<uint32_t> contextUnavailable;
  // Configuration gen each context was last re-armed for (prepare()).
  proton::ThreadSafeMap<uint32_t, uint64_t> contextRearmGen;
  std::mutex reconfigMutex{}; // one prepare() re-configures at a time
  std::mutex pcSamplingMutex{};
  std::mutex contextMutex{};

  // Contiguous stall reason map for USDT probe emission.
  StallReasonMap stallReasonMap;

  // Lightweight cubin metadata for replaying cubin_loaded probes to
  // late-attaching tracers. Protected by contextMutex.
  struct CubinRef {
    uint64_t crc;
    const char *data;
    size_t size;
    uint64_t lastEmittedNs; // monotonic; for emitMetadata staleness check
  };
  std::vector<CubinRef> loadedCubins;

  // Per-device GPU configuration, replayed alongside cubins so the agent can
  // convert PC sample counts to nanoseconds (needs sampling factor, GPU clock).
  // Protected by contextMutex.
  struct GpuConfig {
    uint32_t dev;
    uint32_t samplingFactor;
    uint32_t clockKHz;
    uint32_t smCount;
    uint64_t lastEmittedNs;
  };
  std::vector<GpuConfig> loadedConfigs;

  // emitMetadata tracking: per-probe USDT semaphore count from the last
  // call. Re-emit fires when the current count exceeds this (a new
  // consumer joined, even if another was already attached). A periodic
  // refresh bounds staleness from ABA cycles that net the same count
  // between our reads.
  std::atomic<uint16_t> prevStallSem{0};
  std::atomic<uint16_t> prevCubinSem{0};
  std::atomic<uint16_t> prevConfigSem{0};
  std::atomic<uint64_t> lastRefreshNs{0};
};

// Fire the error USDT probe. Callable from any translation unit.
void fireError(int32_t code, const char *message, const char *component);

} // namespace parcagpu

#endif // PARCAGPU_PC_SAMPLING_H_
