// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PC_SAMPLING_H_
#define PC_SAMPLING_H_

#include <atomic>
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

  void initialize(CUcontext context);

  CUpti_PCSamplingConfigurationInfo configureStallReasons();
  CUpti_PCSamplingConfigurationInfo configureSamplingPeriod();
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
  uint32_t numStallReasons{};
  uint32_t numValidStallReasons{};
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

  // Check if PC sampling is supported (CUPTI >= 12.8.1).
  // Enabled by default; set PARCAGPU_SAMPLING_FACTOR=0 to disable.
  static bool isSupported();

  void initialize(CUcontext context);

  // Start PC sampling — kernels become serialized until stop().
  // No-op if already started. Thread-safe.
  void start(CUcontext context);

  // Stop PC sampling, drain accumulated data, and emit probes.
  // Kernels resume concurrent execution. No-op if not started.
  void stop(CUcontext context);

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
  proton::ThreadSafeMap<size_t, std::pair<CubinData, size_t>>
      cubinCrcToCubinData;
  proton::ThreadSafeSet<uint32_t> contextInitialized;
  proton::ThreadSafeSet<uint32_t> contextFailed; // contexts where enable failed

  // Plain vector of initialized context IDs for iteration in collectAllData.
  // Protected by contextMutex.
  std::vector<uint32_t> initializedContextIds;

  // Tracks whether CUPTI PC sampling is currently active (start/stop).
  // Only one context can be sampling at a time in KERNEL_SERIALIZED mode.
  std::atomic<bool> samplingActive{false};
  CUcontext samplingContext{};
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
