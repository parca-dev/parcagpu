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
#include "correlation_filter.h"
#include "env_config.h"
#include "pc_sampling.h"
#include "token_bucket.h"

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

// Thread-local rate limiter for callback probes (default 100/sec,
// configurable via PARCAGPU_RATE_LIMIT).
thread_local TokenBucket callbackLimiter(100.0);

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

// 50 ms — short enough to give ~20 dice rolls/sec (low rate variance, fast
// response to workload phase changes), long enough that CUPTI start/stop
// cost (~25 us measured) is amortized to <0.05% of wall time.
static constexpr uint64_t kPCSamplingIntervalNs = 50'000'000ULL;
// How often the controller recalibrates probability based on observed rate.
static constexpr uint64_t kPCControlPeriodNs = 5'000'000'000ULL;
// Don't react to <25% rate error (avoids oscillating on noise).
static constexpr double kPCControlTolerance = 0.25;
// Single update can change probability by at most this factor up or down.
static constexpr double kPCControlStepClamp = 2.0;
static constexpr double kPCProbMin = 0.001;
static constexpr double kPCProbMax = 1.0;
// Initial guess; controller converges within 1-2 control periods.
static constexpr double kPCInitialProbability = 0.05;
// Default target rate when PARCAGPU_PC_SAMPLING_RATE is unset.
static constexpr double kPCDefaultTargetRate = 100.0;

struct PCRateController {
  double targetRate = kPCDefaultTargetRate;       // immutable after init
  std::atomic<double> probability{kPCInitialProbability};
  std::atomic<uint64_t> samplesTotal{0};
  std::atomic<uint64_t> lastCheckNs{0};
  std::atomic<uint64_t> lastCheckTotal{0};
};
static PCRateController g_pcController;

// Per-thread sampling state.
struct PCSamplingState {
  bool active = false;        // Currently sampling
  uint64_t windowStartNs = 0; // When the current window opened
  uint64_t lastCheckNs = 0;   // Last time we rolled the dice
  unsigned int rngSeed = 0;   // Thread-local RNG state
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
  ratio = std::clamp(ratio, 1.0 / kPCControlStepClamp, kPCControlStepClamp);
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
    const char *rateEnv = getenv("PARCAGPU_RATE_LIMIT");
    if (rateEnv != nullptr) {
      double rate = atof(rateEnv);
      if (rate > 0) {
        callbackLimiter.setRate(rate);
      }
    }

    const char *targetRateEnv = getenv("PARCAGPU_PC_SAMPLING_RATE");
    if (targetRateEnv) {
      double r = atof(targetRateEnv);
      if (r > 0.0)
        g_pcController.targetRate = r;
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

    // Check if PC sampling is supported
    pcSamplingEnabled = parcagpu::PCSampling::isSupported();
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

    // Enable resource callbacks only if PC sampling is enabled
    if (pcSamplingEnabled) {
      proton::setResourceCallbacks(subscriber, /*enable=*/true);
    }

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

    DEBUG_PRINTF("[PARCAGPU] Successfully initialized CUPTI callbacks\n");
    return true;
  }

  void cleanup() {
    if (!initialized.exchange(false)) {
      return; // Already cleaned up
    }

    DEBUG_PRINTF("[PARCAGPU] Cleanup started\n");

    // PC sampling data is drained in finalize() during CONTEXT_DESTROY_STARTING
    // when the CUDA context is still valid. By the time cleanup() runs, the
    // context may already be dead, so we don't drain here.

    // Disable all callbacks
    if (subscriber) {
      proton::setRuntimeCallbacks(subscriber, /*enable=*/false);
      proton::setLaunchCallbacks(subscriber, /*enable=*/false);
      if (pcSamplingEnabled) {
        proton::setResourceCallbacks(subscriber, /*enable=*/false);
      }
    }

    // Flush any remaining activity records
    proton::cupti::activityFlushAll<true>(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    // Disable activity recording
    proton::cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    // Unsubscribe
    if (subscriber) {
      proton::cupti::unsubscribe<true>(subscriber);
      subscriber = nullptr;
    }

    DEBUG_PRINTF("[PARCAGPU] Cleanup completed\n");
  }

private:
  std::atomic<bool> initialized{false};
  bool pcSamplingEnabled = false;
  CUpti_SubscriberHandle subscriber = nullptr;

  // PC sampling state — owned by this profiler, destroyed with it.
  parcagpu::PCSampling pcSampling;

  // Outstanding event counter for flushing
  size_t outstandingEvents = 0;

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
    auto &profiler = CuptiProfiler::instance();

    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
      // Handle resource callbacks for PC sampling (only if enabled)
      if (!profiler.pcSamplingEnabled) {
        return;
      }

      const CUpti_ResourceData *resData =
          static_cast<const CUpti_ResourceData *>(cbdata_void);

      switch (cbid) {
      case CUPTI_CBID_RESOURCE_MODULE_LOADED: {
        const CUpti_ModuleResourceData *modData =
            static_cast<const CUpti_ModuleResourceData *>(
                resData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
          DEBUG_PRINTF("[PARCAGPU] Module loaded: cubin=%p size=%zu\n",
                       modData->pCubin, modData->cubinSize);
          profiler.pcSampling.loadModule(modData->pCubin, modData->cubinSize);
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
        profiler.pcSampling.initialize(ctx);
        break;
      }
      case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING: {
        CUcontext ctx = resData->context;
        DEBUG_PRINTF("[PARCAGPU] Context destroying: %p\n", ctx);
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
      uint32_t correlationId = cbdata->correlationId;

      // PC sampling windows are aligned to kernel-launch boundaries:
      // start on a launch ENTER, stop on a launch EXIT. The decision to
      // open a window is still time + probability gated; the launch CBIDs
      // just determine when the boundaries fire.
      const bool isKernelLaunchCb =
          domain == CUPTI_CB_DOMAIN_DRIVER_API && proton::isLaunch(cbid);

      // ENTER: open a sampling window on launch boundary if interval+prob hits.
      if (cbdata->callbackSite == CUPTI_API_ENTER) {
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
          runtimeEnterCorrelationId = correlationId;

        if (profiler.pcSamplingEnabled) {
          if (isKernelLaunchCb) {
            auto &st = g_pcSamplingState;
            uint64_t now = nowNs();

            if (!st.active &&
                (now - st.lastCheckNs >= kPCSamplingIntervalNs)) {
              st.lastCheckNs = now;
              const double p = g_pcController.probability.load(
                  std::memory_order_relaxed);
              if (threadRandom(st) < p) {
                st.active = true;
                st.windowStartNs = now;
                profiler.pcSampling.start(cbdata->context);
              }
            }
          }

          profiler.pcSampling.emitMetadata();
        }
        return;
      }

      // Process on EXIT to avoid adding latency to GPU launch
      if (cbdata->callbackSite != CUPTI_API_EXIT) {
        return;
      }

      // EXIT: while a sampling window is open, drain CUPTI's host staging
      // buffer on every CUDA API EXIT — not just launch EXITs. Empty drains
      // are cheap (single API call returning 0 PCs) and missed drains lose
      // samples (CUPTI_ERROR_OUT_OF_MEMORY when staging fills). Window close
      // is still kernel-launch aligned: only check elapsed-time on a launch
      // EXIT, so the window starts and ends at kernel boundaries.
      if (profiler.pcSamplingEnabled) {
        auto &st = g_pcSamplingState;

        if (st.active) {
          if (isKernelLaunchCb &&
              (nowNs() - st.windowStartNs >= kPCSamplingIntervalNs)) {
            profiler.pcSampling.stop(cbdata->context);
            st.active = false;
            controllerMaybeUpdate();
          } else {
            profiler.pcSampling.collectData(cbdata->context);
          }
        }

        profiler.pcSampling.emitMetadata();
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

      // Check if this is a graph launch (never rate limit these)
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

      // Rate limit probes using token bucket.  Skip rate limiting for graph
      // launches (they share one correlation ID across many kernels) and when
      // PC sampling is active (every kernel needs its correlation callback so
      // PC samples can be matched with CPU stacks on the agent side).
      if (!isGraphLaunch && !g_pcSamplingState.active) {
        if (!callbackLimiter.tryAcquire()) {
          DEBUG_PRINTF(
              "[PARCAGPU] Rate limited: skipping probe for correlationId=%u\n",
              correlationId);
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
        proton::cupti::activityFlushAll<true>(0);
        profiler.outstandingEvents = 0;
      }
    }
  }
};

} // namespace parcagpu

// CUPTI initialization function required for CUDA_INJECTION64_PATH
extern "C" int InitializeInjection(void) {
  DEBUG_PRINTF("[PARCAGPU] InitializeInjection called\n");

  auto &profiler = parcagpu::CuptiProfiler::instance();
  if (!profiler.initialize()) {
    return 0; // Return 0 on failure, but don't break injection
  }

  // Register cleanup at exit
  atexit([]() { parcagpu::CuptiProfiler::instance().cleanup(); });

  return 1; // Success
}
