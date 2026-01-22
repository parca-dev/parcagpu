#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/sdt.h>
#include <time.h>
#include <unistd.h>

// Include proton headers
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiCallbacks.h"
#include "Utility/Singleton.h"
#include "correlation_filter.h"
#include "parcagpu_pc_sampling.h"

#include <cupti.h>

namespace parcagpu {

// Debug logging control
bool debug_enabled = false;
bool limiter_disabled = false;

// Global correlation tracking instances
static CorrelationFilter g_correlationFilter;
static GraphCorrelationMap g_graphCorrelationMap;
static std::atomic<uint32_t> g_bufferCycle{0};

// Thread-local tracking: store correlation ID from runtime ENTER
// so we can skip driver EXIT probe when it matches (driver calls happen under runtime calls)
thread_local uint32_t runtimeEnterCorrelationId = 0;

// Thread-local rate limiting
thread_local uint64_t lastProbeTimeNs = 0;

void init_debug() {
  static bool initialized = false;
  if (!initialized) {
    debug_enabled = getenv("PARCAGPU_DEBUG") != nullptr;
    limiter_disabled = getenv("PARCAGPU_LIMITER_DISABLE") != nullptr;
    initialized = true;
  }
}

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
      DEBUG_PRINTF("[PARCAGPU] PC sampling enabled (continuous mode)\n");
    } else {
      DEBUG_PRINTF("[PARCAGPU] PC sampling disabled, using kernel activity only\n");
    }

    // Subscribe to callbacks
    auto result = proton::cupti::subscribe<true>(&subscriber, callbackHandler,
                                            nullptr);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] Failed to subscribe to callbacks: error %d\n", result);
      return false;
    }

    // Enable runtime and driver API callbacks (using Proton's utilities)
    proton::setRuntimeCallbacks(subscriber, /*enable=*/true);
    proton::setDriverCallbacks(subscriber, /*enable=*/true);

    // Enable resource callbacks only if PC sampling is enabled
    if (pcSamplingEnabled) {
      proton::setResourceCallbacks(subscriber, /*enable=*/true);
    }

    // Register activity buffer callbacks (using Proton's pattern)
    result = proton::cupti::activityRegisterCallbacks<true>(allocBuffer,
                                                            completeBuffer);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] Failed to register activity callbacks: error %d\n", result);
      return false;
    }

    // Enable kernel activity recording
    result = proton::cupti::activityEnable<true>(
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[PARCAGPU] Failed to enable concurrent kernel activity: error %d\n", result);
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

    // Flush any remaining activity records
    proton::cupti::activityFlushAll<true>(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    // Disable activity recording
    proton::cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

    // Disable all callbacks (using Proton's utilities)
    if (subscriber) {
      proton::setRuntimeCallbacks(subscriber, /*enable=*/false);
      proton::setDriverCallbacks(subscriber, /*enable=*/false);
      if (pcSamplingEnabled) {
        proton::setResourceCallbacks(subscriber, /*enable=*/false);
      }
      proton::cupti::unsubscribe<true>(subscriber);
      subscriber = nullptr;
    }

    DEBUG_PRINTF("[PARCAGPU] Cleanup completed\n");
  }

private:
  std::atomic<bool> initialized{false};
  bool pcSamplingEnabled = false;
  CUpti_SubscriberHandle subscriber = nullptr;

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

  static void completeBuffer(CUcontext ctx, uint32_t streamId,
                              uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    int recordCount = 0;
    int filteredCount = 0;

    DEBUG_PRINTF(
        "[PARCAGPU] completeBuffer called: buffer=%p validSize=%zu\n",
        buffer, validSize);

    // Start a new buffer cycle for graph correlation tracking
    uint32_t cycle = g_bufferCycle.fetch_add(1);
    g_graphCorrelationMap.cycle_start(cycle);

    while (true) {
      CUptiResult result =
          proton::cupti::activityGetNextRecord<false>(buffer, validSize, &record);
      if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else if (result != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[PARCAGPU] Error reading activity record: error %d\n", result);
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
          shouldEmit = g_graphCorrelationMap.check_and_mark_seen(k->correlationId, cycle);
        } else {
          // Regular kernel - check and remove from correlation filter
          shouldEmit = g_correlationFilter.check_and_remove(k->correlationId);
        }

        if (!shouldEmit) {
          filteredCount++;
          DEBUG_PRINTF("[PARCAGPU] Filtered kernel activity: correlationId=%u "
                       "graphId=%u (not in filter)\n",
                       k->correlationId, k->graphId);
          break;
        }

        DEBUG_PRINTF("[PARCAGPU] Kernel activity: graphId=%u graphNodeId=%lu "
                     "name=%s, correlationId=%u, deviceId=%u, "
                     "streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
                     k->graphId, k->graphNodeId, k->name, k->correlationId,
                     k->deviceId, k->streamId, k->start, k->end,
                     k->end - k->start);

        // Emit USDT probe for kernel execution
        STAP_PROBE8(parcagpu, kernel_executed, k->start, k->end,
                    k->correlationId, k->deviceId, k->streamId, k->graphId,
                    k->graphNodeId, k->name);
        break;
      }
      default:
        DEBUG_PRINTF("[PARCAGPU] Activity record %d: kind=%d\n", recordCount,
                     record->kind);
        break;
      }
    }

    // End cycle - cleanup completed graph entries
    g_graphCorrelationMap.cycle_end();

    DEBUG_PRINTF("[PARCAGPU] Processed %d activity records (%d filtered) from buffer %p\n",
                 recordCount, filteredCount, buffer);

    // Free the buffer (Proton's pattern)
    std::free(buffer);
  }

  static void callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid,
                              const void *cbdata_void) {
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
            static_cast<const CUpti_ModuleResourceData *>(resData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
          DEBUG_PRINTF("[PARCAGPU] Module loaded: cubin=%p size=%zu\n",
                       modData->pCubin, modData->cubinSize);
          parcagpu::PCSampling::instance().loadModule(
              modData->pCubin, modData->cubinSize);
        }
        break;
      }
      case CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING: {
        const CUpti_ModuleResourceData *modData =
            static_cast<const CUpti_ModuleResourceData *>(resData->resourceDescriptor);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
          DEBUG_PRINTF("[PARCAGPU] Module unloading: cubin=%p size=%zu\n",
                       modData->pCubin, modData->cubinSize);
          parcagpu::PCSampling::instance().unloadModule(
              modData->pCubin, modData->cubinSize);
        }
        break;
      }
      case CUPTI_CBID_RESOURCE_CONTEXT_CREATED: {
        CUcontext ctx = resData->context;
        DEBUG_PRINTF("[PARCAGPU] Context created: %p\n", ctx);
        parcagpu::PCSampling::instance().initialize(ctx);
        break;
      }
      case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING: {
        CUcontext ctx = resData->context;
        DEBUG_PRINTF("[PARCAGPU] Context destroying: %p\n", ctx);
        parcagpu::PCSampling::instance().finalize(ctx);
        break;
      }
      default:
        break;
      }
    } else {
      // Handle both Runtime and Driver API callbacks
      // Track runtime ENTER so we can skip driver EXIT when they match
      const CUpti_CallbackData *cbdata =
          static_cast<const CUpti_CallbackData *>(cbdata_void);
      uint32_t correlationId = cbdata->correlationId;

      if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
          cbdata->callbackSite == CUPTI_API_ENTER) {
        runtimeEnterCorrelationId = correlationId;
        return;
      }

      // Process on EXIT to avoid adding latency to GPU launch
      if (cbdata->callbackSite != CUPTI_API_EXIT) {
        return;
      }

      const char *name = cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
      int signedCbid;

      if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        // Skip if this driver call is under a runtime call (same correlation ID)
        if (correlationId == runtimeEnterCorrelationId) {
          DEBUG_PRINTF("[PARCAGPU] Skipping driver EXIT correlationId=%u - runtime will handle\n",
                       correlationId);
          return;
        }
        // Pure driver call (no runtime wrapper) - use negative cbid
        signedCbid = -(int)cbid;
        DEBUG_PRINTF("[PARCAGPU] Driver API callback: cbid=%d, correlationId=%u, func=%s\n",
                     cbid, correlationId, name);
      } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        signedCbid = (int)cbid;
        runtimeEnterCorrelationId = 0; // Clear after use
        DEBUG_PRINTF("[PARCAGPU] Runtime API callback: cbid=%d, correlationId=%u, func=%s\n",
                     cbid, correlationId, name);
      } else {
        return;
      }

      // Check if this is a graph launch (never rate limit these)
      bool isGraphLaunch = false;
      if (signedCbid < 0) {
        // Driver API: cuGraphLaunch = 514, cuGraphLaunch_ptsz = 515
        int driverCbid = -signedCbid;
        isGraphLaunch = (driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
                         driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
      } else {
        // Runtime API: cudaGraphLaunch = 311, cudaGraphLaunch_ptsz = 312
        isGraphLaunch = (signedCbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
                         signedCbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
      }

      // Rate limit probes (skip for graph launches)
      if (!limiter_disabled && !isGraphLaunch) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t nowNs = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
        constexpr uint64_t PROBE_MIN_INTERVAL_NS = 500000; // 500μs
        if (nowNs - lastProbeTimeNs < PROBE_MIN_INTERVAL_NS) {
          DEBUG_PRINTF("[PARCAGPU] Rate limited: skipping probe for correlationId=%u\n",
                       correlationId);
          return;
        }
        lastProbeTimeNs = nowNs;
      }

      profiler.outstandingEvents++;
      // Emit USDT probe with signed cbid (negative for driver, positive for runtime)
      STAP_PROBE3(parcagpu, cuda_correlation, correlationId, signedCbid, name);

      // Insert into correlation filter so we can match kernel activities later
      if (isGraphLaunch) {
        g_graphCorrelationMap.insert(correlationId);
        DEBUG_PRINTF("[PARCAGPU] Inserted correlationId=%u into graph map\n",
                     correlationId);
      } else {
        g_correlationFilter.insert(correlationId);
        DEBUG_PRINTF("[PARCAGPU] Inserted correlationId=%u into correlation filter\n",
                     correlationId);
      }

      // Flush if too many events pile up
      if (profiler.outstandingEvents > 3000) {
        DEBUG_PRINTF("[PARCAGPU] Flushing: outstandingEvents=%zu\n",
                     profiler.outstandingEvents);
        proton::cupti::activityFlushAll<true>(0);
        profiler.outstandingEvents = 0;
      }

      // Collect PC sampling data after kernel launch (continuous mode)
      if (profiler.pcSamplingEnabled) {
        parcagpu::PCSampling::instance().collectData(cbdata->context,
                                                   correlationId);
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
