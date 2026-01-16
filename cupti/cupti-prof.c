#define _POSIX_C_SOURCE 199309L
#include <execinfo.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sdt.h>
#include <time.h>
#include <unistd.h>

#include <cupti.h>

// Debug logging control
static bool debug_enabled = false;

// Activity buffer management
// A kernel activity is around 224 bytes so a 128kb buffer
// will hold ~500 activities, we want to flush regularly since
// we are a continuous profiler so we don't need a huge buffer
// like most CUPTI profilers.  Also a small size avoid malloc
// just going to mmap every time so the allocator should cache
// and re-use these for us.
static size_t activityBufferSize = 128 * 1024;

// Global variables
static CUpti_SubscriberHandle subscriber = 0;

static size_t outstandingEvents = 0;

// Thread-local tracking: store correlation ID from runtime ENTER
// so we can skip driver EXIT probe when it matches (driver calls happen under runtime calls)
static __thread uint32_t runtimeEnterCorrelationId = 0;

// Rate limiting: no more than 1 probe per 500μs per thread
static __thread uint64_t lastProbeTimeNs = 0;
#define PROBE_MIN_INTERVAL_NS 500000 // 500μs

static void init_debug(void) {
  static bool initialized = false;
  if (!initialized) {
    debug_enabled = getenv("PARCAGPU_DEBUG") != NULL;
    initialized = true;
  }
}

#define DEBUG_PRINTF(...)                                                      \
  do {                                                                         \
    init_debug();                                                              \
    if (debug_enabled) {                                                       \
      struct timespec ts;                                                      \
      clock_gettime(CLOCK_REALTIME, &ts);                                      \
      printf("[%ld.%09ld] ", ts.tv_sec, ts.tv_nsec);                           \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

// Forward declarations
static void parcagpuCuptiCallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata);
static void parcagpuBufferRequested(uint8_t **buffer, size_t *size,
                                    size_t *maxNumRecords);
static void parcagpuBufferCompleted(CUcontext ctx, uint32_t streamId,
                                    uint8_t *buffer, size_t size,
                                    size_t validSize);

void cleanup(void);

// CUPTI initialization function required for CUDA_INJECTION64_PATH
int InitializeInjection(void) {
  DEBUG_PRINTF("[CUPTI] InitializeInjection called\n");
  CUptiResult result;

  // Set flush period BEFORE enabling activities (in milliseconds)
  // Try a larger value like 1000ms (1 second) for better compatibility
  result = cuptiActivityFlushPeriod(1000);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to set activity flush period: %s\n",
            errstr);
  } else {
    DEBUG_PRINTF("[CUPTI] Set activity flush period to 1000ms\n");
  }

  // Try to subscribe to callbacks
  result = cuptiSubscribe(&subscriber,
                          (CUpti_CallbackFunc)parcagpuCuptiCallback, NULL);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to subscribe to callbacks: %s\n", errstr);
    return 1; // Still return success to not break the injection
  }

  // Enable all runtime API kernel launch callbacks
  CUpti_CallbackId runtimeCallbacks[] = {
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000,
      CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,
      CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000,
  };
  for (size_t i = 0; i < sizeof(runtimeCallbacks) / sizeof(runtimeCallbacks[0]);
       i++) {
    result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 runtimeCallbacks[i]);
    if (result != CUPTI_SUCCESS) {
      const char *errstr;
      cuptiGetResultString(result, &errstr);
      fprintf(stderr, "[CUPTI] Failed to enable runtime callback %d: %s\n",
              runtimeCallbacks[i], errstr);
    }
  }

  // Enable all driver API kernel launch callbacks
  CUpti_CallbackId driverCallbacks[] = {
      CUPTI_DRIVER_TRACE_CBID_cuLaunch,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice,
      CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,
      CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
  };
  for (size_t i = 0; i < sizeof(driverCallbacks) / sizeof(driverCallbacks[0]);
       i++) {
    result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                 driverCallbacks[i]);
    if (result != CUPTI_SUCCESS) {
      const char *errstr;
      cuptiGetResultString(result, &errstr);
      fprintf(stderr, "[CUPTI] Failed to enable driver callback %d: %s\n",
              driverCallbacks[i], errstr);
    }
  }

  // Register activity buffer callbacks
  result = cuptiActivityRegisterCallbacks(parcagpuBufferRequested,
                                          parcagpuBufferCompleted);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to register activity callbacks: %s\n",
            errstr);
    return 1; // Still return success to not break the injection
  }

  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable concurrent kernel activity: %s\n",
            errstr);
  } else {
    DEBUG_PRINTF("[CUPTI] Enabled CONCURRENT_KERNEL activity\n");
  }

  atexit(cleanup);

  DEBUG_PRINTF("[CUPTI] Successfully initialized CUPTI callbacks with external "
               "correlation and activity API\n");

  return 1;
}

// Helper function to print stack trace
static void print_backtrace(const char *prefix) {
  void *array[20];
  size_t size;
  char **strings;

  size = backtrace(array, 20);
  strings = backtrace_symbols(array, size);

  if (strings != NULL) {
    printf("%s Stack trace (%zu frames):\n", prefix, size);
    for (size_t i = 0; i < size; i++) {
      printf("  [%zu] %s\n", i, strings[i]);
    }
    free(strings);
  }
}

// Callback handler for driver and runtime API
static void parcagpuCuptiCallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata) {
  uint32_t correlationId = cbdata->correlationId;

  // Track runtime ENTER so we can skip driver EXIT when they match
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
      cbdata->callbackSite == CUPTI_API_ENTER) {
    runtimeEnterCorrelationId = correlationId;
    return;
  }

  // We hook on EXIT because that makes our probe overhead not add to GPU
  // launch latency and hopefully covers some of the overhead in the shadow of
  // GPU async work.
  if (cbdata->callbackSite != CUPTI_API_EXIT) {
    return;
  }

  const char *name =
      cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
  int signedCbid;

  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    // Skip if this driver call is under a runtime call (same correlation ID)
    if (correlationId == runtimeEnterCorrelationId) {
      DEBUG_PRINTF(
          "[CUPTI] Skipping driver EXIT correlationId=%u - runtime will handle\n",
          correlationId);
      return;
    }
    // Pure driver call (no runtime wrapper) - use negative cbid
    signedCbid = -(int)cbid;
    DEBUG_PRINTF(
        "[CUPTI] Driver API callback: cbid=%d, correlationId=%u, func=%s\n",
        cbid, correlationId, name);
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    signedCbid = (int)cbid;
    runtimeEnterCorrelationId = 0; // Clear after use
    DEBUG_PRINTF(
        "[CUPTI] Runtime API callback: cbid=%d, correlationId=%u, func=%s\n",
        cbid, correlationId, name);
  } else {
    return;
  }

  // Rate limit: no more than 1 probe per ms per thread
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t nowNs = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  if (nowNs - lastProbeTimeNs < PROBE_MIN_INTERVAL_NS) {
    DEBUG_PRINTF("[CUPTI] Rate limited: skipping probe for correlationId=%u\n",
                 correlationId);
    return;
  }
  lastProbeTimeNs = nowNs;

  outstandingEvents++;
  DTRACE_PROBE3(parcagpu, cuda_correlation, correlationId, signedCbid, name);
  // If we let too many events pile up it overwhelms the perf_event buffers,
  // just another reason to explore just passing the activity buffer through to
  // eBPF.
  if (outstandingEvents > 3000) {
    DEBUG_PRINTF("[CUPTI] Flushing: outstandingEvents=%zu\n",
                 outstandingEvents);
    cuptiActivityFlushAll(0);
    outstandingEvents = 0;
  }
}

// Buffer request callback
static void parcagpuBufferRequested(uint8_t **buffer, size_t *size,
                                    size_t *maxNumRecords) {
  *buffer = (uint8_t *)aligned_alloc(8, activityBufferSize);
  *size = activityBufferSize;
  *maxNumRecords = 0; // Let CUPTI decide

  DEBUG_PRINTF("[CUPTI:bufferRequested] Allocated buffer %p, size=%zu\n",
               *buffer, *size);
}

// Buffer completion callback
static void parcagpuBufferCompleted(CUcontext ctx, uint32_t streamId,
                                    uint8_t *buffer, size_t size,
                                    size_t validSize) {
  CUptiResult result;
  CUpti_Activity *record = NULL;
  int recordCount = 0;
  static int calls = 0;

  DEBUG_PRINTF("[CUPTI] bufferCompleted called: buffer=%p validSize=%zu (%d)\n",
               buffer, validSize, calls++);

  while (1) {
    result = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else if (result != CUPTI_SUCCESS) {
      const char *errstr;
      cuptiGetResultString(result, &errstr);
      fprintf(stderr, "[CUPTI] Error reading activity record: %s\n", errstr);
      break;
    }

    recordCount++;
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      CUpti_ActivityKernel5 *k = (CUpti_ActivityKernel5 *)record;

      DEBUG_PRINTF("[CUPTI] Kernel activity: graphId=%u graphNodeId=%lu "
                   "name=%s, correlationId=%u, deviceId=%u, "
                   "streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
                   k->graphId, k->graphNodeId, k->name, k->correlationId,
                   k->deviceId, k->streamId, k->start, k->end,
                   k->end - k->start);
      DTRACE_PROBE8(parcagpu, kernel_executed, k->start, k->end,
                    k->correlationId, k->deviceId, k->streamId, k->graphId,
                    k->graphNodeId, k->name);
    }
  }

  DEBUG_PRINTF("[CUPTI] Processed %d activity records from buffer %p\n",
               recordCount, buffer);

  // Reset to 0 rather than decrement - one API callback can produce N
  // activities so decrementing by recordCount can cause underflow (size_t wraps
  // to huge value)
  outstandingEvents = 0;

  // Free the buffer
  DEBUG_PRINTF("[CUPTI] Freeing buffer %p\n", buffer);
  free(buffer);

  // Report any records dropped due to buffer overflow
  size_t dropped;
  result = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
  if (result == CUPTI_SUCCESS && dropped > 0) {
    fprintf(stderr, "[CUPTI] Warning: %zu activity records dropped\n", dropped);
  }
}

// Cleanup function (destructor disabled to prevent early cleanup)
void cleanup(void) {
  static bool cleanup_done = false;

  // Make cleanup idempotent - safe to call multiple times
  if (cleanup_done) {
    return;
  }
  cleanup_done = true;

  DEBUG_PRINTF("[CUPTI] Cleanup started\n");
  // Flush any remaining activity records
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

  // Unsubscribe from callbacks
  if (subscriber) {
    cuptiUnsubscribe(subscriber);
    subscriber = 0;
  }

  DEBUG_PRINTF("[CUPTI] Cleanup completed\n");
}
