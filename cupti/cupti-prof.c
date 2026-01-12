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
static size_t activityBufferSize = 10 * 1024 * 1024;

// Global variables
static CUpti_SubscriberHandle subscriber = 0;

static size_t outstandingEvents = 0;

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
static void bufferRequested(uint8_t **buffer, size_t *size,
                            size_t *maxNumRecords);
static void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                            size_t size, size_t validSize);

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

  // Try enabling driver API kernel launch callback like the example
  result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable cuLaunchKernel callback: %s\n",
            errstr);
  }

  // Enable runtime API callbacks for cudaLaunchKernel
  result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                               CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable cudaLaunchKernel callback: %s\n",
            errstr);
  }

  // Enable runtime API callbacks for cudaGraphLaunch
  result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                               CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable cudaGraphLaunch callback: %s\n",
            errstr);
  }

  // Enable runtime API callbacks for cudaGraphLaunch
  result =
      cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                          CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable cudaGraphLaunch callback: %s\n",
            errstr);
  }

  // Register activity buffer callbacks
  result = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to register activity callbacks: %s\n",
            errstr);
    return 1; // Still return success to not break the injection
  }

  // Enable multiple kernel activity recording types
  result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(result, &errstr);
    fprintf(stderr, "[CUPTI] Failed to enable concurrent kernel activity: %s\n",
            errstr);
  } else {
    DEBUG_PRINTF("[CUPTI] Enabled CONCURRENT_KERNEL activity\n");
  }

  // This activity kind serializes execution and gives me errors on a T4:
  // CUPTI_ERROR_NOT_COMPATIBLE But its not a fatal error so do it anyways
  //   result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  //   if (result != CUPTI_SUCCESS) {
  //     const char *errstr;
  //     cuptiGetResultString(result, &errstr);
  //     fprintf(stderr, "[CUPTI] Failed to enable kernel activity: %s\n",
  //     errstr);
  //   } else {
  //     DEBUG_PRINTF("[CUPTI] Enabled KERNEL activity\n");
  //   }

  // Also try enabling runtime activities
  //   result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  //   if (result != CUPTI_SUCCESS) {
  //     const char *errstr;
  //     cuptiGetResultString(result, &errstr);
  //     fprintf(stderr, "[CUPTI] Failed to enable runtime activity: %s\n",
  //     errstr);
  //   } else {
  //     DEBUG_PRINTF("[CUPTI] Enabled RUNTIME activity\n");
  //   }

  // Try enabling graph activities
  //   result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
  //   if (result != CUPTI_SUCCESS) {
  //     const char *errstr;
  //     cuptiGetResultString(result, &errstr);
  //     fprintf(stderr, "[CUPTI] Failed to enable graph trace activity: %s\n",
  //             errstr);
  //   } else {
  //     DEBUG_PRINTF("[CUPTI] Enabled GRAPH_TRACE activity\n");
  //   }

  atexit(cleanup);

  DEBUG_PRINTF("[CUPTI] Successfully initialized CUPTI callbacks with external "
               "correlation and activity API\n");

  // NOTE: If automatic flush still doesn't work, you can implement manual
  // periodic flushing:
  // 1. Create a background thread that calls cuptiActivityFlushAll(0)
  // periodically
  // 2. Or call cuptiActivityFlushAll(0) from your application at regular
  // intervals
  // 3. Or hook into CUDA synchronization points (cudaDeviceSynchronize, etc.)
  // to flush

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

// Callback handler for both runtime and driver API
static void parcagpuCuptiCallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata) {
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    // We hook on EXIT because that makes our probe overhead not add to GPU
    // launch latency and hopefully covers some of the overhead in the shadow of
    // GPU async work.
    if (cbdata->callbackSite == CUPTI_API_EXIT) {
      // Probablistic gate should go here.
      uint32_t correlationId = cbdata->correlationId;
      // Call stub functions for uprobe attachment
      const char *name = cbdata->functionName;
      switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
        if (cbdata->symbolName) {
          DEBUG_PRINTF("----------- %s\n", cbdata->symbolName);
          name = cbdata->symbolName;
        }
      case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000:
        DEBUG_PRINTF("[CUPTI] Runtime API callback: cbid=%d, correlationId=%u, "
                     "func=%s\n",
                     cbid, correlationId, cbdata->functionName);
        outstandingEvents++;
        DTRACE_PROBE3(parcagpu, cuda_correlation, correlationId, cbid, name);
        break;
      default:
        // Debug: print any other runtime API callback we see with backtrace
        DEBUG_PRINTF(
            "[CUPTI] Other Runtime API callback: cbid=%d, correlationId=%u\n",
            cbid, correlationId);
        // Print backtrace to see who's calling this
        if (debug_enabled) {
          print_backtrace("[CUPTI]");
        }
      }
    }
  }
  // If we let too many events pile up it overwhelms the perf_event buffers,
  // just another reason to explore just passing the activity buffer through to
  // eBPF.
  if (outstandingEvents > 3000) {
    DEBUG_PRINTF("[CUPTI] Flushing: outstandingEvents=%zu\n",
                 outstandingEvents);
    cuptiActivityFlushAll(0);
  }
}

// Buffer request callback
static void bufferRequested(uint8_t **buffer, size_t *size,
                            size_t *maxNumRecords) {
  // Allocate 64MB buffer aligned to 8 bytes
  *buffer = (uint8_t *)aligned_alloc(8, activityBufferSize);
  *size = activityBufferSize;
  *maxNumRecords = 0; // Let CUPTI decide

  DEBUG_PRINTF("[CUPTI:bufferRequested] Allocated buffer %p, size=%zu\n",
               *buffer, *size);
}

// Buffer completion callback
static void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                            size_t size, size_t validSize) {
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
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
      CUpti_ActivityAPI *r = (CUpti_ActivityAPI *)record;
      DEBUG_PRINTF("[CUPTI] Runtime activity: correlationId=%u, cbid=%d,\n",
                   r->correlationId, r->cbid);
      break;
    }
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    case CUPTI_ACTIVITY_KIND_KERNEL: {
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
      break;
    }
    // case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: {
    //   CUpti_ActivityGraphTrace *g = (CUpti_ActivityGraphTrace *)record;

    //   DEBUG_PRINTF(
    //       "[CUPTI] Graph activity: graphId=%u, correlationId=%u, deviceId=%u,
    //       " "streamId=%u, start=%lu, end=%lu, duration=%lu ns\n", g->graphId,
    //       g->correlationId, g->deviceId, g->streamId, g->start, g->end,
    //       g->end - g->start);
    //   // Call stub function for uprobe attachment
    //   uint64_t devCorrelationId =
    //       g->correlationId | ((uint64_t)g->deviceId << 32);
    //   DTRACE_PROBE5(parcagpu, graph_executed, g->start, g->end,
    //                 devCorrelationId, g->streamId, g->graphId);
    //   break;
    // }
    default:
      DEBUG_PRINTF("[CUPTI] Activity record %d: kind=%d\n", recordCount,
                   record->kind);
    }
  }

  DEBUG_PRINTF("[CUPTI] Processed %d activity records from buffer %p\n",
               recordCount, buffer);

  // Reset to 0 rather than decrement - one API callback can produce N
  // activities so decrementing by recordCount can cause underflow (size_t wraps
  // to huge value)
  outstandingEvents = 0;

  // Free the buffer
  DEBUG_PRINTF("[CUPTI:bufferCompleted] Freeing buffer %p\n", buffer);
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
