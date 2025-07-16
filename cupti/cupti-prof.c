#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <sys/sdt.h>

#include <cuda.h>
#include <cupti.h>
#include <nvperf_host.h>

// Debug logging control
static bool debug_enabled = false;

static void init_debug(void) {
    static bool initialized = false;
    if (!initialized) {
        debug_enabled = getenv("PARCAGPU_DEBUG") != NULL;
        initialized = true;
    }
}

#define DEBUG_PRINTF(...) do { \
    init_debug(); \
    if (debug_enabled) { \
        printf(__VA_ARGS__); \
    } \
} while (0)

#define CUPTI_CALL(call)                                                        \
do {                                                                            \
    CUptiResult _status = call;                                                 \
    if (_status != CUPTI_SUCCESS) {                                             \
        const char *errstr;                                                     \
        cuptiGetResultString(_status, &errstr);                                 \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                __FILE__, __LINE__, #call, errstr);                             \
        exit(EXIT_FAILURE);                                                      \
    }                                                                           \
} while (0)

// Forward declarations
static void runtimeApiCallback(void* userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const CUpti_CallbackData* cbdata);
static void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
static void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                           size_t size, size_t validSize);

// Global variables
static CUpti_SubscriberHandle subscriber = 0;

// Stub functions for uprobe attachment - API callbacks
// These are intentionally not static/inline to ensure they're available for uprobes
__attribute__((noinline)) void parcagpuLaunchKernel(uint32_t correlationId) {
    DEBUG_PRINTF("[CUPTI] parcagpuLaunchKernel: correlationId=%u\n", correlationId);
}

__attribute__((noinline)) void parcagpuGraphLaunch(uint32_t correlationId) {
    DEBUG_PRINTF("[CUPTI] parcagpuGraphLaunch: correlationId=%u\n", correlationId);
}

// Stub functions for uprobe attachment - Activity events
__attribute__((noinline)) void parcagpuKernelExecuted(uint32_t correlationId, uint64_t duration_ns) {
    DEBUG_PRINTF("[CUPTI] parcagpuKernelExecuted: correlationId=%u, duration=%lu ns\n", correlationId, duration_ns);
}

void cleanup(void);

// CUPTI initialization function required for CUDA_INJECTION64_PATH
int InitializeInjection(void) {
    DEBUG_PRINTF("[CUPTI] InitializeInjection called - CUPTI library loaded via CUDA_INJECTION64_PATH\n");

    // Try to subscribe to callbacks
    CUptiResult result = cuptiSubscribe(&subscriber,
                           (CUpti_CallbackFunc)runtimeApiCallback,
                           NULL);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to subscribe to callbacks: %s\n", errstr);
        return 1; // Still return success to not break the injection
    }

    // Try enabling driver API kernel launch callback like the example
    result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable cuLaunchKernel callback: %s\n", errstr);
    }

    // Enable runtime API callbacks for cudaLaunchKernel
    result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable cudaLaunchKernel callback: %s\n", errstr);
    }

    // Enable runtime API callbacks for cudaGraphLaunch
    result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable cudaGraphLaunch callback: %s\n", errstr);
    }

    // Register activity buffer callbacks
    result = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to register activity callbacks: %s\n", errstr);
        return 1; // Still return success to not break the injection
    }

    // Enable multiple kernel activity recording types
    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable concurrent kernel activity: %s\n", errstr);
    } else {
        DEBUG_PRINTF("[CUPTI] Enabled CONCURRENT_KERNEL activity\n");
    }

    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable kernel activity: %s\n", errstr);
    } else {
        DEBUG_PRINTF("[CUPTI] Enabled KERNEL activity\n");
    }

    // Also try enabling runtime activities
    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable runtime activity: %s\n", errstr);
    } else {
        DEBUG_PRINTF("[CUPTI] Enabled RUNTIME activity\n");
    }

    // Try enabling graph activities
    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "[CUPTI] Failed to enable graph trace activity: %s\n", errstr);
    } else {
        DEBUG_PRINTF("[CUPTI] Enabled GRAPH_TRACE activity\n");
    }

    atexit(cleanup);

    DEBUG_PRINTF("[CUPTI] Successfully initialized CUPTI callbacks with external correlation and activity API\n");
    return 1;
}

// Generate a correlation ID using atomic counter
static uint32_t setCorrelationId(uint32_t id) {
    if (id == 0) {
        static _Atomic uint32_t counter = 0;
        id =  __atomic_add_fetch(&counter, 1, __ATOMIC_SEQ_CST);
        // Push our own correlation ID for external correlation
        cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    }
}

// Callback handler for both runtime and driver API
static void runtimeApiCallback(void* userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const CUpti_CallbackData* cbdata) {
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        if (cbdata->callbackSite == CUPTI_API_ENTER) {
            uint32_t correlationId = cbdata->correlationId;
            // Call stub functions for uprobe attachment
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                // print all cbdata fields for debugging
                DEBUG_PRINTF("[CUPTI] Runtime API callback: correlationId=%u, functionName=%s, contextUid=%u, callbackSite=%u, symbolName=%s\n",
                    correlationId, cbdata->functionName, cbdata->contextUid, cbdata->callbackSite,cbdata->symbolName);
                correlationId = setCorrelationId(correlationId);
                DTRACE_PROBE1(parcagpu, launch_kernel, correlationId);
                parcagpuLaunchKernel(correlationId);
                // Force flush to ensure activities are processed
                cuptiActivityFlushAll(0);
            } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000) {
                correlationId = setCorrelationId(correlationId);
                DTRACE_PROBE1(parcagpu, launch_graph, correlationId);
                parcagpuGraphLaunch(correlationId);
                // Force flush to ensure activities are processed
                cuptiActivityFlushAll(0);
            } else {
                // Debug: print any other runtime API callback we see
                DEBUG_PRINTF("[CUPTI] Runtime API callback: cbid=%d, correlationId=%u\n", cbid, correlationId);
            }
        }
    }
}

// Activity buffer management
static uint8_t* activityBuffer = NULL;
static size_t activityBufferSize = 65536;

// Buffer request callback
static void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    if (activityBuffer == NULL) {
        activityBuffer = (uint8_t*)malloc(activityBufferSize);
    }

    *buffer = activityBuffer;
    *size = activityBufferSize;
    *maxNumRecords = 0; // Let CUPTI decide
}

// Buffer completion callback
static void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                           size_t size, size_t validSize) {
    CUptiResult result;
    CUpti_Activity* record = NULL;
    int recordCount = 0;

    DEBUG_PRINTF("[CUPTI] bufferCompleted called: validSize=%zu\n", validSize);

    while (1) {
        result = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else if (result != CUPTI_SUCCESS) {
            const char* errstr;
            cuptiGetResultString(result, &errstr);
            fprintf(stderr, "[CUPTI] Error reading activity record: %s\n", errstr);
            break;
        }

        recordCount++;
        DEBUG_PRINTF("[CUPTI] Activity record %d: kind=%d\n", recordCount, record->kind);

        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
            record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*)record;
            uint64_t duration_ns = kernel->end - kernel->start;

            DEBUG_PRINTF("[CUPTI] Kernel activity: correlationId=%u, duration=%lu ns\n",
                   kernel->correlationId, duration_ns);

            // Call stub function for uprobe attachment
            DTRACE_PROBE2(parcagpu, kernel_executed, kernel->correlationId, duration_ns);
            parcagpuKernelExecuted(kernel->correlationId, duration_ns);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) {
            CUpti_ActivityGraphTrace* trace = (CUpti_ActivityGraphTrace*)record;
            uint64_t duration_ns = trace->end - trace->start;

            DEBUG_PRINTF("[CUPTI] Graph activity: correlationId=%u, duration=%lu ns\n",
                   trace->correlationId, duration_ns);

            // Call stub function for uprobe attachment
            DTRACE_PROBE2(parcagpu, kernel_executed, trace->correlationId, duration_ns);
            parcagpuKernelExecuted(trace->correlationId, duration_ns);
        }
    }

    DEBUG_PRINTF("[CUPTI] Processed %d activity records\n", recordCount);

    // Report any records dropped due to buffer overflow
    size_t dropped;
    result = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (result == CUPTI_SUCCESS && dropped > 0) {
        fprintf(stderr, "[CUPTI] Warning: %zu activity records dropped\n", dropped);
    }
}

// Cleanup function (destructor disabled to prevent early cleanup)
void cleanup(void) {
    // Flush any remaining activity records
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    // Unsubscribe from callbacks
    if (subscriber) {
        cuptiUnsubscribe(subscriber);
    }

    // Free activity buffer
    if (activityBuffer) {
        free(activityBuffer);
    }

    DEBUG_PRINTF("[CUPTI] Cleanup completed\n");
}