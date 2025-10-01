#define _POSIX_C_SOURCE 199309L
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda.h>
#include <cupti.h>

// Forward declarations for functions we'll call from the library
typedef int (*InitializeInjectionFunc)(void);

// Global callback functions that will be registered by InitializeInjection
static void (*bufferRequestedCallback)(uint8_t **buffer, size_t *size, size_t *maxNumRecords) = NULL;
static void (*bufferCompletedCallback)(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) = NULL;
static void (*runtimeApiCallback)(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata) = NULL;

// Helper to create activity buffer with kernel records
static uint8_t *create_kernel_activity_buffer(size_t *validSize,
                                               uint32_t correlationId, uint32_t deviceId,
                                               uint32_t streamId, const char *kernelName) {
    // Allocate buffer large enough for one kernel record
    size_t bufferSize = sizeof(CUpti_ActivityKernel4) + 256;
    uint8_t *buffer = (uint8_t *)malloc(bufferSize);
    memset(buffer, 0, bufferSize);

    CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)buffer;
    kernel->kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    kernel->correlationId = correlationId;
    kernel->deviceId = deviceId;
    kernel->streamId = streamId;
    kernel->start = 1000000000UL + (correlationId * 1000000UL);
    kernel->end = kernel->start + 500000UL;

    // Copy kernel name
    char *namePtr = (char *)(buffer + sizeof(CUpti_ActivityKernel4));
    strncpy(namePtr, kernelName, 255);
    kernel->name = namePtr;

    *validSize = sizeof(CUpti_ActivityKernel4);
    return buffer;
}

// Helper to create activity buffer with graph records
static uint8_t *create_graph_activity_buffer(size_t *validSize,
                                              uint32_t correlationId, uint32_t deviceId,
                                              uint32_t streamId, uint32_t graphId) {
    size_t bufferSize = sizeof(CUpti_ActivityGraphTrace);
    uint8_t *buffer = (uint8_t *)malloc(bufferSize);
    memset(buffer, 0, bufferSize);

    CUpti_ActivityGraphTrace *graph = (CUpti_ActivityGraphTrace *)buffer;
    graph->kind = CUPTI_ACTIVITY_KIND_GRAPH_TRACE;
    graph->correlationId = correlationId;
    graph->deviceId = deviceId;
    graph->streamId = streamId;
    graph->graphId = graphId;
    graph->start = 1000000000UL + (correlationId * 1000000UL);
    graph->end = graph->start + 300000UL;

    *validSize = sizeof(CUpti_ActivityGraphTrace);
    return buffer;
}

int main(int argc, char **argv) {
    const char *lib_path = argc > 1 ? argv[1] : "./libparcagpucupti.so";
    bool run_forever = false;

    // Check for --forever flag
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--forever") == 0) {
            run_forever = true;
            break;
        }
    }

    fprintf(stderr, "Loading library: %s\n", lib_path);
    if (run_forever) {
        fprintf(stderr, "Running in continuous mode (Ctrl-C to stop)\n");
    }
    void *cupti_prof_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!cupti_prof_handle) {
        fprintf(stderr, "Failed to load library: %s\n", dlerror());
        return 1;
    }

    // Get InitializeInjection function
    InitializeInjectionFunc initFunc = (InitializeInjectionFunc)dlsym(cupti_prof_handle, "InitializeInjection");
    if (!initFunc) {
        fprintf(stderr, "Failed to find InitializeInjection: %s\n", dlerror());
        dlclose(cupti_prof_handle);
        return 1;
    }

    // Call InitializeInjection first to register callbacks
    fprintf(stderr, "Calling InitializeInjection...\n");
    int result = initFunc();
    fprintf(stderr, "InitializeInjection returned: %d\n", result);

    // Now get pointers to the global callback variables from mock CUPTI
    void **runtime_api_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_runtime_api_callback");
    void **buffer_requested_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_buffer_requested_callback");
    void **buffer_completed_cb_ptr = (void **)dlsym(RTLD_DEFAULT, "__cupti_buffer_completed_callback");

    fprintf(stderr, "Looking for callbacks: runtime=%p, requested=%p, completed=%p\n",
            (void *)runtime_api_cb_ptr, (void *)buffer_requested_cb_ptr, (void *)buffer_completed_cb_ptr);

    // Dereference to get the actual callback functions
    if (runtime_api_cb_ptr) {
        runtimeApiCallback = (void (*)(void *, CUpti_CallbackDomain, CUpti_CallbackId, const CUpti_CallbackData *))*runtime_api_cb_ptr;
        fprintf(stderr, "Got runtime callback: %p\n", (void *)runtimeApiCallback);
    }
    if (buffer_requested_cb_ptr) {
        bufferRequestedCallback = (void (*)(uint8_t **, size_t *, size_t *))*buffer_requested_cb_ptr;
    }
    if (buffer_completed_cb_ptr) {
        bufferCompletedCallback = (void (*)(CUcontext, uint32_t, uint8_t *, size_t, size_t))*buffer_completed_cb_ptr;
        fprintf(stderr, "Got buffer completed callback: %p\n", (void *)bufferCompletedCallback);
    }

    // Check if we have the callback pointers
    if (!runtimeApiCallback || !bufferCompletedCallback) {
        fprintf(stderr, "Warning: Could not get callback pointers from mock CUPTI.\n");
        fprintf(stderr, "Test will run but won't be able to simulate full callback flow.\n");
        fprintf(stderr, "The library is loaded and InitializeInjection was called successfully.\n");

        // Sleep for a bit to keep the library loaded
        fprintf(stderr, "Keeping library loaded for 5 seconds...\n");
        sleep(5);

        dlclose(cupti_prof_handle);
        return 0;
    }

    // Now simulate CUPTI callbacks
    fprintf(stderr, "\n=== Starting test simulation (1000 events/second) ===\n");

    uint32_t correlationId = 1;
    struct timespec sleep_time = {0, 1000000}; // 1ms sleep = 1000 events/second

    for (int i = 0; run_forever || i < 100; i++) {
        // Simulate a batch of kernel launches
        for (int j = 0; j < 5; j++) {
            CUpti_CallbackData cbdata = {0};
            cbdata.callbackSite = CUPTI_API_EXIT;
            cbdata.correlationId = correlationId;

            // Alternate between kernel and graph launches
            if (correlationId % 2 == 0) {
                runtimeApiCallback(NULL, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
                                   &cbdata);
            } else {
                runtimeApiCallback(NULL, CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,
                                   &cbdata);
            }

            correlationId++;
        }

        // After every 2 batches, simulate buffer completion with activity records
        if (i % 2 == 0 && i > 0) {
            for (int k = 0; k < 5; k++) {
                uint32_t recCorrelationId = correlationId - 10 + k;
                uint8_t *buffer;
                size_t validSize;

                if (recCorrelationId % 2 == 0) {
                    buffer = create_kernel_activity_buffer(&validSize, recCorrelationId, 0, 1, "mock_cuda_kernel_name");
                    bufferCompletedCallback(NULL, 1, buffer, 32 * 1024, validSize);
                } else {
                    buffer = create_graph_activity_buffer(&validSize, recCorrelationId, 0, 1, recCorrelationId / 2);
                    bufferCompletedCallback(NULL, 1, buffer, 32 * 1024, validSize);
                }

                free(buffer);
            }
        }

        nanosleep(&sleep_time, NULL);

        // Print status periodically when running forever
        if (run_forever && i % 100 == 0) {
            fprintf(stderr, "[Status] Generated %d events so far...\n", correlationId - 1);
        }
    }

    if (!run_forever) {
        fprintf(stderr, "\n=== Test completed. Generated ~%d events ===\n", correlationId - 1);
    }
    fprintf(stderr, "Library will be unloaded, triggering cleanup...\n");

    // Call cleanup explicitly before closing the library to avoid atexit handler issues
    typedef void (*CleanupFunc)(void);
    CleanupFunc cleanup = (CleanupFunc)dlsym(cupti_prof_handle, "cleanup");
    if (cleanup) {
        cleanup();
    }

    // Clear the callback pointers before closing to avoid crashes during cleanup
    if (runtime_api_cb_ptr) *runtime_api_cb_ptr = NULL;
    if (buffer_requested_cb_ptr) *buffer_requested_cb_ptr = NULL;
    if (buffer_completed_cb_ptr) *buffer_completed_cb_ptr = NULL;

    dlclose(cupti_prof_handle);
    fprintf(stderr, "Cleanup complete.\n");

    // Note: We use _exit instead of return to avoid the atexit handler registered
    // by the library from being called after we've already closed it with dlclose.
    // The cleanup() function is now idempotent, so in real usage (where the library
    // isn't dynamically closed) the atexit handler works correctly.
    _exit(0);
}
