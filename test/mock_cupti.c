#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cupti.h>

// Define callback function types if not already defined by CUPTI headers
#ifndef CUpti_BufferRequestFunc
typedef void (*CUpti_BufferRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
#endif

#ifndef CUpti_BufferCompletedFunc
typedef void (*CUpti_BufferCompletedFunc)(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
#endif

// Global storage for registered callbacks (exported for test access)
CUpti_CallbackFunc __cupti_runtime_api_callback = NULL;
void *__cupti_runtime_api_userdata = NULL;
CUpti_BufferRequestFunc __cupti_buffer_requested_callback = NULL;
CUpti_BufferCompletedFunc __cupti_buffer_completed_callback = NULL;

// Mock implementations of CUPTI APIs used by cupti-prof.c

CUptiResult cuptiActivityFlushPeriod(uint32_t period) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityFlushPeriod(%u)\n", period);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiGetResultString(CUptiResult result, const char **str) {
    static const char *success = "CUPTI_SUCCESS";
    static const char *error = "CUPTI_ERROR";
    *str = (result == CUPTI_SUCCESS) ? success : error;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                           CUpti_CallbackFunc callback,
                           void *userdata) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiSubscribe()\n");
    __cupti_runtime_api_callback = callback;
    __cupti_runtime_api_userdata = userdata;
    *subscriber = (CUpti_SubscriberHandle)0x1234;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
    (void)subscriber;  // Mark as intentionally unused
    fprintf(stderr, "[MOCK_CUPTI] cuptiEnableCallback(enable=%u, domain=%u, cbid=%u)\n",
            enable, domain, cbid);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BufferRequestFunc funcBufferRequested,
    CUpti_BufferCompletedFunc funcBufferCompleted) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityRegisterCallbacks()\n");
    __cupti_buffer_requested_callback = funcBufferRequested;
    __cupti_buffer_completed_callback = funcBufferCompleted;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityEnable(kind=%u)\n", kind);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
    fprintf(stderr, "[MOCK_CUPTI] cuptiActivityFlushAll(flag=%u)\n", flag);
    // Don't actually call any callbacks during flush to avoid issues during cleanup
    (void)flag;
    return CUPTI_SUCCESS;
}

// Track iteration state per buffer
typedef struct {
    uint8_t *buffer;
    size_t offset;
} BufferIterState;

static BufferIterState iter_state = {NULL, 0};

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity **record) {
    // Reset state if this is a new buffer
    if (iter_state.buffer != buffer) {
        iter_state.buffer = buffer;
        iter_state.offset = 0;
    }

    // Check if we've reached the end
    if (iter_state.offset >= validBufferSizeBytes) {
        iter_state.buffer = NULL;
        iter_state.offset = 0;
        return CUPTI_ERROR_MAX_LIMIT_REACHED;
    }

    // Get the record at current offset
    CUpti_Activity *activity = (CUpti_Activity *)(buffer + iter_state.offset);

    // Determine record size based on kind
    size_t recordSize = 0;
    switch (activity->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_KERNEL:
            recordSize = sizeof(CUpti_ActivityKernel4);
            break;
        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
            recordSize = sizeof(CUpti_ActivityGraphTrace);
            break;
        default:
            // Unknown kind, can't continue
            iter_state.buffer = NULL;
            iter_state.offset = 0;
            return CUPTI_ERROR_INVALID_KIND;
    }

    // Advance offset for next call
    iter_state.offset += recordSize;

    *record = activity;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context,
                                              uint32_t streamId,
                                              size_t *dropped) {
    (void)context;  // Mark as intentionally unused
    (void)streamId;
    *dropped = 0;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
    (void)subscriber;  // Mark as intentionally unused
    fprintf(stderr, "[MOCK_CUPTI] cuptiUnsubscribe()\n");
    return CUPTI_SUCCESS;
}
