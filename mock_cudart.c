#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int cudaError_t;
typedef unsigned int cudaStreamCaptureStatus;

struct CUstream_st {
    int dummy;
};
typedef struct CUstream_st *cudaStream_t;

struct CUevent_st {
    struct timespec timestamp;
    int recorded;
};
typedef struct CUevent_st *cudaEvent_t;

typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorInvalidResourceHandle 2

#define cudaEventBlockingSync 0x1
#define cudaStreamCaptureStatusNone 0

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (!event) {
        return cudaErrorInvalidValue;
    }

    *event = (cudaEvent_t)malloc(sizeof(struct CUevent_st));
    if (!*event) {
        return cudaErrorInvalidValue;
    }

    (*event)->recorded = 0;
    memset(&(*event)->timestamp, 0, sizeof(struct timespec));

    printf("[MOCK] cudaEventCreateWithFlags: Created event %p with flags %u\n", (void*)*event, flags);
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) {
        return cudaErrorInvalidResourceHandle;
    }

    clock_gettime(CLOCK_MONOTONIC, &event->timestamp);
    event->recorded = 1;

    printf("[MOCK] cudaEventRecord: Recorded event %p on stream %p at time %ld.%09ld\n",
           (void*)event, (void*)stream, event->timestamp.tv_sec, event->timestamp.tv_nsec);
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) {
        return cudaErrorInvalidResourceHandle;
    }

    if (!event->recorded) {
        return cudaErrorInvalidResourceHandle;
    }

    // Simulate some delay
    usleep(1000); // 1ms

    printf("[MOCK] cudaEventSynchronize: Synchronized with event %p\n", (void*)event);
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) {
        return cudaErrorInvalidValue;
    }

    if (!start->recorded || !end->recorded) {
        return cudaErrorInvalidResourceHandle;
    }

    // Calculate elapsed time in milliseconds
    long sec_diff = end->timestamp.tv_sec - start->timestamp.tv_sec;
    long nsec_diff = end->timestamp.tv_nsec - start->timestamp.tv_nsec;

    *ms = (float)(sec_diff * 1000.0 + nsec_diff / 1000000.0);

    // Add a small random component to simulate kernel execution time
    *ms += (float)(rand() % 100) / 10.0; // 0-10ms random addition

    printf("[MOCK] cudaEventElapsedTime: Elapsed time between %p and %p: %.3f ms\n",
           (void*)start, (void*)end, *ms);
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus) {
    if (!pCaptureStatus) {
        return cudaErrorInvalidValue;
    }

    // Always report that the stream is not capturing
    *pCaptureStatus = cudaStreamCaptureStatusNone;

    printf("[MOCK] cudaStreamIsCapturing: Stream %p is not capturing\n", (void*)stream);
    return cudaSuccess;
}

static void busy_loop(int milliseconds) {
    // Simulate a busy wait for the specified number of milliseconds
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    do {
        clock_gettime(CLOCK_MONOTONIC, &end);
    } while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 < milliseconds);
}

cudaError_t cudaLaunchKernel(const void* func,
                            dim3 gridDim,
                            dim3 blockDim,
                            void** args,
                            size_t sharedMem,
                            cudaStream_t stream) {
    printf("[MOCK] cudaLaunchKernel: Launching kernel %p with grid(%u,%u,%u) block(%u,%u,%u) sharedMem=%zu stream=%p\n",
           func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, (void*)stream);

    // spin cpu for 10 to 20 milliseconds to simulate kernel execution
    int sleep_time = 10 + (rand() % 11); // Random sleep between
    busy_loop(sleep_time);

    return cudaSuccess;
}

// Constructor function to initialize the mock
__attribute__((constructor))
void init_mock_cudart() {
    printf("[MOCK] Initialized mock CUDA runtime library\n");
    srand(time(NULL));
}