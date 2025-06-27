#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Minimal CUDA types for testing
typedef int cudaError_t;
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
typedef unsigned int cudaStreamCaptureStatus;

typedef struct {
    unsigned int x, y, z;
} dim3;

// Function declarations
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus);
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                            void** args, size_t sharedMem, cudaStream_t stream);

int main() {
    printf("Testing mock CUDA runtime... (PID: %d)\n", getpid());
    
    // Run test loop for 60 seconds to give time for tracer attachment
    for (int i = 0; i < 20; i++) {
        printf("=== Test iteration %d/20 ===\n", i + 1);
        
        // Test event creation
        cudaEvent_t event1, event2;
        cudaError_t err = cudaEventCreateWithFlags(&event1, 0x1);
        printf("cudaEventCreateWithFlags result: %d\n", err);
        
        err = cudaEventCreateWithFlags(&event2, 0x1);
        printf("cudaEventCreateWithFlags result: %d\n", err);
        
        // Test stream capture status
        cudaStreamCaptureStatus status;
        err = cudaStreamIsCapturing(NULL, &status);
        printf("cudaStreamIsCapturing result: %d, status: %d\n", err, status);
        
        // Test event recording
        err = cudaEventRecord(event1, NULL);
        printf("cudaEventRecord result: %d\n", err);
        
        // Test kernel launch
        dim3 grid = {1, 1, 1};
        dim3 block = {256, 1, 1};
        err = cudaLaunchKernel((void*)0x12345678, grid, block, NULL, 0, NULL);
        printf("cudaLaunchKernel result: %d\n", err);
        
        // Record second event
        err = cudaEventRecord(event2, NULL);
        printf("cudaEventRecord result: %d\n", err);
        
        // Test event synchronization
        err = cudaEventSynchronize(event2);
        printf("cudaEventSynchronize result: %d\n", err);
        
        // Test elapsed time
        float ms;
        err = cudaEventElapsedTime(&ms, event1, event2);
        printf("cudaEventElapsedTime result: %d, elapsed: %.3f ms\n", err, ms);
        
        printf("Iteration %d completed!\n", i + 1);
        sleep(3);  // Wait 3 seconds between iterations
    }
    
    printf("All tests completed!\n");
    return 0;
}