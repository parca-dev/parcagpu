#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Simple kernel that does some work
__global__ void simpleKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple computation to keep the kernel busy
        data[idx] = idx * idx + idx;
    }
}

// Another kernel for variety
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void runSimpleKernel() {
    const int n = 1024;
    const int size = n * sizeof(int);
    
    int *h_data = (int*)malloc(size);
    int *d_data;
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_data, size), "cudaMalloc failed");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
    
    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    printf("[App] Launching simpleKernel with grid=(%d,%d,%d), block=(%d,%d,%d)\n",
           gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
    
    simpleKernel<<<gridSize, blockSize>>>(d_data, n);
    checkCudaError(cudaGetLastError(), "simpleKernel launch failed");
    
    // Wait for completion
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    // Copy result back
    checkCudaError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
}

void runVectorAdd() {
    const int n = 2048;
    const int size = n * sizeof(int);
    
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    int *d_a, *d_b, *d_c;
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a failed");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b failed");
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c failed");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D a failed");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D b failed");
    
    // Launch kernel
    dim3 blockSize(512);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    printf("[App] Launching vectorAdd with grid=(%d,%d,%d), block=(%d,%d,%d)\n",
           gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    checkCudaError(cudaGetLastError(), "vectorAdd launch failed");
    
    // Wait for completion
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    // Copy result back
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c failed");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

void runGraphLaunch() {
    const int n = 1024;
    const int size = n * sizeof(int);
    
    int *h_data = (int*)malloc(size);
    int *d_data;
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_data, size), "cudaMalloc failed");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
    
    // Create CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    checkCudaError(cudaGraphCreate(&graph, 0), "cudaGraphCreate failed");
    
    // Add kernel node to graph
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void*)simpleKernel;
    kernelParams.gridDim = dim3((n + 255) / 256, 1, 1);
    kernelParams.blockDim = dim3(256, 1, 1);
    kernelParams.sharedMemBytes = 0;
    
    void* kernelArgs[] = {(void*)&d_data, (void*)&n};
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    
    cudaGraphNode_t kernelNode;
    checkCudaError(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams), 
                   "cudaGraphAddKernelNode failed");
    
    // Instantiate the graph
    checkCudaError(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), 
                   "cudaGraphInstantiate failed");
    
    printf("[App] Launching CUDA graph with kernel\n");
    
    // Launch the graph
    checkCudaError(cudaGraphLaunch(graphExec, 0), "cudaGraphLaunch failed");
    
    // Wait for completion
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    // Copy result back
    checkCudaError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_data);
    free(h_data);
}

int main() {
    printf("CUPTI Profiling Test Program\n");
    printf("============================\n");
    printf("This program runs real CUDA kernels to test CUPTI profiling\n\n");
    
    // Initialize CUDA driver API (required for CUPTI)
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", cuResult);
        return 1;
    }
    
    // Initialize CUDA runtime
    checkCudaError(cudaSetDevice(0), "cudaSetDevice failed");
    
    int device;
    cudaDeviceProp prop;
    checkCudaError(cudaGetDevice(&device), "cudaGetDevice failed");
    checkCudaError(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");
    
    printf("Using device %d: %s\n", device, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Run different compute patterns in a loop
    for (int iteration = 0; iteration < 10; iteration++) {
        printf("=== Iteration %d ===\n", iteration + 1);
        
        printf("Running simpleKernel...\n");
        runSimpleKernel();
        
        usleep(500000); // 500ms pause
        
        printf("Running vectorAdd...\n");
        runVectorAdd();
        
        usleep(500000); // 500ms pause
        
        printf("Running graph launch...\n");
        runGraphLaunch();
        
        usleep(500000); // 500ms pause
        
        printf("Iteration %d completed\n\n", iteration + 1);
    }
    
    printf("All iterations completed. Check profiling output for timing data.\n");
    
    return 0;
}