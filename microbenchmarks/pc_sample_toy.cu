// pc_sample_toy.cu — a simple GPU busy-loop for testing PC sampling
// Compile: make microbenchmarks  (or: nvcc -g -lineinfo -arch=native -o pc_sample_toy pc_sample_toy.cu)
// Run:     ./pc_sample_toy

#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Kernel A: heavy FP math (sin/cos chain)
__global__ void trig_storm(float *out, int n, unsigned long long iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float x = (float)idx * 0.001f;
  for (unsigned long long i = 0; i < iters; i++) {
    x = sinf(x) * cosf(x) + 0.1f;
  }
  out[idx] = x;
}

// Kernel B: integer bit-twiddling
__global__ void hash_churn(unsigned int *out, int n, unsigned long long iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  unsigned int h = idx ^ 0xdeadbeef;
  for (unsigned long long i = 0; i < iters; i++) {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    h += (unsigned int)i;
  }
  out[idx] = h;
}

// Kernel C: shared-memory bouncing
__global__ void shmem_bounce(float *out, int n, unsigned long long iters) {
  __shared__ float tile[256];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  tile[tid] = (float)idx;
  __syncthreads();

  for (unsigned long long i = 0; i < iters; i++) {
    tile[tid] += tile[(tid + 1) % blockDim.x] * 0.01f;
    __syncthreads();
  }

  if (idx < n)
    out[idx] = tile[tid];
}

void go() {
  const int N = 1 << 18; // 256K elements
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  float *d_float;
  unsigned int *d_uint;

  CHECK(cudaMalloc(&d_float, N * sizeof(float)));
  CHECK(cudaMalloc(&d_uint, N * sizeof(unsigned int)));

  printf("Launching GPU kernels — attach your profiler now.\n");
  printf("PID: %d\n\n", getpid());

  sleep(1);
  // Each kernel runs for roughly 0.5–1 second depending on GPU.
  // Tune the iteration count up/down as needed.

  printf("  [1/3] trig_storm ...\n");
  trig_storm<<<blocks, threads>>>(d_float, N, 500000ULL);
  CHECK(cudaDeviceSynchronize());

  printf("  [2/3] hash_churn ...\n");
  hash_churn<<<blocks, threads>>>(d_uint, N, 2000000ULL);
  CHECK(cudaDeviceSynchronize());

  printf("  [3/3] shmem_bounce ...\n");
  shmem_bounce<<<blocks, threads>>>(d_float, N, 50000ULL);
  CHECK(cudaDeviceSynchronize());

  printf("\nDone.\n");

  CHECK(cudaFree(d_float));
  CHECK(cudaFree(d_uint));
}

int main(int argc, char **argv) {
  int loops = 1;
  if (argc > 1) {
    loops = atoi(argv[1]);
  }
  while (loops-- > 0) {
    go();
  }
}
