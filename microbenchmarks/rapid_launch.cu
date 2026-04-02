// rapid_launch.cu — measures per-kernel-launch overhead from CUPTI injection.
// Launches many tiny kernels to stress the callback path.
//
// Compile: nvcc -o rapid_launch rapid_launch.cu
// Run:     ./rapid_launch [num_launches]
//
// Compare:
//   ./rapid_launch 50000                                          # baseline
//   CUDA_INJECTION64_PATH=.../libparcagpucupti.so ./rapid_launch 50000  # injected

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void empty_kernel() {}

static double now_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  int n = 50000;
  if (argc > 1)
    n = atoi(argv[1]);

  // Warm up the CUDA context and any injection library init.
  empty_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  // Synchronous launches — each one round-trips through CUPTI callbacks.
  double t0 = now_sec();
  for (int i = 0; i < n; i++) {
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
  }
  double t1 = now_sec();

  double elapsed = t1 - t0;
  printf("%d launches in %.3f s  (%.1f us/launch)\n", n, elapsed,
         elapsed / n * 1e6);
  return 0;
}
