// graph_repro.cu — replays a captured graph via the DRIVER cuGraphLaunch (the
// path parcagpu missed). Build: nvcc -o graph_repro test/graph_repro.cu -lcuda
// Run: ./graph_repro [seconds]. See test/graph-repro-real.sh for the guard.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CU_CHECK(x)                                                            \
  do {                                                                         \
    CUresult e = (x);                                                          \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *s = nullptr;                                                 \
      cuGetErrorString(e, &s);                                                 \
      fprintf(stderr, "driver error %s at %s:%d\n", s ? s : "?", __FILE__,     \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void add_one(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += 1.0f;
}

int main(int argc, char **argv) {
  int seconds = (argc > 1) ? atoi(argv[1]) : 20;

  // Force driver + runtime init.
  CUDA_CHECK(cudaFree(0));
  CU_CHECK(cuInit(0));

  const int n = 1 << 16;
  float *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Build a graph by capturing a few kernel launches into the stream.
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int k = 0; k < 8; k++) {
    add_one<<<(n + 255) / 256, 256, 0, stream>>>(d, n);
  }
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t exec;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // cudaGraphExec_t is the same handle as CUgraphExec; replay via driver API.
  CUgraphExec drvExec = reinterpret_cast<CUgraphExec>(exec);
  CUstream drvStream = reinterpret_cast<CUstream>(stream);

  fprintf(stderr,
          "graph_repro: replaying graph via driver cuGraphLaunch for %ds\n",
          seconds);

  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  unsigned long long launches = 0;
  for (;;) {
    CU_CHECK(cuGraphLaunch(drvExec, drvStream));
    launches++;
    if ((launches & 0x3ff) == 0) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      if (now.tv_sec - start.tv_sec >= seconds)
        break;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  fprintf(stderr, "graph_repro: done, %llu driver graph launches\n", launches);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d);
  return 0;
}
