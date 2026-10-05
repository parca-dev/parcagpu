// pc_multi.cu — standalone check of two PC-sampling questions, no shim involved:
//   1. Can one process PC-sample several GPUs at once? (one context per GPU,
//      all started before any kernel runs, kernels run concurrently)
//   2. Can two processes PC-sample the same GPU at once? Run two copies with
//      a hold time, e.g. `./pc_multi 1 20 & sleep 2; ./pc_multi 1 5`.
//
// Build: nvcc -O2 -lineinfo -o pc_multi test/pc_multi.cu -lcuda -lcupti
// Run:   ./pc_multi [ndev=all] [seconds=3]   (PC_MULTI_NOSAMPLE=1 for baseline,
//        PC_MULTI_PERIOD=n for an explicit 2^n-cycle sampling period)
// Prints per-device CUPTI return codes and sample counts. Exit 0 only if every
// device produced samples.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_pcsampling.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <vector>

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

static const char *cuptiErr(CUptiResult r) {
  const char *s = nullptr;
  cuptiGetResultString(r, &s);
  return s ? s : "?";
}

// Spin for roughly `cycles` clocks so the kernel is long enough to sample.
__global__ void spin(long long cycles, unsigned long long *out) {
  long long start = clock64();
  unsigned long long acc = threadIdx.x;
  while (clock64() - start < cycles)
    acc = acc * 6364136223846793005ULL + 1442695040888963407ULL;
  if (acc == 42)
    *out = acc;
}

static double now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static CUpti_PCSamplingData allocData(size_t pcs, size_t nStall) {
  CUpti_PCSamplingData d{};
  d.size = sizeof(CUpti_PCSamplingData);
  d.collectNumPcs = pcs;
  d.pPcData = (CUpti_PCSamplingPCData *)calloc(pcs, sizeof(CUpti_PCSamplingPCData));
  for (size_t i = 0; i < pcs; i++) {
    d.pPcData[i].size = sizeof(CUpti_PCSamplingPCData);
    d.pPcData[i].stallReason =
        (CUpti_PCSamplingStallReason *)calloc(nStall, sizeof(CUpti_PCSamplingStallReason));
  }
  return d;
}

struct Dev {
  int ordinal;
  CUcontext ctx{};
  CUstream stream{};
  unsigned long long *out{};
  size_t nStall{};
  CUpti_PCSamplingData cfgBuf{}, outBuf{};
  CUptiResult enableRes{}, configRes{}, startRes{}, stopRes{};
  CUptiResult getRes{CUPTI_SUCCESS};
  unsigned long long samples{}, dropped{}, pcs{};
};

// Drain everything CUPTI holds for this context into the sample counters.
static void drain(Dev &d) {
  CUpti_PCSamplingGetDataParams p{};
  p.size = CUpti_PCSamplingGetDataParamsSize;
  p.ctx = d.ctx;
  p.pcSamplingData = &d.outBuf;
  for (int i = 0; i < 100000; i++) {
    d.outBuf.totalNumPcs = 0;
    CUptiResult r = cuptiPCSamplingGetData(&p);
    if (r != CUPTI_SUCCESS) {
      d.getRes = r;
      return;
    }
    d.pcs += d.outBuf.totalNumPcs;
    d.dropped += d.outBuf.droppedSamples;
    for (size_t k = 0; k < d.outBuf.totalNumPcs; k++)
      for (size_t s = 0; s < d.outBuf.pPcData[k].stallReasonCount; s++)
        d.samples += d.outBuf.pPcData[k].stallReason[s].samples;
    if (d.outBuf.totalNumPcs == 0 && d.outBuf.remainingNumPcs == 0)
      return;
  }
}

static void setup(Dev &d) {
  CU_CHECK(cuCtxSetCurrent(d.ctx));

  CUpti_PCSamplingEnableParams en{};
  en.size = CUpti_PCSamplingEnableParamsSize;
  en.ctx = d.ctx;
  d.enableRes = cuptiPCSamplingEnable(&en);
  if (d.enableRes != CUPTI_SUCCESS)
    return;

  CUpti_PCSamplingGetNumStallReasonsParams nsp{};
  nsp.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
  nsp.ctx = d.ctx;
  nsp.numStallReasons = &d.nStall;
  cuptiPCSamplingGetNumStallReasons(&nsp);

  std::vector<uint32_t> idx(d.nStall);
  std::vector<char *> names(d.nStall);
  for (auto &n : names)
    n = (char *)calloc(CUPTI_STALL_REASON_STRING_SIZE, 1);
  CUpti_PCSamplingGetStallReasonsParams srp{};
  srp.size = CUpti_PCSamplingGetStallReasonsParamsSize;
  srp.ctx = d.ctx;
  srp.numStallReasons = d.nStall;
  srp.stallReasonIndex = idx.data();
  srp.stallReasons = names.data();
  cuptiPCSamplingGetStallReasons(&srp);

  d.cfgBuf = allocData(1024, d.nStall);
  d.outBuf = allocData(1024, d.nStall);

  CUpti_PCSamplingConfigurationInfo info[6]{};
  info[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  info[0].attributeData.stallReasonData.stallReasonCount = d.nStall;
  info[0].attributeData.stallReasonData.pStallReasonIndex = idx.data();
  info[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  info[1].attributeData.samplingDataBufferData.samplingDataBuffer = &d.cfgBuf;
  info[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  info[2].attributeData.collectionModeData.collectionMode =
      CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
  info[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
  info[3].attributeData.enableStartStopControlData.enableStartStopControl = 1;
  // Same as the shim: CUPTI's default scratch buffer overflows almost at once
  // under a busy kernel and GetData then returns OUT_OF_MEMORY.
  info[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  info[4].attributeData.scratchBufferSizeData.scratchBufferSize = 16 << 20;

  CUpti_PCSamplingConfigurationInfoParams cp{};
  cp.size = CUpti_PCSamplingConfigurationInfoParamsSize;
  cp.ctx = d.ctx;
  cp.numAttributes = 5;
  // PC_MULTI_PERIOD=<5..31>: set an explicit sampling period (2^n cycles)
  // instead of CUPTI's default.
  if (const char *per = getenv("PC_MULTI_PERIOD")) {
    info[5].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    info[5].attributeData.samplingPeriodData.samplingPeriod = atoi(per);
    cp.numAttributes = 6;
  }
  cp.pPCSamplingConfigurationInfo = info;
  d.configRes = cuptiPCSamplingSetConfigurationAttribute(&cp);
}

int main(int argc, char **argv) {
  CU_CHECK(cuInit(0));
  int total = 0;
  CU_CHECK(cuDeviceGetCount(&total));
  int ndev = argc > 1 ? atoi(argv[1]) : total;
  double secs = argc > 2 ? atof(argv[2]) : 3.0;
  if (ndev <= 0 || ndev > total)
    ndev = total;
  // PC_MULTI_NOSAMPLE=1 runs the same loop without PC sampling (baseline).
  const bool sample = getenv("PC_MULTI_NOSAMPLE") == nullptr;
  printf("pid %d: %s %d of %d device(s) for %.1fs\n", getpid(),
         sample ? "sampling" : "NOT sampling", ndev,
         total, secs);

  std::vector<Dev> devs(ndev);
  for (int i = 0; i < ndev; i++) {
    Dev &d = devs[i];
    d.ordinal = i;
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, i));
    CU_CHECK(cuDevicePrimaryCtxRetain(&d.ctx, dev));
    CU_CHECK(cuCtxSetCurrent(d.ctx));
    CU_CHECK(cuStreamCreate(&d.stream, CU_STREAM_NON_BLOCKING));
    CU_CHECK(cuMemAlloc((CUdeviceptr *)&d.out, sizeof(*d.out)));
    if (sample)
      setup(d);
  }

  // Start all contexts before launching anything, so sampling overlaps.
  for (auto &d : devs) {
    if (!sample || d.enableRes || d.configRes)
      continue;
    CU_CHECK(cuCtxSetCurrent(d.ctx));
    CUpti_PCSamplingStartParams sp{};
    sp.size = CUpti_PCSamplingStartParamsSize;
    sp.ctx = d.ctx;
    d.startRes = cuptiPCSamplingStart(&sp);
  }

  // Rounds of short (~2 ms) kernels: one per device per round, all launched
  // before any is waited on, then drain every device. Short kernels matter:
  // in KERNEL_SERIALIZED mode a single multi-second kernel fills CUPTI's HW
  // buffer and GetData returns OUT_OF_MEMORY. If sampling serialized work
  // across devices, a round would take ~ndev x 2 ms instead of ~2 ms.
  long long cycles[64]{};
  for (auto &d : devs) {
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d.ordinal);
    cycles[d.ordinal] = (long long)(0.002 * clockKHz * 1000.0);
  }
  double t0 = now();
  long rounds = 0;
  while (now() - t0 < secs) {
    for (auto &d : devs) {
      CU_CHECK(cuCtxSetCurrent(d.ctx));
      spin<<<1024, 256, 0, (cudaStream_t)d.stream>>>(cycles[d.ordinal], d.out);
    }
    for (auto &d : devs) {
      CU_CHECK(cuCtxSetCurrent(d.ctx));
      CU_CHECK(cuStreamSynchronize(d.stream));
      if (sample && !d.startRes && !d.enableRes && !d.configRes)
        drain(d);
    }
    rounds++;
  }
  double wall = now() - t0;

  int bad = 0;
  for (auto &d : devs) {
    CU_CHECK(cuCtxSetCurrent(d.ctx));
    if (sample && !d.enableRes && !d.configRes && !d.startRes) {
      CUpti_PCSamplingStopParams st{};
      st.size = CUpti_PCSamplingStopParamsSize;
      st.ctx = d.ctx;
      d.stopRes = cuptiPCSamplingStop(&st);
      drain(d);
    }
    CUpti_PCSamplingDisableParams dp{};
    dp.size = CUpti_PCSamplingDisableParamsSize;
    dp.ctx = d.ctx;
    if (sample)
      cuptiPCSamplingDisable(&dp);
    printf("dev %d: enable=%s config=%s start=%s stop=%s getData=%s  "
           "pcs=%llu samples=%llu dropped=%llu\n",
           d.ordinal, cuptiErr(d.enableRes), cuptiErr(d.configRes),
           cuptiErr(d.startRes), cuptiErr(d.stopRes), cuptiErr(d.getRes),
           d.pcs, d.samples, d.dropped);
    if (sample && d.samples == 0)
      bad++;
  }
  printf("%ld rounds, %.3f ms/round across %d device(s) (kernel ~2 ms)\n",
         rounds, wall * 1e3 / rounds, ndev);
  printf("%s\n", bad ? "FAIL: some device produced no samples" : "OK");
  return bad ? 1 : 0;
}
