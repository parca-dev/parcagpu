// pc_probe.cu — scripted CUPTI PC sampling, no parcagpu involved, to find out
// how PC sampling behaves across processes and devices (test/pc-matrix.sh).
//
// Keeps every chosen device busy with ~2 ms kernels and applies a script of
// timed PC sampling actions, printing JSON lines: one per action with its
// CUPTI result, and one per device per second with the samples drained.
//
//   pc_probe --devs=0,1 --secs=10 --script=enable@0,start@0,stop@5:1,disable@6
//
// Actions: enable (enable + configure), start, stop (stop + drain), disable.
// "@T" is seconds from start; ":D" limits the action to device D (default:
// all of this process's devices). PC_PROBE_PERIOD=n sets a 2^n-cycle period.
//
// Build: nvcc -O2 -lineinfo -o pc_probe test/pc_probe.cu -lcuda -lcupti

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_pcsampling.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
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
  char uuid[48];
  CUcontext ctx{};
  CUstream stream{};
  unsigned long long *out{};
  long long cycles{};
  size_t nStall{};
  CUpti_PCSamplingData cfgBuf{}, outBuf{};
  bool enabled{}, started{};
  // This second's counters.
  unsigned long long samples{}, pcs{}, dropped{}, rounds{};
  double maxRoundMs{};
  CUptiResult getErr{CUPTI_SUCCESS};
};

struct Action {
  std::string what;
  double at;
  int dev; // -1 = all
  bool done{};
};

static int g_pid;
static double g_t0;

static void event(const Dev &d, const char *what, CUptiResult r) {
  printf("{\"pid\":%d,\"t\":%.2f,\"dev\":%d,\"uuid\":\"%s\",\"action\":\"%s\","
         "\"res\":\"%s\"}\n",
         g_pid, now() - g_t0, d.ordinal, d.uuid, what, cuptiErr(r));
  fflush(stdout);
}

static void drain(Dev &d) {
  CUpti_PCSamplingGetDataParams p{};
  p.size = CUpti_PCSamplingGetDataParamsSize;
  p.ctx = d.ctx;
  p.pcSamplingData = &d.outBuf;
  for (int i = 0; i < 100000; i++) {
    d.outBuf.totalNumPcs = 0;
    CUptiResult r = cuptiPCSamplingGetData(&p);
    if (r != CUPTI_SUCCESS) {
      d.getErr = r;
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

static CUptiResult enable(Dev &d) {
  CUpti_PCSamplingEnableParams en{};
  en.size = CUpti_PCSamplingEnableParamsSize;
  en.ctx = d.ctx;
  CUptiResult r = cuptiPCSamplingEnable(&en);
  if (r != CUPTI_SUCCESS)
    return r;

  CUpti_PCSamplingGetNumStallReasonsParams nsp{};
  nsp.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
  nsp.ctx = d.ctx;
  nsp.numStallReasons = &d.nStall;
  if ((r = cuptiPCSamplingGetNumStallReasons(&nsp)) != CUPTI_SUCCESS)
    return r;
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
  if ((r = cuptiPCSamplingGetStallReasons(&srp)) != CUPTI_SUCCESS)
    return r;

  if (!d.cfgBuf.pPcData) {
    d.cfgBuf = allocData(1024, d.nStall);
    d.outBuf = allocData(1024, d.nStall);
  }
  // As parcagpu configures it.
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
  info[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  info[4].attributeData.scratchBufferSizeData.scratchBufferSize = 16 << 20;
  CUpti_PCSamplingConfigurationInfoParams cp{};
  cp.size = CUpti_PCSamplingConfigurationInfoParamsSize;
  cp.ctx = d.ctx;
  cp.numAttributes = 5;
  if (const char *per = getenv("PC_PROBE_PERIOD")) {
    info[5].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    info[5].attributeData.samplingPeriodData.samplingPeriod = atoi(per);
    cp.numAttributes = 6;
  }
  cp.pPCSamplingConfigurationInfo = info;
  return cuptiPCSamplingSetConfigurationAttribute(&cp);
}

static void act(Dev &d, const std::string &what) {
  CU_CHECK(cuCtxSetCurrent(d.ctx));
  CUptiResult r = CUPTI_SUCCESS;
  if (what == "enable") {
    r = enable(d);
    d.enabled = r == CUPTI_SUCCESS;
  } else if (what == "start") {
    CU_CHECK(cuCtxSynchronize()); // as parcagpu does
    CUpti_PCSamplingStartParams sp{};
    sp.size = CUpti_PCSamplingStartParamsSize;
    sp.ctx = d.ctx;
    r = cuptiPCSamplingStart(&sp);
    d.started = r == CUPTI_SUCCESS;
  } else if (what == "stop") {
    CUpti_PCSamplingStopParams st{};
    st.size = CUpti_PCSamplingStopParamsSize;
    st.ctx = d.ctx;
    r = cuptiPCSamplingStop(&st);
    d.started = false;
    drain(d);
  } else if (what == "disable") {
    CUpti_PCSamplingDisableParams dp{};
    dp.size = CUpti_PCSamplingDisableParamsSize;
    dp.ctx = d.ctx;
    r = cuptiPCSamplingDisable(&dp);
    d.enabled = d.started = false;
  } else {
    fprintf(stderr, "unknown action %s\n", what.c_str());
    exit(2);
  }
  event(d, what.c_str(), r);
}

int main(int argc, char **argv) {
  std::string devList, script;
  double secs = 10;
  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "--devs=", 7))
      devList = argv[i] + 7;
    else if (!strncmp(argv[i], "--secs=", 7))
      secs = atof(argv[i] + 7);
    else if (!strncmp(argv[i], "--script=", 9))
      script = argv[i] + 9;
    else {
      fprintf(stderr, "usage: %s [--devs=0,1] [--secs=10] [--script=enable@0,start@0,...]\n",
              argv[0]);
      return 2;
    }
  }
  g_pid = getpid();
  CU_CHECK(cuInit(0));
  int total = 0;
  CU_CHECK(cuDeviceGetCount(&total));

  std::vector<Dev> devs;
  if (devList.empty())
    for (int i = 0; i < total; i++)
      devs.push_back(Dev{i});
  else
    for (char *s = strtok(&devList[0], ","); s; s = strtok(nullptr, ","))
      devs.push_back(Dev{atoi(s)});

  std::vector<Action> actions;
  for (char *s = strtok(&script[0], ","); s; s = strtok(nullptr, ",")) {
    Action a;
    char *at = strchr(s, '@');
    if (!at) {
      fprintf(stderr, "bad action %s\n", s);
      return 2;
    }
    a.what.assign(s, at - s);
    char *colon = strchr(at, ':');
    a.at = atof(at + 1);
    a.dev = colon ? atoi(colon + 1) : -1;
    actions.push_back(a);
  }

  for (auto &d : devs) {
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, d.ordinal));
    CUuuid u;
    CU_CHECK(cuDeviceGetUuid(&u, dev));
    const unsigned char *b = (const unsigned char *)u.bytes;
    snprintf(d.uuid, sizeof d.uuid,
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10],
             b[11], b[12], b[13], b[14], b[15]);
    CU_CHECK(cuDevicePrimaryCtxRetain(&d.ctx, dev));
    CU_CHECK(cuCtxSetCurrent(d.ctx));
    CU_CHECK(cuStreamCreate(&d.stream, CU_STREAM_NON_BLOCKING));
    CU_CHECK(cuMemAlloc((CUdeviceptr *)&d.out, sizeof(*d.out)));
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d.ordinal);
    d.cycles = (long long)(0.002 * clockKHz * 1000.0);
  }

  g_t0 = now();
  double nextReport = 1;
  while (now() - g_t0 < secs) {
    double t = now() - g_t0;
    for (auto &a : actions) {
      if (a.done || t < a.at)
        continue;
      a.done = true;
      for (auto &d : devs)
        if (a.dev < 0 || a.dev == d.ordinal)
          act(d, a.what);
    }
    double r0 = now();
    for (auto &d : devs) {
      CU_CHECK(cuCtxSetCurrent(d.ctx));
      spin<<<1024, 256, 0, (cudaStream_t)d.stream>>>(d.cycles, d.out);
    }
    for (auto &d : devs) {
      CU_CHECK(cuCtxSetCurrent(d.ctx));
      CU_CHECK(cuStreamSynchronize(d.stream));
      if (d.started)
        drain(d);
      double ms = (now() - r0) * 1e3;
      d.rounds++;
      if (ms > d.maxRoundMs)
        d.maxRoundMs = ms;
    }
    if (t >= nextReport) {
      for (auto &d : devs) {
        printf("{\"pid\":%d,\"t\":%.0f,\"dev\":%d,\"uuid\":\"%s\",\"enabled\":%d,"
               "\"started\":%d,\"samples\":%llu,\"pcs\":%llu,\"dropped\":%llu,"
               "\"rounds\":%llu,\"max_round_ms\":%.1f,\"getdata\":\"%s\"}\n",
               g_pid, nextReport, d.ordinal, d.uuid, d.enabled, d.started,
               d.samples, d.pcs, d.dropped, d.rounds, d.maxRoundMs,
               cuptiErr(d.getErr));
        d.samples = d.pcs = d.dropped = d.rounds = 0;
        d.maxRoundMs = 0;
        d.getErr = CUPTI_SUCCESS;
      }
      fflush(stdout);
      nextReport += 1;
    }
  }
  for (auto &d : devs) {
    if (d.started)
      act(d, "stop");
    if (d.enabled)
      act(d, "disable");
  }
  return 0;
}
