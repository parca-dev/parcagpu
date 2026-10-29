// relrace.cu — standalone reproducer for the PC sampling deadlock, no shim involved.
//
// cuptiPCSamplingEnable/Disable/Stop deadlock against another thread that is
// inside a kernel launch or async memcpy: in KERNEL_SERIALIZED mode CUPTI hooks
// those calls and takes its mutex under the driver lock, while the PC sampling
// call holds that mutex and waits for the driver lock. No cuptiSubscribe is
// needed for it. What holds: no other thread may be inside a driver API call
// while PC sampling changes state (--gate).
//
// App threads loop D2D copies and kernel launches on their own streams. A control
// thread cycles PC sampling: Enable+configure, Start, wait --window-ms, Stop,
// drain, Disable. --mode picks where those CUPTI calls run:
//
//   thread : on the control thread itself (what parcagpu's release thread does).
//   selfcb : on the control thread, but inside its own driver API callback: it
//            calls cuCtxGetCurrent and does the work in the ENTER callback.
//   app    : on an app thread, in its next driver API ENTER callback; the control
//            thread only posts the request and waits for it.
//   hybrid : Start and Stop+drain on an app thread as in "app"; Enable+configure
//            and Disable on the control thread (use with --gate).
//
// --keep-enabled Enables once at startup; cycles are then Start / Stop+drain.
// --continuous uses CONTINUOUS collection instead of KERNEL_SERIALIZED.
// --gate quiesces the app first: every driver API call holds a gate from its ENTER
// to its EXIT callback, and the CUPTI work runs only once the gate is closed and
// no other thread is inside a driver call. --gate-launch-only gates kernel and graph
// launches only (the calls parcagpu already has callbacks on); --gate-skip-sync
// leaves the synchronize calls ungated; --gate-narrow gates only Enable and Disable.
//
// Build: nvcc -O2 -std=c++17 -o relrace test/relrace.cu -lcuda -lcupti -lpthread
// Run:   ./relrace --mode thread --cycles 400            (hangs within a cycle or two)
//        ./relrace --mode hybrid --gate --cycles 1000    (completes)
// A watchdog aborts with "HANG" (exit 3) if app threads or the control thread make
// no progress for --hang-s; with RELRACE_GDB=<file> it first writes all thread
// stacks there with gdb. JSON lines on stdout.
//
// RTX PRO 6000 (sm_120), CUDA 13.0, 1000 cycles: every ungated mode except
// app-thread Start/Stop with --keep-enabled hangs, as do --continuous and
// --gate-launch-only; every --gate run completes, app threads waiting at most
// 19-45 ms for the gate to close and Enable/Disable taking up to 90 ms under it.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>

#include <getopt.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

extern "C" __global__ void spin_k(long long cycles, int *sink) {
  long long t0 = clock64();
  unsigned x = threadIdx.x + blockIdx.x * 3;
  while (clock64() - t0 < cycles) x = x * 1664525u + 1013904223u;
  if (x == 0x12345678u) *sink = (int)x;
}

static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static std::mutex g_logm;
static void logj(const char *ev, const char *fmt = nullptr, ...) {
  std::lock_guard<std::mutex> l(g_logm);
  printf("{\"ev\":\"%s\",\"ts\":%llu", ev, (unsigned long long)now_ns());
  if (fmt) {
    putchar(',');
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
  }
  puts("}");
  fflush(stdout);
}

struct Opts {
  std::string mode = "thread";
  int app_threads = 4;
  int cycles = 2000;
  double window_ms = 20;
  double gap_ms = 5;
  double hang_s = 15;
  bool subscribe = true;
  double kernel_us = 200;
  bool continuous = false;
  bool gate = false;
  bool gate_skip_sync = false;
  bool gate_narrow = false;  // gate only Enable+configure and Disable
  bool keep_enabled = false; // Enable once at startup; cycles are Start / Stop+drain
  bool gate_launch_only = false;
};
static Opts o;

static CUcontext g_ctx;
static CUpti_SubscriberHandle g_sub;

// ------------------------------------------------------------ PC sampling ops

static CUpti_PCSamplingData alloc_pcdata(size_t npcs, size_t nstall) {
  CUpti_PCSamplingData d{};
  d.size = sizeof(d);
  d.collectNumPcs = npcs;
  d.pPcData = (CUpti_PCSamplingPCData *)calloc(npcs, sizeof(CUpti_PCSamplingPCData));
  for (size_t i = 0; i < npcs; i++) {
    d.pPcData[i].size = sizeof(CUpti_PCSamplingPCData);
    d.pPcData[i].stallReason =
        (CUpti_PCSamplingStallReason *)calloc(nstall, sizeof(CUpti_PCSamplingStallReason));
  }
  return d;
}

static size_t g_nstall;
static uint32_t *g_stallIdx;
static CUpti_PCSamplingData g_cfgBuf, g_outBuf;

// Stall reasons are only reported for an enabled context: Enable, read them, Disable.
static void pcs_init_buffers() {
  CUpti_PCSamplingEnableParams ep{};
  ep.size = CUpti_PCSamplingEnableParamsSize;
  ep.ctx = g_ctx;
  cuptiPCSamplingEnable(&ep);
  CUpti_PCSamplingGetNumStallReasonsParams np{};
  np.size = CUpti_PCSamplingGetNumStallReasonsParamsSize;
  np.ctx = g_ctx;
  np.numStallReasons = &g_nstall;
  cuptiPCSamplingGetNumStallReasons(&np);
  g_stallIdx = (uint32_t *)calloc(g_nstall, sizeof(uint32_t));
  char **names = (char **)calloc(g_nstall, sizeof(char *));
  for (size_t i = 0; i < g_nstall; i++) names[i] = (char *)calloc(CUPTI_STALL_REASON_STRING_SIZE, 1);
  CUpti_PCSamplingGetStallReasonsParams sp{};
  sp.size = CUpti_PCSamplingGetStallReasonsParamsSize;
  sp.ctx = g_ctx;
  sp.numStallReasons = g_nstall;
  sp.stallReasonIndex = g_stallIdx;
  sp.stallReasons = names;
  cuptiPCSamplingGetStallReasons(&sp);
  g_cfgBuf = alloc_pcdata(1024, g_nstall);
  g_outBuf = alloc_pcdata(4096, g_nstall);
  CUpti_PCSamplingDisableParams dp{};
  dp.size = CUpti_PCSamplingDisableParamsSize;
  dp.ctx = g_ctx;
  cuptiPCSamplingDisable(&dp);
  fprintf(stderr, "stall reasons: %zu\n", g_nstall);
}

static CUptiResult pcs_enable() {
  CUpti_PCSamplingEnableParams ep{};
  ep.size = CUpti_PCSamplingEnableParamsSize;
  ep.ctx = g_ctx;
  CUptiResult r = cuptiPCSamplingEnable(&ep);
  if (r != CUPTI_SUCCESS) {
    logj("enable_err", "\"rc\":%d", r);
    return r;
  }
  CUpti_PCSamplingConfigurationInfo cfg[5] = {};
  cfg[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  cfg[0].attributeData.stallReasonData.stallReasonCount = g_nstall;
  cfg[0].attributeData.stallReasonData.pStallReasonIndex = g_stallIdx;
  cfg[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  cfg[1].attributeData.collectionModeData.collectionMode =
      o.continuous ? CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS
                   : CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
  cfg[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
  cfg[2].attributeData.enableStartStopControlData.enableStartStopControl = 1;
  cfg[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  cfg[3].attributeData.samplingDataBufferData.samplingDataBuffer = &g_cfgBuf;
  cfg[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
  cfg[4].attributeData.scratchBufferSizeData.scratchBufferSize = 16u << 20;
  // One attribute per call, so a rejected one is named in the log.
  for (int i = 0; i < 5; i++) {
    CUpti_PCSamplingConfigurationInfoParams cp{};
    cp.size = CUpti_PCSamplingConfigurationInfoParamsSize;
    cp.ctx = g_ctx;
    cp.numAttributes = 1;
    cp.pPCSamplingConfigurationInfo = &cfg[i];
    r = cuptiPCSamplingSetConfigurationAttribute(&cp);
    if (r != CUPTI_SUCCESS) {
      logj("cfg_err", "\"attr\":%d,\"rc\":%d", (int)cfg[i].attributeType, r);
      return r;
    }
  }
  return CUPTI_SUCCESS;
}

static CUptiResult pcs_start() {
  CUpti_PCSamplingStartParams p{};
  p.size = CUpti_PCSamplingStartParamsSize;
  p.ctx = g_ctx;
  return cuptiPCSamplingStart(&p);
}

static CUptiResult pcs_stop_drain(uint64_t *samples) {
  CUpti_PCSamplingStopParams sp{};
  sp.size = CUpti_PCSamplingStopParamsSize;
  sp.ctx = g_ctx;
  CUptiResult r = cuptiPCSamplingStop(&sp);
  for (int g = 0; g < 1000; g++) {
    g_outBuf.totalNumPcs = g_outBuf.remainingNumPcs = 0;
    CUpti_PCSamplingGetDataParams gp{};
    gp.size = CUpti_PCSamplingGetDataParamsSize;
    gp.ctx = g_ctx;
    gp.pcSamplingData = &g_outBuf;
    if (cuptiPCSamplingGetData(&gp) != CUPTI_SUCCESS) break;
    for (size_t i = 0; i < g_outBuf.totalNumPcs; i++)
      for (size_t j = 0; j < g_outBuf.pPcData[i].stallReasonCount; j++)
        *samples += g_outBuf.pPcData[i].stallReason[j].samples;
    if (g_outBuf.remainingNumPcs == 0) break;
  }
  return r;
}

static CUptiResult pcs_disable() {
  CUpti_PCSamplingDisableParams dp{};
  dp.size = CUpti_PCSamplingDisableParamsSize;
  dp.ctx = g_ctx;
  return cuptiPCSamplingDisable(&dp);
}

// ------------------------------------------------------------ requests

enum Op { OP_NONE = 0, OP_ON, OP_OFF, OP_START, OP_STOP };
static std::atomic<int> g_req{OP_NONE};   // posted by the control thread
static std::atomic<int> g_done{OP_NONE};  // set by whoever performed it
static std::atomic<int> g_rc{0};
static std::atomic<uint64_t> g_samples{0};
static std::mutex g_opm;
static thread_local bool t_in_op;
static thread_local bool t_is_ctl;

// ------------------------------------------------------------ gate

// Writer-preferring: once closing, new driver calls wait at ENTER, and the closer
// waits for calls already inside to reach EXIT.
static std::mutex g_gm;
static std::condition_variable g_gcv;
static bool g_closing;
static int g_inside;
static thread_local int t_depth;

static void gate_enter() {
  if (t_depth++ > 0) return;
  std::unique_lock<std::mutex> l(g_gm);
  g_gcv.wait(l, [] { return !g_closing; });
  g_inside++;
}

static void gate_exit() {
  if (--t_depth > 0) return;
  std::lock_guard<std::mutex> l(g_gm);
  if (--g_inside <= 1) g_gcv.notify_all();
}

// A caller inside a gated driver call (an app thread in its callback) counts itself.
static bool gate_close(uint64_t timeout_ns) {
  std::unique_lock<std::mutex> l(g_gm);
  g_closing = true;
  int self = t_depth > 0 ? 1 : 0;
  if (!g_gcv.wait_for(l, std::chrono::nanoseconds(timeout_ns), [&] { return g_inside == self; })) {
    g_closing = false;
    g_gcv.notify_all();
    return false;
  }
  return true;
}

static void gate_open() {
  std::lock_guard<std::mutex> l(g_gm);
  g_closing = false;
  g_gcv.notify_all();
}

static std::atomic<uint64_t> g_gate_waits_ns{0}, g_gate_timeouts{0}, g_gate_wait_max_ns{0},
    g_op_max_ns{0};

static bool is_sync(CUpti_CallbackId id) {
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
      return true;
  }
  return false;
}

static bool is_launch(CUpti_CallbackId id) {
  switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
      return true;
  }
  return false;
}

static void atomic_max(std::atomic<uint64_t> &a, uint64_t v) {
  uint64_t c = a.load();
  while (v > c && !a.compare_exchange_weak(c, v)) {
  }
}

// Performs the posted request, if any, on the calling thread.
// Runs fn with the gate closed (when --gate). False if the gate didn't close in time.
template <typename F>
static bool gated(F fn) {
  uint64_t g0 = now_ns();
  if (o.gate && !gate_close(2000000000ull)) {
    g_gate_timeouts++;
    return false;
  }
  uint64_t g1 = now_ns();
  g_gate_waits_ns += g1 - g0;
  atomic_max(g_gate_wait_max_ns, g1 - g0);
  fn();
  atomic_max(g_op_max_ns, now_ns() - g1);
  if (o.gate) gate_open();
  return true;
}

static void perform_pending() {
  int op = g_req.load(std::memory_order_acquire);
  if (op == OP_NONE || t_in_op) return;
  std::unique_lock<std::mutex> l(g_opm, std::try_to_lock);
  if (!l.owns_lock()) return;
  op = g_req.load(std::memory_order_acquire);
  if (op == OP_NONE) return;
  t_in_op = true;
  CUptiResult r = CUPTI_SUCCESS;
  bool ok = true;
  if (op == OP_START) {
    ok = gated([&] { r = pcs_start(); });
  } else if (op == OP_STOP) {
    ok = gated([&] {
      uint64_t s = 0;
      r = pcs_stop_drain(&s);
      g_samples += s;
    });
  } else if (o.keep_enabled && op == OP_ON) {
    ok = gated([&] { r = pcs_start(); });
  } else if (o.keep_enabled) {
    ok = gated([&] {
      uint64_t s = 0;
      r = pcs_stop_drain(&s);
      g_samples += s;
    });
  } else if (op == OP_ON && o.gate_narrow) {
    ok = gated([&] { r = pcs_enable(); });
    if (ok && r == CUPTI_SUCCESS) r = pcs_start();
  } else if (op == OP_ON) {
    ok = gated([&] {
      r = pcs_enable();
      if (r == CUPTI_SUCCESS) r = pcs_start();
    });
  } else if (o.gate_narrow) {
    // Stop+drain once; a retry after a gate timeout only redoes the Disable.
    static thread_local bool stopped;
    if (!stopped) {
      uint64_t s = 0;
      r = pcs_stop_drain(&s);
      g_samples += s;
      stopped = true;
    }
    ok = gated([&] { r = pcs_disable(); });
    if (ok) stopped = false;
  } else {
    ok = gated([&] {
      uint64_t s = 0;
      r = pcs_stop_drain(&s);
      CUptiResult rd = pcs_disable();
      if (r == CUPTI_SUCCESS) r = rd;
      g_samples += s;
    });
  }
  t_in_op = false;
  if (!ok) return;  // retried by the caller
  g_rc = r;
  g_req = OP_NONE;
  g_done = op;
}

static void CUPTIAPI callback(void *, CUpti_CallbackDomain dom, CUpti_CallbackId id, const void *data) {
  if (dom != CUPTI_CB_DOMAIN_DRIVER_API) return;
  auto *cb = (const CUpti_CallbackData *)data;
  if (o.gate && !t_in_op && !(o.gate_skip_sync && is_sync(id)) &&
      !(o.gate_launch_only && !is_launch(id))) {
    if (cb->callbackSite == CUPTI_API_ENTER) gate_enter();
    else gate_exit();
  }
  if (cb->callbackSite != CUPTI_API_ENTER) return;
  if ((o.mode == "app" || o.mode == "hybrid") && !t_is_ctl) perform_pending();
  if (o.mode == "selfcb" && t_is_ctl) perform_pending();
}

// ------------------------------------------------------------ threads

static std::atomic<bool> g_stop{false};
static std::vector<std::atomic<uint64_t>> *g_progress;
static std::atomic<uint64_t> g_cycles{0};
static std::atomic<uint64_t> g_ctl_beat{0};

static void *app_main(void *arg) {
  int idx = (int)(intptr_t)arg;
  cuCtxSetCurrent(g_ctx);
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  const size_t n = 1 << 20;
  void *a, *b;
  int *sink;
  cudaMalloc(&a, n);
  cudaMalloc(&b, n);
  cudaMalloc(&sink, sizeof(int));
  int clockKHz = 0;
  cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
  long long cycles = (long long)(o.kernel_us * clockKHz / 1000.0);
  uint64_t i = 0;
  while (!g_stop) {
    cudaMemcpyAsync(b, a, n, cudaMemcpyDeviceToDevice, s);
    spin_k<<<8, 128, 0, s>>>(cycles, sink);
    if (++i % 8 == 0) cudaStreamSynchronize(s);
    (*g_progress)[idx]++;
  }
  cudaStreamSynchronize(s);
  return nullptr;
}

static bool wait_done(int op) {
  uint64_t t = now_ns();
  while (g_done.load(std::memory_order_acquire) != op) {
    if (g_stop) return false;
    if (o.mode == "selfcb") {
      CUcontext c;
      cuCtxGetCurrent(&c);  // its ENTER callback performs the request
    } else {
      usleep(200);
    }
    g_ctl_beat++;
    if (now_ns() - t > 30ull * 1000000000ull) return false;
  }
  return true;
}

static void request(int op) {
  g_done = OP_NONE;
  if (o.mode == "thread") {
    g_req = op;
    while (g_req.load() != OP_NONE && !g_stop) perform_pending();
    return;
  }
  g_req = op;
}

static void *ctl_main(void *) {
  t_is_ctl = true;
  pthread_setname_np(pthread_self(), "relrace-ctl");
  cuCtxSetCurrent(g_ctx);
  uint64_t fails = 0;
  for (int c = 0; c < o.cycles && !g_stop && o.mode == "hybrid"; c++) {
    CUptiResult r = CUPTI_SUCCESS;
    t_in_op = true;
    while (!g_stop && !gated([&] { r = pcs_enable(); })) {
    }
    t_in_op = false;
    if (r != CUPTI_SUCCESS) logj("op_err", "\"op\":\"enable\",\"rc\":%d,\"cycle\":%d", (int)r, c);
    request(OP_START);
    if (!wait_done(OP_START)) { logj("ctl_timeout", "\"op\":\"start\",\"cycle\":%d", c); fails++; continue; }
    usleep((useconds_t)(o.window_ms * 1000));
    request(OP_STOP);
    if (!wait_done(OP_STOP)) { logj("ctl_timeout", "\"op\":\"stop\",\"cycle\":%d", c); fails++; continue; }
    t_in_op = true;
    while (!g_stop && !gated([&] { r = pcs_disable(); })) {
    }
    t_in_op = false;
    if (r != CUPTI_SUCCESS) logj("op_err", "\"op\":\"disable\",\"rc\":%d,\"cycle\":%d", (int)r, c);
    g_cycles++;
    g_ctl_beat++;
    usleep((useconds_t)(o.gap_ms * 1000));
  }
  for (int c = 0; c < o.cycles && !g_stop && o.mode != "hybrid"; c++) {
    request(OP_ON);
    if (!wait_done(OP_ON)) { logj("ctl_timeout", "\"op\":\"on\",\"cycle\":%d", c); fails++; continue; }
    if (g_rc != CUPTI_SUCCESS) logj("op_err", "\"op\":\"on\",\"rc\":%d,\"cycle\":%d", (int)g_rc, c);
    usleep((useconds_t)(o.window_ms * 1000));
    request(OP_OFF);
    if (!wait_done(OP_OFF)) { logj("ctl_timeout", "\"op\":\"off\",\"cycle\":%d", c); fails++; continue; }
    if (g_rc != CUPTI_SUCCESS) logj("op_err", "\"op\":\"off\",\"rc\":%d,\"cycle\":%d", (int)g_rc, c);
    g_cycles++;
    g_ctl_beat++;
    usleep((useconds_t)(o.gap_ms * 1000));
  }
  logj("ctl_done", "\"cycles\":%llu,\"timeouts\":%llu,\"gate_wait_ms\":%.1f,\"gate_timeouts\":%llu,\"gate_wait_max_ms\":%.1f,\"op_max_ms\":%.1f",
       (unsigned long long)g_cycles.load(), (unsigned long long)fails, g_gate_waits_ns.load() / 1e6,
       (unsigned long long)g_gate_timeouts.load(), g_gate_wait_max_ns.load() / 1e6,
       g_op_max_ns.load() / 1e6);
  g_stop = true;
  return nullptr;
}

static void *watchdog_main(void *) {
  std::vector<uint64_t> last(o.app_threads, 0);
  uint64_t last_beat = 0, last_change = now_ns(), last_report = now_ns();
  while (!g_stop) {
    usleep(500000);
    bool moved = true;
    uint64_t total = 0;
    for (int i = 0; i < o.app_threads; i++) {
      uint64_t p = (*g_progress)[i];
      if (p == last[i]) moved = false;
      last[i] = p;
      total += p;
    }
    uint64_t beat = g_ctl_beat;
    if (beat == last_beat) moved = false;
    last_beat = beat;
    if (moved) last_change = now_ns();
    if (now_ns() - last_report > 5000000000ull) {
      logj("progress", "\"cycles\":%llu,\"app_iters\":%llu,\"samples\":%llu",
           (unsigned long long)g_cycles.load(), (unsigned long long)total,
           (unsigned long long)g_samples.load());
      last_report = now_ns();
    }
    if (now_ns() - last_change > (uint64_t)(o.hang_s * 1e9)) {
      std::string per;
      for (int i = 0; i < o.app_threads; i++) per += std::to_string(last[i]) + " ";
      logj("HANG", "\"mode\":\"%s\",\"cycles\":%llu,\"app_iters\":\"%s\",\"pending\":%d",
           o.mode.c_str(), (unsigned long long)g_cycles.load(), per.c_str(), g_req.load());
      fflush(stdout);
      if (getenv("RELRACE_GDB")) {
        char cmd[256];
        snprintf(cmd, sizeof cmd, "gdb -p %d -batch -ex 'thread apply all bt 40' > %s 2>&1", getpid(),
                 getenv("RELRACE_GDB"));
        if (system(cmd) != 0) fprintf(stderr, "gdb failed\n");
      }
      _exit(3);
    }
  }
  return nullptr;
}

int main(int argc, char **argv) {
  static struct option lo[] = {{"mode", 1, 0, 'm'},   {"app-threads", 1, 0, 'a'},
                               {"cycles", 1, 0, 'c'}, {"window-ms", 1, 0, 'w'},
                               {"gap-ms", 1, 0, 'g'}, {"hang-s", 1, 0, 'h'},
                               {"no-subscribe", 0, 0, 'n'}, {"kernel-us", 1, 0, 'k'},
                               {"continuous", 0, 0, 'C'}, {"gate", 0, 0, 'G'},
                               {"gate-skip-sync", 0, 0, 'S'}, {"gate-narrow", 0, 0, 'N'},
                               {"keep-enabled", 0, 0, 'K'},
                               {"gate-launch-only", 0, 0, 'L'},
                               {0, 0, 0, 0}};
  int ch;
  while ((ch = getopt_long(argc, argv, "", lo, nullptr)) != -1) {
    switch (ch) {
      case 'm': o.mode = optarg; break;
      case 'a': o.app_threads = atoi(optarg); break;
      case 'c': o.cycles = atoi(optarg); break;
      case 'w': o.window_ms = atof(optarg); break;
      case 'g': o.gap_ms = atof(optarg); break;
      case 'h': o.hang_s = atof(optarg); break;
      case 'n': o.subscribe = false; break;
      case 'k': o.kernel_us = atof(optarg); break;
      case 'C': o.continuous = true; break;
      case 'G': o.gate = true; break;
      case 'S': o.gate = o.gate_skip_sync = true; break;
      case 'N': o.gate = o.gate_narrow = true; break;
      case 'K': o.keep_enabled = true; break;
      case 'L': o.gate = o.gate_launch_only = true; break;
      default: return 2;
    }
  }
  if (o.mode != "thread" && o.mode != "selfcb" && o.mode != "app" && o.mode != "hybrid") return 2;
  if ((o.mode != "thread" || o.gate) && !o.subscribe) return 2;

  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&g_ctx, dev);
  cuCtxSetCurrent(g_ctx);
  cudaFree(0);
  if (o.subscribe) {
    cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)callback, nullptr);
    cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API);
  }
  pcs_init_buffers();
  if (o.keep_enabled && pcs_enable() != CUPTI_SUCCESS) return 1;
  logj("start", "\"mode\":\"%s\",\"app_threads\":%d,\"cycles\":%d,\"subscribe\":%d", o.mode.c_str(),
       o.app_threads, o.cycles, o.subscribe);

  g_progress = new std::vector<std::atomic<uint64_t>>(o.app_threads);
  std::vector<pthread_t> apps(o.app_threads);
  for (int i = 0; i < o.app_threads; i++) pthread_create(&apps[i], nullptr, app_main, (void *)(intptr_t)i);
  pthread_t ctl, wd;
  pthread_create(&wd, nullptr, watchdog_main, nullptr);
  pthread_create(&ctl, nullptr, ctl_main, nullptr);
  pthread_join(ctl, nullptr);
  for (auto &t : apps) pthread_join(t, nullptr);
  uint64_t total = 0;
  for (auto &p : *g_progress) total += p;
  logj("done", "\"mode\":\"%s\",\"cycles\":%llu,\"app_iters\":%llu,\"samples\":%llu", o.mode.c_str(),
       (unsigned long long)g_cycles.load(), (unsigned long long)total,
       (unsigned long long)g_samples.load());
  return 0;
}
