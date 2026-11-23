// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENT, see event_timing.h.

#include "event_timing.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <deque>
#include <mutex>
#include <string>
#include <time.h>
#include <unordered_set>
#include <vector>

#include "probes.h"

#include "Driver/Dispatch.h"
#include "Driver/GPU/CuptiApi.h"

namespace proton::cuda {
struct ExternLibCudaEvents : public ExternLibBase {
  using RetType = CUresult;
  static constexpr const char *name = "libcuda.so.1";
  static constexpr RetType success = CUDA_SUCCESS;
  static inline void *lib = nullptr;
};
template <bool C> CUresult evCreate(CUevent *, unsigned int);
template <bool C> CUresult evRecord(CUevent, CUstream);
template <bool C> CUresult evQuery(CUevent);
template <bool C> CUresult evElapsed(float *, CUevent, CUevent);
DEFINE_DISPATCH(ExternLibCudaEvents, evCreate, cuEventCreate, CUevent *,
                unsigned int)
DEFINE_DISPATCH(ExternLibCudaEvents, evRecord, cuEventRecord, CUevent,
                CUstream)
DEFINE_DISPATCH(ExternLibCudaEvents, evQuery, cuEventQuery, CUevent)
DEFINE_DISPATCH(ExternLibCudaEvents, evElapsed, cuEventElapsedTime,
                float *, CUevent, CUevent)
} // namespace proton::cuda

namespace proton::cupti {
template <bool C> CUptiResult getTimestampEv(uint64_t *);
DEFINE_DISPATCH(ExternLibCupti, getTimestampEv, cuptiGetTimestamp, uint64_t *)
} // namespace proton::cupti

namespace parcagpu::event_timing {

namespace {

int g_mode = 0;
double g_p = 0.05;

struct Pair {
  CUcontext ctx = nullptr;
  CUevent start = nullptr;
  CUevent end = nullptr;
};

struct Pending {
  Pair ev;
  uint32_t correlationId;
  uint32_t deviceId;
  bool graph;
  const char *name; // CUPTI symbol name; stable for the function's lifetime
  uint64_t cpuExitNs;
};

std::mutex g_mu;
std::deque<Pending> g_pending;      // recorded, not yet complete
std::vector<Pair> g_free;           // reusable pairs (per context)
std::unordered_set<uint32_t> g_cmp; // mode 2: correlation ids event-timed
std::atomic<uint64_t> g_timed{0}, g_dropped{0};

thread_local Pair t_pair;
thread_local CUstream t_stream = nullptr;
thread_local uint32_t t_corr = 0;
thread_local uint32_t t_dev = 0;
thread_local bool t_graph = false;
thread_local const char *t_name = nullptr;
thread_local uint32_t t_sampledCorr = 0;

uint64_t cuptiNow() {
  uint64_t t = 0;
  proton::cupti::getTimestampEv<false>(&t);
  return t;
}

bool streamOf(const CUpti_CallbackData *cb, CUpti_CallbackId cbid,
              CUstream *out) {
  const void *p = cb->functionParams;
  switch (cbid) {
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
    *out = static_cast<const cuLaunchKernel_params *>(p)->hStream;
    return true;
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
    *out = static_cast<const cuLaunchCooperativeKernel_params *>(p)->hStream;
    return true;
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
    const CUlaunchConfig *c =
        static_cast<const cuLaunchKernelEx_params *>(p)->config;
    if (!c)
      return false;
    *out = c->hStream;
    return true;
  }
  case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
  case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
    *out = static_cast<const cuGraphLaunch_params *>(p)->hStream;
    return true;
  default:
    return false;
  }
}

bool getPair(CUcontext ctx, Pair *out) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    for (size_t i = 0; i < g_free.size(); i++) {
      if (g_free[i].ctx == ctx) {
        *out = g_free[i];
        g_free[i] = g_free.back();
        g_free.pop_back();
        return true;
      }
    }
  }
  // Context is current inside the launch callback.
  Pair p{ctx, nullptr, nullptr};
  if (proton::cuda::evCreate<false>(&p.start, CU_EVENT_DEFAULT) !=
      CUDA_SUCCESS)
    return false;
  if (proton::cuda::evCreate<false>(&p.end, CU_EVENT_DEFAULT) !=
      CUDA_SUCCESS)
    return false; // leaks one event; experiment
  *out = p;
  return true;
}

} // namespace

void init() {
  if (const char *v = getenv("PARCAGPU_EVENT_TIMING"))
    g_mode = atoi(v);
  if (const char *v = getenv("PARCAGPU_EVENT_SAMPLE_P")) {
    double p = atof(v);
    if (p > 0 && p <= 1)
      g_p = p;
  }
  if (g_mode)
    fprintf(stderr, "[PARCAGPU] EXPERIMENT event timing mode=%d p=%g\n",
            g_mode, g_p);
}

int mode() { return g_mode; }
double sampleP() { return g_p; }

bool begin(const CUpti_CallbackData *cb, CUpti_CallbackId cbid,
           uint32_t deviceId) {
  CUstream s;
  if (!streamOf(cb, cbid, &s))
    return false;
  Pair p;
  if (!getPair(cb->context, &p))
    return false;
  if (proton::cuda::evRecord<false>(p.start, s) != CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_free.push_back(p);
    return false;
  }
  t_pair = p;
  t_stream = s;
  t_corr = cb->correlationId;
  t_dev = deviceId;
  t_graph = cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
            cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
  t_name = cb->symbolName ? cb->symbolName : cb->functionName;
  return true;
}

void end(const CUpti_CallbackData *cb) {
  if (t_corr == 0 || cb->correlationId != t_corr)
    return;
  const uint32_t corr = t_corr;
  t_corr = 0;
  bool launched = *static_cast<const CUresult *>(cb->functionReturnValue) ==
                  CUDA_SUCCESS;
  if (!launched ||
      proton::cuda::evRecord<false>(t_pair.end, t_stream) != CUDA_SUCCESS) {
    // The start event still completes; just recycle after it does. Simplest:
    // drop the pair (experiment).
    g_dropped++;
    return;
  }
  t_sampledCorr = corr;
  std::lock_guard<std::mutex> lk(g_mu);
  g_pending.push_back(
      {t_pair, corr, t_dev, t_graph, t_name, 0});
  if (g_mode == 2) {
    if (g_cmp.size() > 20000)
      g_cmp.clear();
    g_cmp.insert(corr);
  }
}

uint32_t pendingCorrelation() { return t_sampledCorr; }
void consumeCorrelation() { t_sampledCorr = 0; }

void reap() {
  std::unique_lock<std::mutex> lk(g_mu, std::try_to_lock);
  if (!lk.owns_lock())
    return;
  // In order per stream mostly; stop at the first incomplete one, but look a
  // few deep since entries come from several streams.
  int looked = 0;
  for (auto it = g_pending.begin(); it != g_pending.end() && looked < 8;) {
    looked++;
    CUresult q = proton::cuda::evQuery<false>(it->ev.end);
    if (q == CUDA_ERROR_NOT_READY) {
      ++it;
      continue;
    }
    float ms = 0;
    bool ok = q == CUDA_SUCCESS &&
              proton::cuda::evElapsed<false>(&ms, it->ev.start, it->ev.end) ==
                  CUDA_SUCCESS;
    if (ok) {
      uint64_t durNs = (uint64_t)((double)ms * 1e6);
      g_timed++;
      if (g_mode == 1) {
        // End is "about now" (when we noticed); start = end - duration.
        uint64_t endNs = cuptiNow();
        PARCAGPU_KERNEL_EXECUTED(endNs - durNs, endNs, it->correlationId,
                                 it->deviceId, 0, it->graph ? 1 : 0, 0,
                                 it->name);
      } else if (g_mode == 2) {
        fprintf(stderr, "PGEV E %u %llu %d %s\n", it->correlationId,
                (unsigned long long)durNs, it->graph ? 1 : 0,
                it->name ? it->name : "?");
      }
    } else {
      g_dropped++;
    }
    g_free.push_back(it->ev);
    it = g_pending.erase(it);
  }
}

void logCuptiRecord(uint32_t correlationId, uint64_t start, uint64_t end,
                    uint32_t graphId) {
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_cmp.count(correlationId))
      return;
  }
  fprintf(stderr, "PGEV C %u %llu %llu %u\n", correlationId,
          (unsigned long long)start, (unsigned long long)end, graphId);
}

} // namespace parcagpu::event_timing
