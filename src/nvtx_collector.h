// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_NVTX_COLLECTOR_H_
#define PARCAGPU_NVTX_COLLECTOR_H_

#include <cstdint>
#include <cupti.h>

namespace parcagpu {

// NvtxCollector captures NVTX events from CUPTI_CB_DOMAIN_NVTX callbacks,
// batches them per-thread, and fires nvtx_event_batch USDT probes.
//
// Lifecycle is owned by CuptiProfiler. enable() sets NVTX_INJECTION64_PATH
// (so CUPTI's NVTX shim attaches) and turns on the CUPTI callback IDs we
// care about; onCallback() is the dispatch entry; disable() reverses it.
class NvtxCollector {
public:
  NvtxCollector() = default;
  ~NvtxCollector() = default;

  NvtxCollector(const NvtxCollector &) = delete;
  NvtxCollector &operator=(const NvtxCollector &) = delete;

  bool enable(CUpti_SubscriberHandle subscriber);
  void disable(CUpti_SubscriberHandle subscriber);

  // Called from CuptiProfiler::callbackHandler for domain==CUPTI_CB_DOMAIN_NVTX.
  // correlationId is the calling thread's most recent runtime ENTER correlation
  // (the existing thread-local `runtimeEnterCorrelationId` from cupti.cpp), or
  // 0 if no CUDA call is currently in flight on this thread.
  void onCallback(CUpti_CallbackId cbid, const void *cbdata,
                  uint32_t correlationId);

  // Flush the calling thread's pending batch. Called by CuptiProfiler::cleanup
  // (best-effort: only flushes the cleanup-running thread; per-thread
  // batches on other live threads drain via pthread_key destructor).
  void flushCurrentThread();
};

} // namespace parcagpu

#endif // PARCAGPU_NVTX_COLLECTOR_H_
