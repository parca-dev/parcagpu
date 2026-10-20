// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_NVTX_COLLECTOR_H_
#define PARCAGPU_NVTX_COLLECTOR_H_

#include <cstdint>

namespace parcagpu {

// NVTX capture. libparcagpucupti.so is itself an NVTX injection library: it
// exports InitializeInjectionNvtx2 (nvtx_injection.cpp), so every NVTX client
// copy in the process (torch, the Python nvtx package, cuDNN, ...) that
// starts with NVTX_INJECTION64_PATH pointing at this library calls straight
// into us. The injection resolves domains and registered strings and hands
// the events to the collector below, which batches them per thread and fires
// the nvtx_event_batch / nvtx_resource_name USDT probes.
namespace nvtx {

// Called from InitializeInjection (cuInit). Points NVTX_INJECTION64_PATH at
// this library if nothing else claimed it, so NVTX clients that haven't
// initialized yet still find us, and enables device-id lookup.
void onCudaInjected();

// Flush the calling thread's pending batch. Called by CuptiProfiler::cleanup
// (best-effort: only flushes the cleanup-running thread; per-thread batches
// on other live threads drain via the pthread_key destructor).
void flushCurrentThread();

// Whether a consumer is attached (probe semaphore hot), or a test/debug
// override is on. Checked before doing any per-event work.
bool captureEnabled();
bool resourceNameEnabled();

// Record one event. message must stay valid until the call returns (it is
// copied into the per-thread pool).
void captureEvent(uint16_t kind, uint32_t domainId, uint64_t rangeId,
                  uint32_t color, uint64_t payload, const char *message);

// Fire nvtx_resource_name immediately (no batching).
void emitResourceName(uint32_t kind, uint64_t id, const char *name);

} // namespace nvtx

// The calling thread's in-flight CUDA runtime correlation id, or 0. Defined
// in cupti.cpp.
uint32_t currentRuntimeCorrelationId();

} // namespace parcagpu

#endif // PARCAGPU_NVTX_COLLECTOR_H_
