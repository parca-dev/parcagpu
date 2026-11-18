// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

// Sampled GPU memory allocation profiling.
//
// Driver-level allocations (cuMemAlloc*, stream-ordered/pool allocations,
// managed memory, VMM cuMemMap) are sampled by bytes, Poisson-style, and
// reported with an unbiased weighted_bytes estimate. Frees are reported
// unsampled so a consumer can pair them with the allocations it kept.
//
// Two sinks, each gated on its own USDT semaphore:
//   otel_gpu_memory:alloc/free  - opentelemetry-ebpf-profiler #1747 contract
//   parcagpu:gpu_mem_alloc/free - adds device id and memory kind

#ifndef PARCAGPU_GPU_MEMORY_H_
#define PARCAGPU_GPU_MEMORY_H_

#include <cstdint>

#include <cupti.h>

namespace parcagpu {
namespace gpu_memory {

// Memory kinds carried by parcagpu:gpu_mem_alloc. Mirrored in
// ebpf/cupti_bpf.h (GPU_MEM_KIND_*).
enum Kind : uint32_t {
  kKindDevice = 1,  // cuMemAlloc, cuMemAllocPitch
  kKindManaged = 2, // cuMemAllocManaged
  kKindAsync = 3,   // cuMemAllocAsync, cuMemAllocFromPoolAsync
  kKindVmm = 4,     // cuMemMap
};

// Default mean sampling interval, in bytes.
constexpr uint64_t kDefaultSamplingInterval = 512 * 1024;

// Read PARCAGPU_MEM_SAMPLING_INTERVAL. Called once from init_debug().
void configure();

// Driver-API callback ids this module handles.
bool isMemCbid(CUpti_CallbackId cbid);

// Enable or disable the CBIDs above on a subscriber.
void setCallbacks(CUpti_SubscriberHandle subscriber, bool enable);

// Handle one driver-API callback whose cbid satisfies isMemCbid().
void onCallback(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata);

} // namespace gpu_memory
} // namespace parcagpu

#endif // PARCAGPU_GPU_MEMORY_H_
