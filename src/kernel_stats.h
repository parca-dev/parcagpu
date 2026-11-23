// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_KERNEL_STATS_H_
#define PARCAGPU_KERNEL_STATS_H_

#include <cstdint>

#include <cupti.h>

// Exact kernel execution counts and durations, independent of the rate
// limiter that decides which launches get a CPU stack and a kernel_executed
// probe.
//
// Every kernel activity record CUPTI hands to completeBuffer is counted here
// before the correlation filter drops the unsampled ones. Counts accumulate
// per (device, stream, kernel name) and leave through the kernel_stats probe
// as deltas: at most once a second from completeBuffer, and once more from
// cleanup after the final flush. Nothing is counted while no consumer is
// attached to kernel_stats, or while parca-agent has them off.
//
// The layouts below are the probe's ABI: the eBPF consumer reads them from
// this process's memory (see ebpf/cupti_bpf.h).

namespace parcagpu::kernel_stats {

struct Header {
  uint64_t startNs; // CUPTI timestamp: earliest kernel start in the interval
  uint64_t endNs;   // CUPTI timestamp: latest kernel end in the interval
  uint32_t seq;     // per-process firing counter; a gap means a lost firing
  uint32_t _pad;
};
static_assert(sizeof(Header) == 24, "kernel_stats header ABI");

struct Entry {
  uint32_t deviceId; // CUDA ordinal (join to UUID via device_info)
  uint32_t streamId;
  uint64_t executions; // kernel records, graph kernel nodes included
  uint64_t totalNs;    // sum of end - start
  const char *name;    // mangled kernel name, NUL-terminated
};
static_assert(sizeof(Entry) == 32, "kernel_stats entry ABI");

// Max entries per probe firing; a larger interval fires several times with
// the same header. Matches the eBPF consumer's tail-call budget.
constexpr uint32_t kMaxEntriesPerFiring = 256;

// Whether kernel stats are on: the agent's PARCAGPU_FEATURE_KERNEL_STATS bit
// while it controls this process, else always (the probe's semaphore then
// decides). One atomic load.
bool enabled();

// Counts one kernel activity record. Cheap no-op when nothing is attached,
// or while kernel stats are off.
void record(const CUpti_ActivityKernel5 *k);

// Fires kernel_stats if at least a second has passed since the last firing,
// or unconditionally when force is set (process exit).
void maybeEmit(bool force);

// Whether CUPTI's kernel activity was enabled at startup; set once by the
// profiler. Without it no kernel is recorded.
void setActivityEnabled(bool on);

// Logs (info) when parcagpu starts or stops recording every kernel: kernel
// activity on, consumers attached to cuda_correlation (which gets CUPTI
// activity buffers allocated) and kernel_stats, and the agent's
// PARCAGPU_FEATURE_KERNEL_STATS bit on while it controls the process.
// Called on CUDA API callbacks and from the agent control thread.
void logStateChange();

} // namespace parcagpu::kernel_stats

#endif // PARCAGPU_KERNEL_STATS_H_
