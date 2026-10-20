// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PARCAGPU_NVTX_H_
#define PARCAGPU_NVTX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wire format for the nvtx_event_batch USDT probe. The probe argument is a
// pointer to an array of `count` const-pointers to nvtx_event records; each
// record is a stable, packed C struct that BPF reads via bpf_probe_read_user.
// Layout must not change without coordinating with parca-prof's BPF and Go
// interpreter (support/ebpf/cuda.ebpf.c, interpreter/gpu/nvtx.go).
enum nvtx_event_kind {
  NVTX_KIND_RANGE_PUSH  = 1,  // nvtx[Domain]RangePush{A,Ex}
  NVTX_KIND_RANGE_POP   = 2,  // nvtx[Domain]RangePop
  NVTX_KIND_RANGE_START = 3,  // nvtx[Domain]RangeStart{A,Ex}
  NVTX_KIND_RANGE_END   = 4,  // nvtx[Domain]RangeEnd
  NVTX_KIND_MARK        = 5,  // nvtx[Domain]Mark{A,Ex}
};

// device_id is captured inline so that NVTX events with correlation_id == 0
// (typical for ranges that bracket CUDA calls rather than overlap them) still
// carry the thread's current CUDA device. stream_id is intentionally NOT
// inline: streams aren't thread-local, so the interpreter resolves them by
// joining correlation_id against the kernel_executed probe stream when one
// is in flight.
struct nvtx_event {
  uint64_t timestamp_ns;     // CLOCK_MONOTONIC at callback entry
  uint32_t tid;              // syscall(SYS_gettid)
  uint32_t correlation_id;   // in-flight CUDA runtime correlation; 0 if none
  uint32_t domain_id;        // 0 = default domain; else see NVTX_RESOURCE_DOMAIN
  uint16_t kind;             // enum nvtx_event_kind
  uint16_t flags;            // reserved; 0 in v1
  uint32_t color;            // ARGB from nvtxEventAttributes_t::color; 0 if unset
  uint32_t device_id;        // thread's current CUDA device; UINT32_MAX if unknown
  uint64_t payload;          // raw payload bits; type discriminator deferred
  uint64_t range_id;         // RANGE_START/END pairing; 0 for stacked push/pop
  const char *message;       // user-owned NUL-terminated string; BPF copies
};

// Resource kinds for the nvtx_resource_name probe.
enum nvtx_resource_kind {
  NVTX_RESOURCE_OS_THREAD   = 1,
  NVTX_RESOURCE_CUDA_STREAM = 2,
  NVTX_RESOURCE_CUDA_DEVICE = 3,
  NVTX_RESOURCE_CUDA_KERNEL = 4,
  // id is the nvtx_event domain_id, name the nvtxDomainCreateA name. Fired
  // on domain creation and re-fired periodically while the domain is in use,
  // so a consumer that attached late still learns the name.
  NVTX_RESOURCE_DOMAIN      = 5,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PARCAGPU_NVTX_H_
