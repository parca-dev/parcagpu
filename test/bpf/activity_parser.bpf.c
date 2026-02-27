// SPDX-License-Identifier: GPL-2.0
// eBPF program that reads CUPTI activity records via USDT probe.
// Attaches to the parcagpu:activity_batch USDT probe site(s) in
// the target shared library. The probe fires with
// (const void **ptrs, uint32_t num_activities) where ptrs is an array
// of pointers to raw CUPTI activity records of any kind.
// This program filters for kernel activities and sends them to user-space
// via ring buffer. Other activity kinds are skipped.
//
// Argument locations vary per probe site (encoded in .note.stapsdt).
// The Go loader parses the ELF notes, populates __bpf_usdt_specs,
// and attaches uprobes with a cookie carrying the spec ID so
// bpf_usdt_arg() reads from the correct register/memory at each site.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// usdt_compat.h provides macros and guards so the otel-ebpf-profiler
// usdt headers compile in our vmlinux.h-based environment.
#include "usdt_args.h"
#include "usdt_compat.h"

#include "cupti_activity_bpf.h"

#define MAX_BATCH_SIZE 128
#define MAX_KERNEL_NAME 128

// USDT spec map — populated by Go loader before uprobe attachment.
// Keyed by spec ID (uint32); value is struct bpf_usdt_spec.
// Old-style SEC("maps") definition to match the extern in usdt_args.h.
struct bpf_map_def __bpf_usdt_specs SEC("maps") = {
    .type = BPF_MAP_TYPE_HASH,
    .key_size = sizeof(u32),
    .value_size = sizeof(struct bpf_usdt_spec),
    .max_entries = 256,
};

// Event sent to user-space for each kernel activity found.
struct kernel_event {
  u64 start;
  u64 end;
  u32 correlation_id;
  u32 device_id;
  u32 stream_id;
  u32 graph_id;
  u64 graph_node_id;
  char name[MAX_KERNEL_NAME];
};

// Ring buffer for sending events to user-space.
struct bpf_map_def events SEC("maps") = {
    .type = BPF_MAP_TYPE_RINGBUF,
    .max_entries = 1 << 20, // 1 MB
};

// Stats counters.
struct bpf_map_def stats SEC("maps") = {
    .type = BPF_MAP_TYPE_ARRAY,
    .key_size = sizeof(u32),
    .value_size = sizeof(u64),
    .max_entries = 4,
};

enum stat_key {
  STAT_BATCHES = 0,    // number of batch probe invocations
  STAT_ACTIVITIES = 1, // total activity records scanned
  STAT_KERNELS = 2,    // kernel activities emitted
  STAT_DROPS = 3,      // ring buffer full (events dropped)
};

static __always_inline void bump_stat(enum stat_key key) {
  u32 k = key;
  u64 *val = bpf_map_lookup_elem(&stats, &k);
  if (val)
    __sync_fetch_and_add(val, 1);
}

SEC("usdt/parcagpu/activity_batch")
int BPF_USDT(handle_activity_batch, u64 ptrs_base, u32 num_activities) {

  bump_stat(STAT_BATCHES);

  if (num_activities > MAX_BATCH_SIZE)
    num_activities = MAX_BATCH_SIZE;

  // Iterate through pointers in the batch.
  // The loop bound is a compile-time constant so the verifier accepts it.
  for (u32 i = 0; i < MAX_BATCH_SIZE; i++) {
    if (i >= num_activities)
      break;

    // Read the i-th pointer from the array: ptrs[i]
    u64 record_ptr = 0;
    int ret = bpf_probe_read_user(&record_ptr, sizeof(record_ptr),
                                  (void *)(ptrs_base + (u64)i * sizeof(u64)));
    if (ret != 0 || record_ptr == 0)
      break;

    bump_stat(STAT_ACTIVITIES);

    // Read just the kind field first to decide how to handle this record.
    u32 kind = 0;
    ret = bpf_probe_read_user(&kind, sizeof(kind), (void *)record_ptr);
    if (ret != 0)
      continue;

    // Only process kernel activities; skip memcpy, memset, etc.
    if (kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
        kind != CUPTI_ACTIVITY_KIND_KERNEL)
      continue;

    // Read the full kernel activity record.
    struct cupti_activity_kernel5 record = {};
    ret = bpf_probe_read_user(&record, sizeof(record), (void *)record_ptr);
    if (ret != 0)
      continue;

    bump_stat(STAT_KERNELS);

    // Reserve space in the ring buffer for the event.
    struct kernel_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
    if (!evt) {
      bump_stat(STAT_DROPS);
      continue;
    }

    evt->start = record.start;
    evt->end = record.end;
    evt->correlation_id = record.correlation_id;
    evt->device_id = record.device_id;
    evt->stream_id = record.stream_id;
    evt->graph_id = record.graph_id;
    evt->graph_node_id = record.graph_node_id;

    // Read kernel name string from user-space pointer.
    if (record.name_ptr) {
      bpf_probe_read_user_str(evt->name, sizeof(evt->name),
                              (void *)record.name_ptr);
    } else {
      evt->name[0] = '\0';
    }

    bpf_ringbuf_submit(evt, 0);
  }

  return 0;
}

char LICENSE[] SEC("license") = "GPL";
