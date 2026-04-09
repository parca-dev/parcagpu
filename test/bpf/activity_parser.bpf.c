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
#include "usdt_compat.h"

#include "usdt_args.h"

#include "cupti_bpf.h"

#define MAX_BATCH_SIZE 128
#define MAX_KERNEL_NAME 128
#define MAX_CUBIN_SIZE (64 * 1024 * 1024) // 64MB safety cap

// USDT spec map — populated by Go loader before uprobe attachment.
// Keyed by spec ID (uint32); value is struct bpf_usdt_spec.
struct usdt_specs_t {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, u32);
    __type(value, struct bpf_usdt_spec);
    __uint(max_entries, 256);
} __bpf_usdt_specs SEC(".maps");

// Event sent to user-space for each kernel activity found.
struct kernel_event {
  u32 event_type; // EVENT_TYPE_KERNEL
  u32 _pad;
  u64 start;
  u64 end;
  u32 correlation_id;
  u32 device_id;
  u32 stream_id;
  u32 graph_id;
  u64 graph_node_id;
  char name[MAX_KERNEL_NAME];
};

// Cubin load/unload events — Go reads actual bytes via /proc/pid/mem.
struct cubin_event {
  u32 event_type; // EVENT_TYPE_CUBIN_LOADED or EVENT_TYPE_CUBIN_UNLOADED
  u32 _pad;
  u64 cubin_crc;
  u64 cubin_ptr; // user-space address (for /proc/pid/mem read)
  u64 cubin_size;
};

// Event type tags for the ring buffer.
#define EVENT_TYPE_KERNEL 1
#define EVENT_TYPE_CUBIN_LOADED 2
#define EVENT_TYPE_CUBIN_UNLOADED 3
#define EVENT_TYPE_PC_SAMPLE 4
#define EVENT_TYPE_ERROR 5

// PC sample event sent to user-space.
struct pc_sample_event {
  u32 event_type; // EVENT_TYPE_PC_SAMPLE
  u32 stall_reason_count;
  u64 cubin_crc;
  u64 pc_offset;
  u32 function_index;
  u32 correlation_id; // kernel correlation ID (CUDA 12.4+ / CUPTI v22+, else 0)
  char function_name[MAX_FUNC_NAME];
  struct cupti_stall_reason stall_reasons[MAX_STALL_REASONS];
};

// Error event sent to user-space.
#define MAX_ERROR_MSG 256
#define MAX_ERROR_COMPONENT 64
struct error_event {
  u32 event_type; // EVENT_TYPE_ERROR
  s32 error_code;
  char message[MAX_ERROR_MSG];
  char component[MAX_ERROR_COMPONENT];
};

// Ring buffer for sending events to user-space.
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20); // 1 MB
} events SEC(".maps");

// Stall reason name table — indexed by stall reason index.
// Value is a 64-byte null-terminated string.
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, char[STALL_REASON_NAME_LEN]);
    __uint(max_entries, MAX_STALL_REASONS);
} stall_reasons SEC(".maps");

// Whether the stall reason map has been populated.
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, u32);
    __uint(max_entries, 1);
} stall_map_loaded SEC(".maps");

// Stats counters.
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __type(key, u32);
    __type(value, u64);
    __uint(max_entries, 4);
} stats SEC(".maps");

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

    evt->event_type = EVENT_TYPE_KERNEL;
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

SEC("usdt/parcagpu/stall_reason_map")
int BPF_USDT(handle_stall_reason_map, u64 names_base, u32 count) {
  // Only load the map once.
  u32 zero = 0;
  u32 *loaded = bpf_map_lookup_elem(&stall_map_loaded, &zero);
  if (!loaded)
    return 0;
  if (*loaded)
    return 0;

  if (count > MAX_STALL_REASONS)
    count = MAX_STALL_REASONS;

  // Read each 64-byte name slot and store in the BPF map.
  for (u32 i = 0; i < MAX_STALL_REASONS; i++) {
    if (i >= count)
      break;

    char name[STALL_REASON_NAME_LEN] = {};
    int ret = bpf_probe_read_user(
        name, sizeof(name),
        (void *)(names_base + (u64)i * STALL_REASON_NAME_LEN));
    if (ret != 0)
      continue;

    bpf_map_update_elem(&stall_reasons, &i, name, BPF_ANY);
  }

  u32 one = 1;
  bpf_map_update_elem(&stall_map_loaded, &zero, &one, BPF_ANY);
  return 0;
}

SEC("usdt/parcagpu/cubin_loaded")
int BPF_USDT(handle_cubin_loaded, u64 cubin_crc, u64 cubin_ptr,
             u64 cubin_size) {
  if (cubin_size == 0 || cubin_size > MAX_CUBIN_SIZE)
    return 0;

  struct cubin_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
  if (!evt) {
    bump_stat(STAT_DROPS);
    return 0;
  }

  evt->event_type = EVENT_TYPE_CUBIN_LOADED;
  evt->cubin_crc = cubin_crc;
  evt->cubin_ptr = cubin_ptr;
  evt->cubin_size = cubin_size;
  bpf_ringbuf_submit(evt, 0);
  return 0;
}

SEC("usdt/parcagpu/cubin_unloaded")
int BPF_USDT(handle_cubin_unloaded, u64 cubin_crc) {
  struct cubin_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
  if (!evt) {
    bump_stat(STAT_DROPS);
    return 0;
  }

  evt->event_type = EVENT_TYPE_CUBIN_UNLOADED;
  evt->cubin_crc = cubin_crc;
  evt->cubin_ptr = 0;
  evt->cubin_size = 0;
  bpf_ringbuf_submit(evt, 0);
  return 0;
}

SEC("usdt/parcagpu/pc_sample_batch")
int BPF_USDT(handle_pc_sample_batch, u64 ptrs_base, u32 count) {
  if (count > MAX_PC_BATCH_SIZE)
    count = MAX_PC_BATCH_SIZE;

  for (u32 i = 0; i < MAX_PC_BATCH_SIZE; i++) {
    if (i >= count)
      break;

    // Read the i-th pointer from the array.
    u64 rec_ptr = 0;
    int ret = bpf_probe_read_user(&rec_ptr, sizeof(rec_ptr),
                                  (void *)(ptrs_base + (u64)i * sizeof(u64)));
    if (ret != 0 || rec_ptr == 0)
      continue;

    // Chase the pointer to read the CUPTI PC data record.
    struct cupti_pc_data rec = {};
    ret = bpf_probe_read_user(&rec, sizeof(rec), (void *)rec_ptr);
    if (ret != 0)
      continue;

    // Reserve ring buffer space for the event.
    struct pc_sample_event *evt =
        bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
    if (!evt) {
      bump_stat(STAT_DROPS);
      continue;
    }

    evt->event_type = EVENT_TYPE_PC_SAMPLE;
    evt->cubin_crc = rec.cubin_crc;
    evt->pc_offset = rec.pc_offset;
    evt->function_index = rec.function_index;

    // Read correlationId if the struct is large enough (CUDA 12.4+).
    // It sits right after the stallReason pointer at offset 56.
    evt->correlation_id = 0;
    if (rec.size > CUPTI_PC_DATA_BASE_SIZE) {
      u32 corr = 0;
      bpf_probe_read_user(&corr, sizeof(corr),
                          (void *)(rec_ptr + CUPTI_PC_DATA_BASE_SIZE));
      evt->correlation_id = corr;
    }

    // Chase the function name pointer.
    if (rec.function_name_ptr) {
      bpf_probe_read_user_str(evt->function_name, sizeof(evt->function_name),
                              (void *)rec.function_name_ptr);
    } else {
      evt->function_name[0] = '\0';
    }

    // Chase the stall reason pointer.
    u32 sr_count = rec.stall_reason_count;
    if (sr_count > MAX_STALL_REASONS)
      sr_count = MAX_STALL_REASONS;
    evt->stall_reason_count = sr_count;

    if (rec.stall_reason_ptr && sr_count > 0) {
      bpf_probe_read_user(evt->stall_reasons,
                          sr_count * sizeof(struct cupti_stall_reason),
                          (void *)rec.stall_reason_ptr);
    }

    bpf_ringbuf_submit(evt, 0);
  }

  return 0;
}

SEC("usdt/parcagpu/error")
int BPF_USDT(handle_error, s32 code, u64 message_ptr, u64 component_ptr) {
  struct error_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
  if (!evt) {
    bump_stat(STAT_DROPS);
    return 0;
  }

  evt->event_type = EVENT_TYPE_ERROR;
  evt->error_code = code;
  evt->message[0] = '\0';
  evt->component[0] = '\0';
  if (message_ptr)
    bpf_probe_read_user_str(evt->message, sizeof(evt->message),
                            (void *)message_ptr);
  if (component_ptr)
    bpf_probe_read_user_str(evt->component, sizeof(evt->component),
                            (void *)component_ptr);
  bpf_ringbuf_submit(evt, 0);
  return 0;
}

char LICENSE[] SEC("license") = "GPL";
