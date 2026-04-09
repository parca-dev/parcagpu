// CUPTI BPF definitions — shared between test/bpf/ and production cuda.ebpf.c.
//
// Contains BPF-side layouts for:
//   - CUpti_ActivityKernel5 (kernel activity records)
//   - CUpti_PCSamplingPCData (PC sampling records)
//   - CUpti_PCSamplingStallReason (stall reason entries)

#ifndef CUPTI_BPF_H
#define CUPTI_BPF_H

// ---------------------------------------------------------------------------
// Kernel activity records
// ---------------------------------------------------------------------------

// CUpti_ActivityKind values we care about
#define CUPTI_ACTIVITY_KIND_KERNEL            3
#define CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL 10

// Matches the layout of CUpti_ActivityKernel5 exactly.
// Explicit padding replaces __packed__ to avoid unnecessary unaligned-access
// handling in BPF.  The struct uses aligned(8) and a static size assert.
struct cupti_activity_kernel5 {
  u32 kind;            // offset 0   - CUpti_ActivityKind
  u8 _pad1[12];        // offset 4   - cacheConfig, sharedMemConfig,
                       //              registersPerThread,
                       //              partitionedGlobalCache x2
  u64 start;           // offset 16  - kernel start timestamp (ns)
  u64 end;             // offset 24  - kernel end timestamp (ns)
  u64 completed;       // offset 32  - completion timestamp
  u32 device_id;       // offset 40
  u32 context_id;      // offset 44
  u32 stream_id;       // offset 48
  u8 _pad2[40];        // offset 52  - gridX/Y/Z, blockX/Y/Z,
                       //              staticSharedMemory,
                       //              dynamicSharedMemory,
                       //              localMemoryPerThread,
                       //              localMemoryTotal
  u32 correlation_id;  // offset 92
  s64 grid_id;         // offset 96
  u64 name_ptr;        // offset 104 - const char* (user-space pointer)
  u64 _reserved0;      // offset 112
  u64 queued;          // offset 120
  u64 submitted;       // offset 128
  u8 _pad3[8];         // offset 136 - launchType, isSharedMemoryCarveout,
                       //              sharedMemoryCarveoutRequested,
                       //              padding, sharedMemoryExecuted
  u64 graph_node_id;   // offset 144
  u32 shmem_limit_cfg; // offset 152 - CUpti_FuncShmemLimitConfig
  u32 graph_id;        // offset 156
} __attribute__((aligned(8)));

_Static_assert(
  sizeof(struct cupti_activity_kernel5) == 160, "cupti_activity_kernel5 size mismatch");

// ---------------------------------------------------------------------------
// PC sampling records
// ---------------------------------------------------------------------------

#define STALL_REASON_NAME_LEN 64
#define MAX_STALL_REASONS     64
#define MAX_PC_BATCH_SIZE     512
#define MAX_FUNC_NAME         128

// Matches CUpti_PCSamplingStallReason (packed, aligned 8).
struct cupti_stall_reason {
  u32 stall_reason_index;
  u32 samples;
};

// Matches CUpti_PCSamplingPCData (packed, aligned 8).
// Contains user-space pointers that BPF chases with bpf_probe_read_user.
// We read the base 56-byte struct, then conditionally read correlationId
// if the size field indicates CUPTI 12.4+ / v22+ (size > 56).
struct cupti_pc_data {
  u64 size; // struct size (56 = pre-12.4, 60+ = CUDA 12.4+)
  u64 cubin_crc;
  u64 pc_offset;
  u32 function_index;
  u32 _pc_pad;
  u64 function_name_ptr; // const char* in user-space
  u64 stall_reason_count;
  u64 stall_reason_ptr; // CUpti_PCSamplingStallReason* in user-space
} __attribute__((__packed__)) __attribute__((aligned(8)));

#define CUPTI_PC_DATA_BASE_SIZE 56

#endif // CUPTI_BPF_H
