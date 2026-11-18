# parcagpu - CUPTI Profiler with USDT Probes

CUDA profiling library that exposes GPU activity via USDT/DTRACE probes for eBPF consumption. Captures kernel executions, PC sampling with stall reasons, and cubin module loading.

## Optional cubin file provenance

The legacy `cubin_loaded(crc, ptr, size)` probe remains unchanged. The optional
`cubin_loaded_file(crc, ptr, size, path)` probe carries the same loaded image and
an absolute original filename when a synchronous module resource callback
occurs inside `cuModuleLoad`. The path is resolved in the application's working
directory and filesystem namespace. Both probes fire on load and metadata
replay, including when a consumer joins either probe.

Consumers should attach to one cubin probe, prefer the file-bearing probe when
available, and treat an empty path as normal. `cuModuleLoadData`/`DataEx`, JIT
linking and deferred loads have no filename here. A path is only a hint: PTX,
fatbinaries and shared libraries need not contain the exact executed image as a
standalone file. A consumer must validate the complete file against the loaded
image before using it, resolve paths through the target process's root, and
handle later deletion or mutation. parcagpu does not create a disk cache.

Old consumers continue using the legacy probe. New consumers must support old
producers without this optional probe. Missing file provenance must not prevent
PC offsets, kernel identity, stall reasons or sample counts from being reported.

## Building

```bash
make local     # Build libparcagpucupti.so locally (CMake, RelWithDebInfo)
make debug     # Build with full debug, no optimizations
make clean     # Clean all build artifacts
```

Docker cross-compilation:

```bash
make build-amd64   # Build .so for AMD64
make build-arm64   # Build .so for ARM64
make build-all     # Both architectures
make docker-push   # Push multi-arch image to ghcr.io
```

## Usage

### As CUDA Injection Library

```bash
export CUDA_INJECTION64_PATH=/path/to/libparcagpucupti.so
./my_cuda_app
```

### NVTX Annotations

The library is also an NVTX injection library (`InitializeInjectionNvtx2`).
To capture NVTX ranges and marks (torch.cuda.nvtx, the Python `nvtx`
package, TensorRT-LLM, cuDNN, ...), point `NVTX_INJECTION64_PATH` at the same
library:

```bash
export CUDA_INJECTION64_PATH=/path/to/libparcagpucupti.so
export NVTX_INJECTION64_PATH=/path/to/libparcagpucupti.so
./my_cuda_app
```

Each NVTX client copy loads its injection on its first NVTX call. Libraries
often make that call before CUDA initializes (at import time), and a client
that finds no injection stays a no-op for the rest of the process. So set
`NVTX_INJECTION64_PATH` in the launcher. As a fallback, parcagpu sets it at
`cuInit` if it's unset, which only covers clients that haven't made an NVTX
call yet. The injection slot holds one library, so this can't be combined
with Nsight Systems' NVTX capture. Events are only recorded while a consumer
is attached to the `nvtx_event_batch` probe.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PARCAGPU_DEBUG` | off | Enable debug logging |
| `PARCAGPU_RATE_LIMIT` | 100 | Token-bucket rate limit for callback probes (events/sec per thread) |
| `PARCAGPU_GRAPH_RATE_LIMIT` | 20 | Separate token-bucket rate limit for graph launches (`cuGraphLaunch`/`cudaGraphLaunch`), events/sec per thread. Graph launches draw from this bucket instead of `PARCAGPU_RATE_LIMIT`, so graph replays can't flood the probes and eager launch bursts can't starve graph sampling. |
| `PARCAGPU_PC_SAMPLING_RATE` | (unset) | Setting this env var to a non-negative number enables PC sampling. The value is the target samples/sec; a controller adjusts the per-window dice-roll probability internally to converge on this rate. Set to 0 to enable sampling with rate control disabled (controller off, fixed probability). |
| `PARCAGPU_SAMPLING_FACTOR` | 20 | Hardware PC sampling period (log2 cycles between samples). Only takes effect when PC sampling is enabled via `PARCAGPU_PC_SAMPLING_RATE`. |
| `PARCAGPU_PC_HW_BUFFER_MB` | (unset) | Force the CUPTI PC-sampling hardware buffer to this many MB. Unset or 0 leaves it at the CUPTI default. |
| `PARCAGPU_MEM_SAMPLING_INTERVAL` | (unset) | Setting this env var enables GPU memory profiling (see [GPU Memory Profiling](#gpu-memory-profiling)). The value is the mean bytes between sampled allocations; 524288 (512 KiB) is a reasonable start. 0 samples every allocation. |

### GPU Memory Profiling

parcagpu samples driver-level GPU allocations so a profiler can build
allocated and in-use GPU memory profiles keyed by the CPU stack that
allocated them. It works like the host heap profiling in
opentelemetry-ebpf-profiler
([#1747](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/pull/1747)).

- **Hooked calls**: allocations are `cuMemAlloc`, `cuMemAllocPitch`,
  `cuMemAllocManaged`, `cuMemAllocAsync`, `cuMemAllocFromPoolAsync` and VMM
  `cuMemMap`. Frees are `cuMemFree`, `cuMemFreeAsync` and `cuMemUnmap`. The
  runtime API (`cudaMalloc`, `cudaMallocAsync`, ...) goes through these
  calls. Pinned host memory is not tracked.
- **Byte sampling**: an allocation is sampled when a point of a Poisson
  process over bytes, with mean `PARCAGPU_MEM_SAMPLING_INTERVAL`, falls inside
  it. That happens with probability `p = 1 - exp(-size/interval)`, and each
  sampled allocation reports `weighted_bytes = size / p`, an unbiased
  estimate. Allocations much larger than the interval are always sampled, at
  their real size.
- **Frees are not sampled**: every free is reported, and consumers ignore
  pointers they never saw allocated.
- **Off by default**: the alloc/free callbacks are only registered when
  `PARCAGPU_MEM_SAMPLING_INTERVAL` is set. Even then nothing runs unless a
  consumer is attached to an alloc probe.

Two providers carry the same sampling decisions:

| Probe | Arguments |
|---|---|
| `otel_gpu_memory:alloc` | ptr, size, weighted_bytes |
| `otel_gpu_memory:free` | ptr |
| `parcagpu:gpu_mem_alloc` | ptr, size, weightedBytes, deviceId, kind (1 device, 2 managed, 3 async, 4 vmm) |
| `parcagpu:gpu_mem_free` | ptr, deviceId |

`otel_gpu_memory` is the `otel_memory` contract from #1747 under its own
provider name, so GPU memory never lands in a host heap profile. As that
contract requires, the arguments are always in the ABI argument registers
(`rdi`/`rsi`/`rdx`, `x0`/`x1`/`x2`). `scripts/check-otel-usdt-regs.sh`
checks this at build time.

Hooks sit at the driver level. With a caching allocator, as in PyTorch, or
with vLLM's preallocated KV cache, the profile shows which code grew the
pool, not which tensors use it.

### Monitoring with bpftrace

```bash
sudo bpftrace parcagpu.bt
```

### Monitoring with the BPF Activity Parser

```bash
make bpf-test
sudo test/bpf/activity_parser -pid <PID> -lib <path/to/libparcagpucupti.so> -v
```

The activity parser attaches to all USDT probes via eBPF, captures events through a ring buffer, and resolves PC samples to source lines using `llvm-dwarfdump`.

## Testing

```bash
make test           # Basic mock CUPTI test (no GPU, no BPF)
make test-nvtx      # NVTX collector unit test (no GPU, no BPF)
make test-nvtx-real # Real NVTX clients + BPF parser, end to end (no GPU; sudo; NVTX_PYTHON=... adds the Python nvtx case)
make test-gpu-mem      # GPU allocation sampler unit test (no GPU, no BPF)
make test-gpu-mem-e2e  # GPU alloc/free through both providers + BPF parser, mock driver (no GPU; sudo)
make test-gpu-mem-real # Same checks against the real driver (requires root + GPU)
make test-pc-mock   # Mock PC sampling with BPF activity parser (no GPU, requires root)
make test-pc-real   # Real PC sampling with GPU (requires root + GPU)
make test-multi     # test_cupti_prof + BPF activity parser in parallel (requires root)
```

### BPF Prerequisites

The BPF-based tests (`test-pc-mock`, `test-pc-real`, `test-multi`, `test-nvtx-real`, `test-gpu-mem-e2e`, `test-gpu-mem-real`) require:

- Root (sudo) for BPF
- clang, libbpf-dev, bpftool
- Go 1.21+

Build just the BPF activity parser:

```bash
make generate   # Compile BPF objects via bpf2go
make bpf-test   # generate + build the Go binary
```

### Microbenchmarks

CUDA microbenchmarks for testing with real hardware:

```bash
make microbenchmarks   # Build all .cu files in microbenchmarks/
make test-pc-real      # Run pc_sample_toy under parcagpu with BPF
```

## USDT Probes

Defined in `src/probes.d`, provider `parcagpu`:

| Probe | Arguments | Description |
|---|---|---|
| `cuda_correlation` | correlationId, cbid, name | API callback correlation |
| `kernel_executed` | start, end, correlationId, deviceId, streamId, graphId, graphNodeId, name | Kernel execution timing |
| `activity_batch` | ptrs, count | Batch of CUPTI activity records |
| `pc_sample_batch` | records, count | Batch of PC sampling records |
| `stall_reason_map` | names, count | Stall reason name table |
| `cubin_loaded` | cubinCrc, cubin, cubinSize | Module load event |
| `cubin_unloaded` | cubinCrc | Module unload event |
| `error` | code, message, component | Profiler error event |
| `nvtx_event_batch` | ptrs, count | Batch of NVTX events (`struct nvtx_event`, `src/nvtx.h`); flushed at 256 events or when the oldest is 100ms old |
| `nvtx_resource_name` | kind, id, name | NVTX name for an OS thread, CUDA device/stream or domain (`NVTX_RESOURCE_*`) |
| `gpu_mem_alloc` | ptr, size, weightedBytes, deviceId, kind | Sampled GPU allocation (see [GPU Memory Profiling](#gpu-memory-profiling)) |
| `gpu_mem_free` | ptr, deviceId | GPU free (every free; not sampled) |

`src/otel_gpu_memory_probes.d` defines provider `otel_gpu_memory` (`alloc`, `free`), described under [GPU Memory Profiling](#gpu-memory-profiling).

## Requirements

- CUDA Toolkit (CUPTI headers/libraries)
- CMake
- dtrace (systemtap-sdt-dev)
- bpftrace (for probe monitoring)
- clang, libbpf-dev, bpftool, Go 1.21+ (for BPF tests)

## Directory Structure

```
.
├── Makefile              # Top-level build orchestration
├── CMakeLists.txt        # CMake build for library and test infrastructure
├── src/
│   ├── cupti.cpp         # Main CUPTI profiler implementation
│   ├── pc_sampling.cpp   # PC sampling support
│   ├── probes.d          # USDT probe definitions
│   └── ...
├── ebpf/
│   └── cupti_bpf.h       # Shared BPF struct definitions
├── test/
│   ├── test_cupti_prof.c # Mock CUPTI test harness
│   ├── mock_cupti.c      # Mock CUPTI library
│   ├── mock_cuda.c       # Mock CUDA driver library
│   ├── test-pc-mock.sh   # Mock PC sampling end-to-end test
│   ├── test-pc-real.sh   # Real GPU PC sampling end-to-end test
│   └── bpf/              # BPF activity parser (Go + eBPF)
├── microbenchmarks/      # CUDA microbenchmarks (.cu)
└── parcagpu.bt           # bpftrace monitoring script
```
