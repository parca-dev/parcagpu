# parcagpu - CUPTI Profiler with USDT Probes

CUDA profiling library that exposes GPU activity via USDT/DTRACE probes for eBPF consumption. Captures kernel executions, PC sampling with stall reasons, and cubin module loading.

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

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PARCAGPU_DEBUG` | off | Enable debug logging |
| `PARCAGPU_RATE_LIMIT` | 100 | Token-bucket rate limit for callback probes (events/sec per thread) |
| `PARCAGPU_SAMPLING_FACTOR` | 18 | PC sampling period; set to 0 to disable PC sampling |
| `PARCAGPU_PC_SAMPLING_PROBABILITY` | 0.01 | Probability of sampling in each interval window (0-1) |
| `PARCAGPU_PC_SAMPLING_INTERVAL` | 1.0 | PC sampling interval window in seconds |

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
make test-pc-mock   # Mock PC sampling with BPF activity parser (no GPU, requires root)
make test-pc-real   # Real PC sampling with GPU (requires root + GPU)
make test-multi     # test_cupti_prof + BPF activity parser in parallel (requires root)
```

### BPF Prerequisites

The BPF-based tests (`test-pc-mock`, `test-pc-real`, `test-multi`) require:

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
