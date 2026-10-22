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

### Runtime control from parca-agent

parcagpu also takes configuration at runtime from parca-agent over
[procctl](https://github.com/parca-dev/procctl) (vendored as `src/procctl.h`):
each process creates a small sealed memfd named `procctl:parcagpu-v1` that the
agent finds through `/proc` and writes to, and reports back the devices it
uses. The schema is `src/parcagpu_ctl.h`.

The agent drives PC sampling. CUPTI can PC sample one process per GPU at a
time, so parca-agent hands each GPU to one process at a time and rotates it.
Its configuration says, per device the process reported, whether to sample
there, the sampling factor (`PARCAGPU_SAMPLING_FACTOR`'s meaning, applied
live), and the duty cycle: windows of a given length (default 30 ms) with a
given gap between them, each opening on the first kernel launch after the gap.
A process sets a context up for PC sampling only when its device is granted,
and disables it when the device is taken away: on a GB10 the GPU's sampler
belongs to the first process that enabled PC sampling, until it disables it.
parcagpu acknowledges a configuration once it has taken effect, so an ack with
the device off means the GPU is free. A small thread makes that happen within
~100 ms even when the application makes no CUDA calls; it never starts
sampling itself. A device that another process samples, or whose setup
failed, is reported as unavailable.

With no agent, an agent that sends no configuration, or an agent whose lease
has expired (it stopped heartbeating), parcagpu runs on the environment
variables above. A configuration that fails validation is not acknowledged
and has no effect.

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
make test-agent-ctl # Agent control over procctl, from a Go agent (no GPU, no root)
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
| `nvtx_event_batch` | ptrs, count | Batch of NVTX events (`struct nvtx_event`, `src/nvtx.h`); flushed at 256 events or when the oldest is 100ms old |
| `nvtx_resource_name` | kind, id, name | NVTX name for an OS thread, CUDA device/stream or domain (`NVTX_RESOURCE_*`) |

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
│   ├── agent_ctl.cpp     # Runtime control from parca-agent (procctl)
│   ├── parcagpu_ctl.h    # Its schema
│   ├── procctl.h         # Vendored procctl (process side)
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
│   ├── agentctl/         # Agent control tests (Go, procctl agent side)
│   └── bpf/              # BPF activity parser (Go + eBPF)
├── microbenchmarks/      # CUDA microbenchmarks (.cu)
└── parcagpu.bt           # bpftrace monitoring script
```
