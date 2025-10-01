# parcagpu - CUPTI Profiler with USDT Probes

CUDA profiling library that exposes kernel and graph execution events via USDT/DTRACE probes for eBPF/bpftrace monitoring.

## Building

```bash
make        # Build everything
make test   # Build and run tests
make clean  # Clean all build artifacts
```

### Components Built

1. **libparcagpucupti.so** (CMake + real CUPTI)
   - Production library for CUDA injection
   - Located in `cupti/build/`
   - Links against real NVIDIA CUPTI

2. **Test Infrastructure** (Zig)
   - Mock CUPTI library for testing
   - Test program that simulates CUDA events
   - Located in `zig-out/`

## Usage

### As CUDA Injection Library

```bash
export CUDA_INJECTION64_PATH=/path/to/libparcagpucupti.so
# Run your CUDA application
./my_cuda_app
```

### Monitoring with bpftrace

```bash
# Terminal 1: Monitor probes
sudo bpftrace parcagpu.bt

# Terminal 2: Run CUDA application
./my_cuda_app
```

## Testing

```bash
# Run test suite with bpftrace monitoring
make test

# Run test continuously (for extended monitoring)
LD_LIBRARY_PATH=zig-out/lib zig-out/bin/test_cupti_prof cupti/build/libparcagpucupti.so --forever
```

See [README_TEST.md](README_TEST.md) for detailed testing documentation.

## USDT Probes

The library exposes two USDT probes:

### parcagpu:kernel_executed
- **arg0**: start timestamp (ns)
- **arg1**: end timestamp (ns)
- **arg2**: correlationId | (deviceId << 32)
- **arg3**: streamId
- **arg4**: kernel name (string pointer)

### parcagpu:graph_executed
- **arg0**: start timestamp (ns)
- **arg1**: end timestamp (ns)
- **arg2**: correlationId | (deviceId << 32)
- **arg3**: streamId
- **arg4**: graphId

## Requirements

- CUDA Toolkit (CUPTI libraries)
- Zig (for building test infrastructure)
- CMake (for building production library)
- bpftrace (for probe monitoring)

## Directory Structure

```
.
├── Makefile              # Top-level build orchestration
├── build.zig             # Zig build for test infrastructure
├── cupti/
│   ├── CMakeLists.txt    # CMake build for production library
│   ├── cupti-prof.c      # Main profiler implementation
│   └── build/            # CMake build output
├── test/
│   ├── mock_cupti.c      # Mock CUPTI for testing
│   └── test_cupti_prof.c # Test program
├── parcagpu.bt           # bpftrace monitoring script
└── test.sh               # Test runner
```
