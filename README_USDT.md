# USDT Tracepoint Implementation for parcagpu

This document describes the USDT (User Statically Defined Tracing) implementation added to the parcagpu library on the `probe` branch.

## Overview

USDT provides a way for external tracing tools to observe kernel launch timing data without requiring special privileges or affecting the application's normal operation when not being traced.

## Implementation Details

### USDT Provider Definition

```rust
#[usdt::provider]
mod parcagpu_provider {
    fn kernel_launch(kernel_id: u32, duration_ms: f32) {}
}
```

### Tracepoint Location

The USDT tracepoint is emitted in the asynchronous message processing paths:

1. **With Unix Socket Connection** (`serve_stream`): 
   - When parca-agent or similar tool is connected
   - Timing calculated asynchronously without blocking main thread
   - Tracepoint emitted after sending data to socket

2. **Without Unix Socket Connection** (`process_messages`):
   - When no external agent is connected
   - Timing calculated asynchronously in background task
   - Tracepoint emitted even when data would normally be dropped

This design ensures:
- **Non-blocking operation**: Host thread never waits for GPU synchronization
- **Always available**: USDT data emitted regardless of socket connection state
- **Async timing**: Leverages existing async infrastructure for timing calculation

### Tracepoint Data

The `parcagpu_provider::kernel_launch` tracepoint provides:
- `kernel_id` (u32): Random ID identifying the specific kernel launch
- `duration_ms` (f32): Elapsed time in milliseconds for kernel execution

### Dependencies Added

- `usdt = "0.5"` - Core USDT support for eBPF tracing
- `serde = "1.0"` - Required by USDT implementation

### Probe Registration

For eBPF tracing, **no runtime registration is needed**. The USDT probes are embedded in the binary at compile time and eBPF tools can attach directly to them. The `#[usdt::provider]` macro automatically handles probe generation during compilation.

## Testing

### Basic Functionality Test

```bash
# Build the library
cargo build --release

# Run test program with parcagpu library
LD_LIBRARY_PATH=. LD_PRELOAD=./target/release/libparcagpu.so ./test_mock
```

The test should show kernel launches with timing information.

### External Tracing

The USDT tracepoint can be observed by external tools like:

- **bpftrace**: `sudo bpftrace -e 'usdt:*:parcagpu_provider:kernel_launch { printf("Kernel %08x: %.3f ms\n", arg0, arg1); }'`
- **SystemTap**: Custom scripts targeting the `parcagpu_provider:kernel_launch` probe
- **perf**: `perf record -e sdt_parcagpu:kernel_launch`

### Checking for Probes

```bash
# Check if probes are embedded (may require additional configuration)
readelf -n target/release/libparcagpu.so | grep -i stapsdt

# List available USDT probes (if bpftrace is installed)
bpftrace -l 'usdt:*parcagpu*'
```

## Advantages of USDT Approach

1. **No Root Required**: Unlike eBPF, USDT probes can be traced without root privileges in many cases
2. **Zero Overhead**: When not being traced, USDT probes have virtually no performance impact
3. **Standard Interface**: Works with existing tracing infrastructure (SystemTap, bpftrace, perf)
4. **No Special Permissions**: Applications can run normally without any special setup
5. **Real-time**: Timing data is available immediately as kernels complete

## Usage in Production

External monitoring tools can attach to any process using the parcagpu library and observe:
- Kernel launch frequency
- Kernel execution times
- Performance bottlenecks
- Real-time GPU utilization patterns

The tracing can be enabled/disabled dynamically without restarting the target application.