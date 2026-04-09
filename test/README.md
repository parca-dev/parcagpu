# CUPTI Profiler Test Infrastructure

This directory contains test infrastructure for `libparcagpucupti.so` using CMake as the build system.

## Components

- **test/mock_cupti.c**: Mock CUPTI library that provides stub implementations of all CUPTI APIs used by the profiler
- **test/test_cupti_prof.c**: Test program that dynamically loads libparcagpucupti.so and simulates CUPTI callbacks
- **CMakeLists.txt**: CMake build configuration (at project root)
- **test.sh**: Test script (at project root)

## Building

From the project root:
```bash
make
```

This builds:
1. `libcupti.so` - Mock CUPTI library with stub implementations
2. `libparcagpucupti.so` - The profiler library linked against the mock CUPTI
3. `test_cupti_prof` - Test executable that loads and exercises the profiler

All outputs go to `build/lib/` and `build/bin/`.

## Running

Using the test script (recommended):
```bash
./test.sh
```

Using Make directly:
```bash
make test
```

Or manually:
```bash
make
LD_LIBRARY_PATH=build/lib build/bin/test_cupti_prof build/lib/libparcagpucupti.so
```

### Running Continuously

To run the test in continuous mode (useful for monitoring probes with bpftrace):
```bash
LD_LIBRARY_PATH=build/lib build/bin/test_cupti_prof build/lib/libparcagpucupti.so --forever
```

In this mode, the test will:
- Generate events indefinitely at 1000 events/second
- Print status every 100 iterations (~500 events)
- Run until interrupted with Ctrl-C

## Test Behavior

The test program:
1. Dynamically loads `libparcagpucupti.so`
2. Calls `InitializeInjection()` to initialize the profiler
3. Simulates ~1000 CUDA events per second by calling `runtimeApiCallback()` in a loop
4. Alternates between kernel launches and graph launches
5. Periodically calls `bufferCompleted()` with mock activity records containing:
   - Kernel execution records (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
   - Graph execution records (CUPTI_ACTIVITY_KIND_GRAPH_TRACE)
6. Each activity record triggers the DTRACE probes:
   - `parcagpu:cuda_correlation` - Fired on API callback with correlationId
   - `parcagpu:kernel_executed` - Fired with kernel timing and metadata
   - `parcagpu:graph_executed` - Fired with graph timing and metadata

## Rate Limiting

The test generates exactly 1000 events per second (5 events every 5ms) to match the requirement. The loop runs for 100 iterations, generating ~500 total events.

## Debugging

The test script automatically enables `PARCAGPU_DEBUG=1` to show detailed debug output including:
- CUPTI initialization steps
- Activity buffer management with timestamps
- All callback invocations
- Cleanup operations

To run without debug output:
```bash
LD_LIBRARY_PATH=build/lib build/bin/test_cupti_prof build/lib/libparcagpucupti.so
```

## Verifying DTRACE Probes

To verify that the DTRACE/USDT probes are firing correctly, use the provided bpftrace script:

**Terminal 1** - Run bpftrace to monitor probes:
```bash
sudo bpftrace parcagpu.bt
```

**Terminal 2** - Run the test:
```bash
./test.sh
```

You should see output like:
```
[PID] Kernel executed:
  start=1006000000, end=1006500000, duration=500000 ns
  correlationId=6, deviceId=0, streamId=1
  name=mock_cuda_kernel_name

[PID] Graph executed:
  start=1007000000, end=1007300000, duration=300000 ns
  correlationId=7, deviceId=0, streamId=1, graphId=3
```

The summary at the end will show total counts for kernel and graph executions.

## Implementation Notes

- The mock CUPTI library stores callback function pointers in global variables that are exported to the test program
- The test program retrieves these callbacks after `InitializeInjection()` is called
- Activity records are properly formatted and parsed by `cuptiActivityGetNextRecord`
- The DTRACE probes (`parcagpu:kernel_executed` and `parcagpu:graph_executed`) fire with correct values:
  - **deviceId**: 0
  - **streamId**: 1
  - **start/end**: Reasonable timestamp values with 500μs duration for kernels, 300μs for graphs
  - **name**: "mock_cuda_kernel_name" for kernel events
- The test generates approximately 500 events over the course of execution at 1000 events/second
- **Cleanup is idempotent**: The `cleanup()` function in cupti-prof.c can safely be called multiple times
- The test explicitly calls `cleanup()` before `dlclose()` and uses `_exit()` to avoid the atexit handler being called after the library is unloaded
