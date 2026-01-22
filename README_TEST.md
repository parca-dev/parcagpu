# Testing libparcagpucupti.so

This project includes comprehensive test infrastructure for the CUPTI profiler library.

## Building

```bash
# Build everything (libparcagpucupti.so + test infrastructure)
make

# Or build and test in one step
make test
```

This builds:
- `build/lib/libparcagpucupti.so` - Production library (CMake)
- `build/lib/libcupti.so` - Mock CUPTI for test infrastructure
- `build/bin/test_cupti_prof` - Test program

## Quick Start

```bash
# Build and run the test (generates ~500 events)
make test

# Or run test script directly (builds automatically)
./test.sh
```

This will:
1. Build the mock CUPTI library and test program
2. Start bpftrace to monitor DTRACE probes
3. Run the test with debug output enabled
4. Generate ~500 CUDA events (kernels and graph launches) at 1000 events/second
5. Show detailed output with timestamps for all CUPTI operations
6. Display captured DTRACE probe results

## Running Continuously

For extended testing or continuous probe monitoring:
```bash
LD_LIBRARY_PATH=build/lib build/bin/test_cupti_prof build/lib/libparcagpucupti.so --forever
```

This runs indefinitely at 1000 events/second until interrupted (Ctrl-C).

## Verifying DTRACE Probes

To verify that USDT probes are firing with correct values:

**Terminal 1** - Monitor with bpftrace:
```bash
sudo bpftrace parcagpu.bt
```

**Terminal 2** - Run the test:
```bash
./test.sh
```

Expected probe output:
```
[PID] Kernel executed:
  start=1006000000, end=1006500000, duration=500000 ns
  correlationId=6, deviceId=0, streamId=1
  name=mock_cuda_kernel_name

[PID] Graph executed:
  start=1007000000, end=1007300000, duration=300000 ns
  correlationId=7, deviceId=0, streamId=1, graphId=3

=== Summary ===
Graph executions: @graph_launches: count 117
Kernel executions: @kernel_executions: count 117
```

## Test Details

See [test/README.md](test/README.md) for complete documentation including:
- Test architecture and components
- Build system details
- Manual test execution
- Implementation notes
