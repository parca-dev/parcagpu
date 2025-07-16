#!/bin/bash

set -e

echo "USDT Tracing Demo for parcagpu"
echo "=============================="

# Build everything using Makefile
echo "Building components..."
make all

echo ""
echo "Starting USDT tracing demonstration..."

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$TEST_PID" ]; then
        kill $TEST_PID 2>/dev/null || true
    fi
    if [ -n "$TRACER_PID" ]; then
        sudo kill $TRACER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start the test program in background
echo "Starting test program..."
echo "PID: $$"
echo "Running single test iteration..."
LD_LIBRARY_PATH=. PARCAGPU_USE_SOCKET=0 LD_PRELOAD=./target/release/libparcagpu.so ./test_mock > mock.out 2>&1 &
TEST_PID=$!

echo "Test program PID: $TEST_PID"

# Start the eBPF-based tracer
sudo ./gotracer/gotracer $TEST_PID &
TRACER_PID=$!

# Wait for the test program to complete
wait $TEST_PID
TEST_EXIT_CODE=$?

echo ""
echo "Test program completed with exit code: $TEST_EXIT_CODE"

# Give tracer a moment to process any final events
sleep 2

echo ""
echo "Demo completed successfully!"
echo ""
echo "What happened:"
echo "- Test program ran a single iteration with parcagpu library preloaded"
echo "- USDT tracer attached to the test program PID"
echo "- Kernel launches triggered USDT tracepoints with timing data"
echo "- Real-time GPU kernel profiling without special app permissions"