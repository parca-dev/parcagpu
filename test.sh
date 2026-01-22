#!/bin/bash

set -e  # Exit on error

cd "$(dirname "$0")"

# Parse arguments
USE_BPFTRACE=0
ARCH="${ARCH:-amd64}"
CUDA_MAJOR="${CUDA_MAJOR:-12}"
for arg in "$@"; do
    case $arg in
        --bpftrace)
            USE_BPFTRACE=1
            shift
            ;;
        *)
            # Pass through any other arguments to test program
            ;;
    esac
done

echo ""
echo "=== Building test infrastructure with CMake ==="
mkdir -p build-local
cd build-local
cmake ..
make -j$(nproc)
cd ..

# Start bpftrace if requested
if [ "$USE_BPFTRACE" -eq 1 ]; then
    if command -v bpftrace &> /dev/null; then
        echo "=== Starting bpftrace to monitor DTRACE probes ==="
        # Start bpftrace in background with sudo, using tee for output redirection
        sudo -b sh -c "bpftrace parcagpu.bt 2>&1 | tee /tmp/parcagpu_probes.log > /dev/null"
        # Get the PID of the bpftrace process (approximate - this is the shell wrapper)
        sleep 1
        BPFTRACE_PID=$(pgrep -f "bpftrace parcagpu.bt" || echo "")
        if [ -n "$BPFTRACE_PID" ]; then
            echo "bpftrace started (PID: $BPFTRACE_PID)"
        fi
        # Give bpftrace time to attach
        sleep 2
    else
        echo "Error: --bpftrace requested but bpftrace not found."
        echo "Install with: sudo apt-get install bpftrace"
        exit 1
    fi
fi

echo ""
echo "=== Running test program ==="
# Set LD_LIBRARY_PATH so the test can find libcupti.so and libparcagpucupti.so at runtime
# Set PARCAGPU_DEBUG to enable debug output
export LD_LIBRARY_PATH="$(pwd)/build-local/lib:$LD_LIBRARY_PATH"
export PARCAGPU_DEBUG=1
# Run the test program with path to library
./build-local/bin/test_cupti_prof build-local/lib/libparcagpucupti.so "$@"

# If bpftrace was started, stop it and show results
if [ "$USE_BPFTRACE" -eq 1 ]; then
    echo ""
    echo "=== Stopping bpftrace and showing probe results ==="
    # Give bpftrace a moment to process final events
    sleep 1
    # Kill bpftrace (need sudo since it was started with sudo)
    if [ -n "$BPFTRACE_PID" ]; then
        sudo kill $BPFTRACE_PID 2>/dev/null || true
        sleep 1
    fi

    echo ""
    echo "=== DTRACE Probe Results ==="
    if [ -f /tmp/parcagpu_probes.log ]; then
        cat /tmp/parcagpu_probes.log
        sudo rm -f /tmp/parcagpu_probes.log
    else
        echo "No probe results found"
    fi
fi

echo ""
echo "=== Test completed successfully ==="
