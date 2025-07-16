#!/bin/bash

# Script to test parcagpu in containers
# This will:
# 1. Build binaries locally using Makefile and glibc-2.35 script
# 2. Run test_mock with libparcagpu.so preloaded in a container
# 3. Run gotracer in another container to trace the CUDA calls
# 4. Demonstrate container-to-container tracing

set -e

echo "=== Building binaries locally ==="

echo "Building mock CUDA runtime and test program..."
make mock-cudart test_mock

echo "Building eBPF object file..."
make bpf

echo "Building gotracer..."
make gotracer

echo "Building libparcagpu.so with glibc 2.35..."
./build-glibc-2.35.sh

echo ""
echo "=== Building Docker images ==="

echo "Building test_mock image..."
docker build -f Dockerfile.test_mock -t test-mock-cuda .

echo "Building gotracer image..."
docker build -f Dockerfile.gotracer -t gotracer .

echo ""
echo "=== Starting containers ==="

# Start the test_mock container in the background
echo "Starting test_mock container..."
docker run --rm -d \
    --name test-mock-container \
    test-mock-cuda

# Give it a moment to start
sleep 2

# Get the PID of the test_mock process from the host's perspective
TEST_PID=$(docker inspect -f '{{.State.Pid}}' test-mock-container)
echo "test_mock container is running with host PID: $TEST_PID"

echo ""
echo "=== Running gotracer ==="
echo "Tracing CUDA calls from test_mock (PID: $TEST_PID)..."
echo "Press Ctrl+C to stop tracing"

# Run the tracer
# Note: We use --pid host and --privileged to allow tracing across containers
# Check if we're in a TTY
if [ -t 0 ]; then
    DOCKER_FLAGS="-it"
else
    DOCKER_FLAGS=""
fi

docker run --rm $DOCKER_FLAGS \
    --name gotracer-container \
    --pid host \
    --privileged \
    -v /sys/kernel/debug:/sys/kernel/debug:rw \
    gotracer $TEST_PID

# Cleanup
echo ""
echo "=== Cleaning up ==="
docker stop test-mock-container 2>/dev/null || true

echo "Test completed!"