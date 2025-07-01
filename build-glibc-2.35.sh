#!/bin/bash
# Build script for GLIBC 2.35 compatibility

# Create output directory
mkdir -p target-glibc235

# Option 1: Build in Docker with Ubuntu 22.04 (GLIBC 2.35)
cat > Dockerfile.ubuntu2204 << 'EOF'
FROM ubuntu:22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy only necessary files (exclude .cargo directory if it exists)
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Create a stub libcudart.so for linking
RUN echo 'void cudaEventCreateWithFlags() {}' > stub.c && \
    echo 'void cudaEventRecord() {}' >> stub.c && \
    echo 'void cudaEventSynchronize() {}' >> stub.c && \
    echo 'void cudaEventElapsedTime() {}' >> stub.c && \
    echo 'void cudaStreamIsCapturing() {}' >> stub.c && \
    echo 'void cudaLaunchKernel() {}' >> stub.c && \
    gcc -shared -fPIC stub.c -o /usr/lib/libcudart.so

# Build the library
RUN cargo build --release

# The built library will be at /build/target/release/libparcagpu.so
EOF

echo "Building in Ubuntu 22.04 container (GLIBC 2.35)..."
docker build -f Dockerfile.ubuntu2204 -t parcagpu-glibc235 .

echo "Extracting built library..."
docker run --rm -v "$PWD/target-glibc235:/output" parcagpu-glibc235 \
    bash -c "cp target/release/libparcagpu.so /output/ && echo 'Library copied successfully'"

if [ -f "target-glibc235/libparcagpu.so" ]; then
    echo "Library built for GLIBC 2.35 is available at: target-glibc235/libparcagpu.so"

    # Verify GLIBC requirements
    echo "Verifying GLIBC requirements..."
    objdump -T target-glibc235/libparcagpu.so | grep GLIBC | sort -u | tail -10
else
    echo "Error: Library was not built successfully"
    exit 1
fi