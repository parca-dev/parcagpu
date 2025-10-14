# Multi-platform build for libparcagpucupti.so
# Supports both AMD64 and ARM64 architectures
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build tools
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    gcc \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /build/cupti
COPY . .

# Build the library
ENV CUDA_ROOT=/usr/local/cuda
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_ROOT=${CUDA_ROOT} .. && \
    make VERBOSE=1

# Extract the built library (for local builds)
FROM scratch AS export
COPY --from=builder /build/cupti/build/libparcagpucupti.so /

# Runtime image (for container registry)
FROM busybox:latest AS runtime
COPY --from=builder /build/cupti/build/libparcagpucupti.so /usr/local/lib/
