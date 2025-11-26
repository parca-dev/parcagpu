# Multi-platform build for libparcagpucupti.so
# Supports both AMD64 and ARM64 architectures
# Builds both CUDA 12 and 13 versions in a single container
#
# Build args:
#   CUDA_12_FULL_VERSION: Full CUDA 12 version (default: 12.9.1)
#   CUDA_13_FULL_VERSION: Full CUDA 13 version (default: 13.0.2)
#
# Stages:
#   builder-cuda12: Builds library for CUDA 12
#   builder-cuda13: Builds library for CUDA 13
#   runtime: Final image with both CUDA versions included

ARG CUDA_12_FULL_VERSION=12.9.1
ARG CUDA_13_FULL_VERSION=13.0.2

# Build stage for CUDA 12
FROM nvidia/cuda:${CUDA_12_FULL_VERSION}-devel-ubuntu22.04 AS builder-cuda12

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

# Build the library for CUDA 12
ENV CUDA_ROOT=/usr/local/cuda
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_ROOT=${CUDA_ROOT} .. && \
    make VERBOSE=1 && \
    mv libparcagpucupti.so libparcagpucupti.so.12

# Build stage for CUDA 13
FROM nvidia/cuda:${CUDA_13_FULL_VERSION}-devel-ubuntu22.04 AS builder-cuda13

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

# Build the library for CUDA 13
ENV CUDA_ROOT=/usr/local/cuda
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_ROOT=${CUDA_ROOT} .. && \
    make VERBOSE=1 && \
    mv libparcagpucupti.so libparcagpucupti.so.13

# Export stages for extracting single libraries (used by Makefile and release binaries)
FROM scratch AS export-cuda12
COPY --from=builder-cuda12 /build/cupti/build/libparcagpucupti.so.12 /

FROM scratch AS export-cuda13
COPY --from=builder-cuda13 /build/cupti/build/libparcagpucupti.so.13 /

# Runtime image with both CUDA versions (for container registry)
FROM busybox:latest AS runtime
COPY --from=builder-cuda12 /build/cupti/build/libparcagpucupti.so.12 /usr/lib/
COPY --from=builder-cuda13 /build/cupti/build/libparcagpucupti.so.13 /usr/lib/
# Default symlink points to CUDA 12
RUN ln -s /usr/lib/libparcagpucupti.so.12 /usr/lib/libparcagpucupti.so
