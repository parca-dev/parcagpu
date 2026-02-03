# Slim multi-platform build for libparcagpucupti.so
# Uses pre-built CUDA header images instead of full CUDA development images
# This significantly reduces build time and disk space requirements

# CUDA header images (can be overridden at build time)
ARG CUDA_12_HEADERS=ghcr.io/parca-dev/cuda-headers:12
ARG CUDA_13_HEADERS=ghcr.io/parca-dev/cuda-headers:13

# Import CUDA 12 headers
FROM ${CUDA_12_HEADERS} AS cuda12-headers

# Import CUDA 13 headers
FROM ${CUDA_13_HEADERS} AS cuda13-headers

# Build stage for CUDA 12
FROM ubuntu:22.04 AS builder-cuda12

# Install only build tools (no CUDA toolkit needed)
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    gcc \
    g++ \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/cupti

# Copy CUDA headers and libraries from header image
COPY --from=cuda12-headers /usr/local/cuda /usr/local/cuda

# Copy source code
COPY cupti/cupti-prof.c cupti/correlation_filter.cpp cupti/correlation_filter.h cupti/CMakeLists.txt ./

# Build the library for CUDA 12
ENV CUDA_ROOT=/usr/local/cuda
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_ROOT=${CUDA_ROOT} -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && \
    make VERBOSE=1 && \
    mv libparcagpucupti.so libparcagpucupti.so.12

# Build stage for CUDA 13
FROM ubuntu:22.04 AS builder-cuda13

# Install only build tools (no CUDA toolkit needed)
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    gcc \
    g++ \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/cupti

# Copy CUDA headers and libraries from header image
COPY --from=cuda13-headers /usr/local/cuda /usr/local/cuda

# Copy source code
COPY cupti/cupti-prof.c cupti/correlation_filter.cpp cupti/correlation_filter.h cupti/CMakeLists.txt ./

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
