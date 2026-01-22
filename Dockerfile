# Slim multi-platform build for libparcagpucupti.so
# Uses pre-built CUDA header images instead of full CUDA development images
# This significantly reduces build time and disk space requirements
#
# Thanks to Proton's dynamic CUPTI loading, we only need to build once
# and the library works with any CUDA version at runtime.

# CUDA header image (can be overridden at build time)
ARG CUDA_HEADERS=ghcr.io/parca-dev/cuda-headers:12

# Import CUDA headers
FROM ${CUDA_HEADERS} AS cuda-headers

# Build stage
FROM ubuntu:24.04 AS builder

# Install only build tools (no CUDA toolkit needed)
RUN apt-get update && apt-get install -y \
    cmake \
    make \
    g++ \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy CUDA headers from header image
COPY --from=cuda-headers /usr/local/cuda /usr/local/cuda

# Copy parcagpu source files and proton submodule
COPY src /build/src
COPY proton /build/proton
COPY CMakeLists.txt /build/

# Build the library (disable tests for Docker build)
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_INCLUDE_DIR=/usr/local/cuda/include -DBUILD_TESTS=OFF .. && \
    make -j$(nproc)

# Export stage for extracting the library (used by Makefile and release binaries)
FROM scratch AS export
COPY --from=builder /build/build/lib/libparcagpucupti.so /

# Runtime image (for container registry)
FROM busybox:latest AS runtime
COPY --from=builder /build/build/lib/libparcagpucupti.so /usr/lib/libparcagpucupti.so
