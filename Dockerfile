# Slim multi-platform build for libparcagpucupti.so
# Uses pre-built CUDA header images instead of full CUDA development images
# This significantly reduces build time and disk space requirements
#
# Thanks to Proton's dynamic CUPTI loading, we only need to build once
# and the library works with any CUDA version at runtime.

# CUDA header image (can be overridden at build time)
ARG CUDA_HEADERS=ghcr.io/parca-dev/cuda-headers:12

# Build toolchain image, see Dockerfile.builder (can be overridden at build time)
ARG BUILDER=ghcr.io/parca-dev/parcagpu-builder:24.04

# Import CUDA headers
FROM ${CUDA_HEADERS} AS cuda-headers

# Build stage (no CUDA toolkit needed)
FROM ${BUILDER} AS builder

WORKDIR /build

# Copy CUDA headers from header image
COPY --from=cuda-headers /usr/local/cuda /usr/local/cuda

# Version for parcagpu's startup log record (no .git in the build context).
ARG PARCAGPU_VERSION=unknown

# Copy parcagpu source files and proton submodule
COPY src /build/src
COPY proton /build/proton
COPY CMakeLists.txt /build/

# Build the library (disable tests for Docker build)
RUN mkdir -p build && \
    cd build && \
    cmake -DCUDA_INCLUDE_DIR=/usr/local/cuda/include -DBUILD_TESTS=OFF \
          -DPARCAGPU_VERSION="${PARCAGPU_VERSION}" .. && \
    make -j$(nproc) && \
    cp -L /usr/local/cuda/lib64/libcupti.so lib/libcupti.so

# Export stage for extracting the library (used by Makefile and release binaries).
# libcupti.so (redistributable under the CUDA EULA) ships next to
# libparcagpucupti.so, which loads it when no other libcupti is found.
FROM scratch AS export
COPY --from=builder /build/build/lib/libparcagpucupti.so /build/build/lib/libcupti.so /

# Runtime image (for container registry)
FROM busybox:latest AS runtime
COPY --from=builder /build/build/lib/libparcagpucupti.so /build/build/lib/libcupti.so /usr/lib/
