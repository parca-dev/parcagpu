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

# Bundled CUPTI (redistributable under the CUDA EULA), from NVIDIA's
# cuda_cupti archives: libcupti.so.12 (CUDA 12.9.1), libcupti.so.13 (CUDA
# 13.0.2), and the libnvperf_host.so each one loads for PC sampling, renamed
# libnvperf_host.so.12/.13. libparcagpucupti.so picks one pair at runtime.
FROM --platform=$BUILDPLATFORM alpine:3.22 AS cupti
ARG TARGETARCH
WORKDIR /cupti
RUN set -e; mkdir /out; \
    case "$TARGETARCH" in \
      amd64) arch=x86_64; \
        sum12=f779a24e8f0177b4ce45cbf118cb470139fb5107858df96689d0a0aa01f0fba1; \
        sum13=92fb7e0430521517174c4c171173b888aaa8d78d8762e59752dea2ed14a0ad7d ;; \
      arm64) arch=sbsa; \
        sum12=84f8657375c23a425b2d107c3cde1ce75777bff9798b744b78e146fdc02d2de0; \
        sum13=f6f34d534cce56f91b1496abf51be3b1559ba879985d34eb89c808004b77513a ;; \
      *) echo "unsupported TARGETARCH $TARGETARCH"; exit 1 ;; \
    esac; \
    for v in "12 12.9.79 $sum12" "13 13.0.85 $sum13"; do \
      set -- $v; \
      dir=cuda_cupti-linux-$arch-$2-archive; \
      wget -q https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/linux-$arch/$dir.tar.xz; \
      echo "$3  $dir.tar.xz" | sha256sum -c -; \
      tar -xJf $dir.tar.xz $dir/lib; \
      cp -L $dir/lib/libcupti.so.$1 /out/; \
      cp -L $dir/lib/libnvperf_host.so /out/libnvperf_host.so.$1; \
    done

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
    make -j$(nproc)

# Export stage for extracting the library (used by Makefile and release binaries).
# The bundled CUPTI files ship next to libparcagpucupti.so, which loads them
# when no other libcupti is found.
FROM scratch AS export
COPY --from=builder /build/build/lib/libparcagpucupti.so /
COPY --from=cupti /out/ /

# Runtime image (for container registry)
FROM busybox:latest AS runtime
COPY --from=builder /build/build/lib/libparcagpucupti.so /usr/lib/
COPY --from=cupti /out/ /usr/lib/
