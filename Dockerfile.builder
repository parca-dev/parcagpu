# Build toolchain image for libparcagpucupti.so
# Pushed to ghcr.io/parca-dev/parcagpu-builder and used as the base of the
# builder stage in Dockerfile, so release builds never hit the Ubuntu mirrors.
#
# The base image determines the minimum glibc required by the built library
# (ubuntu 24.04 -> glibc 2.39).
#
# Usage: docker buildx build -f Dockerfile.builder --platform linux/amd64,linux/arm64 \
#          -t ghcr.io/parca-dev/parcagpu-builder:24.04 --push .

FROM ubuntu:24.04

# systemtap-sdt-dev provides dtrace (used to generate probes.h / probes.o)
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    cmake \
    make \
    g++ \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*
