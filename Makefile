.PHONY: all clean test cupti-amd64 cupti-arm64 test-infra prepare-cuda-headers

# CUDA headers source (local CUDA installation or extracted from NVIDIA container)
CUDA_HEADERS_DIR ?= /usr/local/cuda-12.9

# Default target: build everything for native architecture
all: cupti-all test-infra

# Prepare CUDA headers directory for Docker build
prepare-cuda-headers:
	@echo "=== Preparing CUDA headers from $(CUDA_HEADERS_DIR) ==="
	@rm -rf .cuda_headers
	@mkdir -p .cuda_headers/include .cuda_headers/CUPTI/include
	@if [ -d "$(CUDA_HEADERS_DIR)/include" ]; then \
		cp -r $(CUDA_HEADERS_DIR)/include/* .cuda_headers/include/; \
		if [ -d "$(CUDA_HEADERS_DIR)/extras/CUPTI/include" ]; then \
			cp -r $(CUDA_HEADERS_DIR)/extras/CUPTI/include/* .cuda_headers/CUPTI/include/; \
		fi; \
		echo "CUDA headers prepared successfully"; \
	else \
		echo "Error: CUDA headers not found at $(CUDA_HEADERS_DIR)"; \
		echo "Set CUDA_HEADERS_DIR to your CUDA installation path"; \
		exit 1; \
	fi

# Build libparcagpucupti.so for AMD64 using Docker
cupti-amd64: prepare-cuda-headers
	@echo "=== Building libparcagpucupti.so for AMD64 with Docker ==="
	@mkdir -p cupti/build-amd64
	@docker buildx build -f Dockerfile.amd64 \
		--build-arg CUDA_HEADERS_SRC=.cuda_headers \
		--output type=local,dest=cupti/build-amd64 \
		--platform linux/amd64 .
	@echo "AMD64 library built: cupti/build-amd64/libparcagpucupti.so"

# Build libparcagpucupti.so for ARM64 using Docker cross-compilation
cupti-arm64: prepare-cuda-headers
	@echo "=== Building libparcagpucupti.so for ARM64 with Docker ==="
	@mkdir -p cupti/build-arm64
	@docker buildx build -f Dockerfile.arm64 \
		--build-arg CUDA_HEADERS_SRC=.cuda_headers \
		--output type=local,dest=cupti/build-arm64 \
		--platform linux/arm64 .
	@echo "ARM64 library built: cupti/build-arm64/libparcagpucupti.so"

# Build both architectures
cupti-all: cupti-amd64 cupti-arm64

# Build test infrastructure with Zig
test-infra:
	@echo "=== Building test infrastructure with Zig ==="
	@zig build

# Run tests (using AMD64 library)
test: cupti-amd64 test-infra
	@./test.sh

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	@rm -rf cupti/build cupti/build-amd64 cupti/build-arm64
	@rm -rf zig-out
	@rm -rf .zig-cache
	@rm -rf .cuda_headers
	@echo "Clean complete"
