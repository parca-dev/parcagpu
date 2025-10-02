.PHONY: all clean test cupti cupti-arm64 test-infra

# Default target: build everything
all: cupti test-infra

# Build libparcagpucupti.so with CMake (native)
cupti:
	@echo "=== Building libparcagpucupti.so with CMake ==="
	@mkdir -p cupti/build
	@cd cupti/build && cmake .. && $(MAKE)

# Build libparcagpucupti.so for ARM64 using Docker cross-compilation
cupti-arm64:
	@echo "=== Building libparcagpucupti.so for ARM64 with Docker ==="
	@mkdir -p cupti/build-arm64
	@docker buildx build -f Dockerfile.arm64 \
		--build-context cuda-headers=/usr/local/cuda-12.9 \
		--output type=local,dest=cupti/build-arm64 .
	@echo "ARM64 library built: cupti/build-arm64/libparcagpucupti.so"

# Build test infrastructure with Zig
test-infra:
	@echo "=== Building test infrastructure with Zig ==="
	@zig build

# Run tests
test: all
	@./test.sh

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	@rm -rf cupti/build cupti/build-arm64
	@rm -rf zig-out
	@rm -rf .zig-cache
	@echo "Clean complete"
