.PHONY: all clean test cupti-amd64 cupti-arm64 test-infra

# Default target: build everything for native architecture
all: cupti-all test-infra

# Build libparcagpucupti.so for AMD64 using Docker
cupti-amd64:
	@echo "=== Building libparcagpucupti.so for AMD64 with Docker ==="
	@mkdir -p cupti/build-amd64
	@docker buildx build -f Dockerfile \
		--output type=local,dest=cupti/build-amd64 \
		--platform linux/amd64 .
	@echo "AMD64 library built: cupti/build-amd64/libparcagpucupti.so"

# Build libparcagpucupti.so for ARM64 using Docker
cupti-arm64:
	@echo "=== Building libparcagpucupti.so for ARM64 with Docker ==="
	@mkdir -p cupti/build-arm64
	@docker buildx build -f Dockerfile \
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
	@echo "Clean complete"
