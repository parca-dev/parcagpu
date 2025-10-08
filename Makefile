.PHONY: all clean test cupti-amd64 cupti-arm64 test-infra docker-push

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

# Build and push multi-arch Docker images to ghcr.io
# Set IMAGE_TAG to override the default tag (e.g., make docker-push IMAGE_TAG=v1.0.0)
IMAGE_TAG ?= latest
docker-push:
	@echo "=== Setting up buildx builder ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@echo "=== Building and pushing multi-arch Docker images to ghcr.io/parca-dev/parcagpu:$(IMAGE_TAG) ==="
	@docker buildx build -f Dockerfile \
		--target runtime \
		--platform linux/amd64,linux/arm64 \
		--tag ghcr.io/parca-dev/parcagpu:$(IMAGE_TAG) \
		--push \
		.
	@echo "Images pushed successfully to ghcr.io/parca-dev/parcagpu:$(IMAGE_TAG)"
