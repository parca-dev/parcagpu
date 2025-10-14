.PHONY: all clean test cupti-amd64 cupti-arm64 test-infra docker-push docker-test-build docker-test-run

# Default target: build everything for native architecture
all: cupti-all test-infra

# Build libparcagpucupti.so for AMD64 using Docker
cupti-amd64:
	@echo "=== Building libparcagpucupti.so for AMD64 with Docker ==="
	@mkdir -p /tmp/parcagpu-build-amd64
	@docker buildx use default
	@docker buildx build -f Dockerfile \
		--target export \
		--output type=local,dest=/tmp/parcagpu-build-amd64 \
		--platform linux/amd64 cupti
	@mkdir -p build/amd64
	@cp /tmp/parcagpu-build-amd64/libparcagpucupti.so build/amd64/
	@echo "AMD64 library built: build/amd64/libparcagpucupti.so"

# Build libparcagpucupti.so for ARM64 using Docker
cupti-arm64:
	@echo "=== Building libparcagpucupti.so for ARM64 with Docker ==="
	@mkdir -p /tmp/parcagpu-build-arm64
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target export \
		--output type=local,dest=/tmp/parcagpu-build-arm64 \
		--platform linux/arm64 cupti
	@mkdir -p build/arm64
	@cp /tmp/parcagpu-build-arm64/libparcagpucupti.so build/arm64/
	@echo "ARM64 library built: build/arm64/libparcagpucupti.so"

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
	@rm -rf cupti/build cupti/build-amd64 cupti/build-arm64 build
	@rm -rf zig-out
	@rm -rf .zig-cache
	@echo "Clean complete"

# Build and push multi-arch Docker images to ghcr.io
# Set IMAGE_TAG to override the default tag (e.g., make docker-push IMAGE_TAG=v1.0.0)
# Set IMAGE to override the image name (e.g., make docker-push IMAGE=ghcr.io/myuser/parcagpu)
IMAGE ?= ghcr.io/parca-dev/parcagpu
IMAGE_TAG ?= latest
docker-push:
	@echo "=== Setting up buildx builder ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@echo "=== Building and pushing multi-arch Docker images to $(IMAGE):$(IMAGE_TAG) ==="
	@docker buildx build -f Dockerfile \
		--target runtime \
		--platform linux/amd64,linux/arm64 \
		--tag $(IMAGE):$(IMAGE_TAG) \
		--push \
		.
	@echo "Images pushed successfully to $(IMAGE):$(IMAGE_TAG)"

# Build test container image
docker-test-build: cupti-amd64 test-infra
	@echo "=== Building test container image ==="
	@docker build -f Dockerfile.test -t parcagpu-test:latest .
	@echo "Test container built: parcagpu-test:latest"

# Run tests in container
# Pass arguments with ARGS variable (e.g., make docker-test-run ARGS="--forever")
docker-test-run: docker-test-build
	@echo "=== Running tests in container ==="
	@docker run --rm parcagpu-test:latest $(ARGS)
