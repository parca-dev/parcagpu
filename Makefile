.PHONY: all clean test cupti-amd64 cupti-arm64 cupti-all cupti-all-versions cross test-infra docker-push docker-test-build docker-test-run format

# CUDA version configuration
CUDA_MAJOR ?= 12
CUDA_FULL_VERSION ?= 12.9.1
LIB_NAME = libparcagpucupti.so.$(CUDA_MAJOR)

# Default target: build everything for native architecture
all: cupti-all test-infra

# Build libparcagpucupti.so for AMD64 using Docker
cupti-amd64:
	@echo "=== Building $(LIB_NAME) for AMD64 with Docker (CUDA $(CUDA_MAJOR)) ==="
	@mkdir -p /tmp/parcagpu-build-amd64
	@docker buildx use default
	@docker buildx build -f Dockerfile \
		--target export-cuda$(CUDA_MAJOR) \
		--output type=local,dest=/tmp/parcagpu-build-amd64 \
		--platform linux/amd64 cupti
	@mkdir -p build/$(CUDA_MAJOR)/amd64
	@cp /tmp/parcagpu-build-amd64/$(LIB_NAME) build/$(CUDA_MAJOR)/amd64/
	@ln -sf $(LIB_NAME) build/$(CUDA_MAJOR)/amd64/libparcagpucupti.so
	@echo "AMD64 library built: build/$(CUDA_MAJOR)/amd64/$(LIB_NAME)"

# Build libparcagpucupti.so for ARM64 using Docker
cupti-arm64:
	@echo "=== Building $(LIB_NAME) for ARM64 with Docker (CUDA $(CUDA_MAJOR)) ==="
	@mkdir -p /tmp/parcagpu-build-arm64
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target export-cuda$(CUDA_MAJOR) \
		--output type=local,dest=/tmp/parcagpu-build-arm64 \
		--platform linux/arm64 cupti
	@mkdir -p build/$(CUDA_MAJOR)/arm64
	@cp /tmp/parcagpu-build-arm64/$(LIB_NAME) build/$(CUDA_MAJOR)/arm64/
	@ln -sf $(LIB_NAME) build/$(CUDA_MAJOR)/arm64/libparcagpucupti.so
	@echo "ARM64 library built: build/$(CUDA_MAJOR)/arm64/$(LIB_NAME)"

# Build both architectures for current CUDA version
cupti-all: cupti-amd64 cupti-arm64

# Build runtime container with both CUDA versions for both architectures
# Note: Builds but doesn't load into Docker (multi-platform images can't be loaded)
# Use docker-push to push to a registry, or remove one platform to load locally
cross:
	@echo "=== Building runtime container for AMD64 and ARM64 (includes CUDA 12 and 13) ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target runtime \
		--platform linux/amd64,linux/arm64 \
		cupti
	@echo "Runtime container built for both platforms (cached, not loaded into Docker)"

# Build all artifacts (CUDA 12 & 13 for both amd64 and arm64)
cupti-all-versions:
	@echo "=== Building all CUDA versions (12 and 13) for both architectures ==="
	@$(MAKE) cupti-amd64 CUDA_MAJOR=12 CUDA_FULL_VERSION=12.9.1
	@$(MAKE) cupti-arm64 CUDA_MAJOR=12 CUDA_FULL_VERSION=12.9.1
	@$(MAKE) cupti-amd64 CUDA_MAJOR=13 CUDA_FULL_VERSION=13.0.2
	@$(MAKE) cupti-arm64 CUDA_MAJOR=13 CUDA_FULL_VERSION=13.0.2
	@echo "=== All artifacts built ==="
	@echo "CUDA 12: build/12/amd64/libparcagpucupti.so.12"
	@echo "CUDA 12: build/12/arm64/libparcagpucupti.so.12"
	@echo "CUDA 13: build/13/amd64/libparcagpucupti.so.13"
	@echo "CUDA 13: build/13/arm64/libparcagpucupti.so.13"

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
# Note: Runtime image includes both CUDA 12 and 13
IMAGE ?= ghcr.io/parca-dev/parcagpu
IMAGE_TAG ?= latest
docker-push:
	@echo "=== Setting up buildx builder ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@echo "=== Building and pushing multi-arch Docker images to $(IMAGE):$(IMAGE_TAG) (includes CUDA 12 and 13) ==="
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

format:
	clang-format -i -style=file cupti/*.[ch]