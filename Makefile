.PHONY: all clean test cupti-amd64 cupti-arm64 cupti-all cupti-all-versions cross test-infra docker-push push-cuda-headers docker-test-build docker-test-run format

# CUDA version configuration
CUDA_MAJOR ?= 12
CUDA_FULL_VERSION ?= 12.9.1
LIB_NAME = libparcagpucupti.so.$(CUDA_MAJOR)

# Default target: build all CUDA versions (12 & 13) for both architectures and test infrastructure
all: cupti-all-versions test-infra

local:
	mkdir -p build && \
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build -S cupti && \
    cd build && make -j$(nproc)

# Build libparcagpucupti.so for AMD64 using Docker
cupti-amd64:
	@echo "=== Building $(LIB_NAME) for AMD64 with Docker (CUDA $(CUDA_MAJOR)) ==="
	@mkdir -p /tmp/parcagpu-build-amd64
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--build-arg CUDA_12_HEADERS=$(CUDA_12_HEADERS) \
		--build-arg CUDA_13_HEADERS=$(CUDA_13_HEADERS) \
		--target export-cuda$(CUDA_MAJOR) \
		--output type=local,dest=/tmp/parcagpu-build-amd64 \
		--platform linux/amd64 .
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
		--build-arg CUDA_12_HEADERS=$(CUDA_12_HEADERS) \
		--build-arg CUDA_13_HEADERS=$(CUDA_13_HEADERS) \
		--target export-cuda$(CUDA_MAJOR) \
		--output type=local,dest=/tmp/parcagpu-build-arm64 \
		--platform linux/arm64 .
	@mkdir -p build/$(CUDA_MAJOR)/arm64
	@cp /tmp/parcagpu-build-arm64/$(LIB_NAME) build/$(CUDA_MAJOR)/arm64/
	@ln -sf $(LIB_NAME) build/$(CUDA_MAJOR)/arm64/libparcagpucupti.so
	@echo "ARM64 library built: build/$(CUDA_MAJOR)/arm64/$(LIB_NAME)"

# Build both architectures for current CUDA version (controlled by CUDA_MAJOR variable)
# Example: make cupti-all CUDA_MAJOR=13
cupti-all: cupti-amd64 cupti-arm64

# Build all local artifacts: CUDA 12 & 13 for both amd64 and arm64
# This extracts the .so files to build/12/{amd64,arm64}/ and build/13/{amd64,arm64}/
# Use this target when you want local build artifacts for testing
cupti-all-versions:
	@echo "=== Building all CUDA versions (12 and 13) for both architectures ==="
	@$(MAKE) cupti-amd64 CUDA_MAJOR=12 CUDA_FULL_VERSION=12.9.1
	@$(MAKE) cupti-arm64 CUDA_MAJOR=12 CUDA_FULL_VERSION=12.9.1
	@$(MAKE) cupti-amd64 CUDA_MAJOR=13 CUDA_FULL_VERSION=13.0.2
	@$(MAKE) cupti-arm64 CUDA_MAJOR=13 CUDA_FULL_VERSION=13.0.2
	@echo "=== All artifacts built ==="
	@echo "CUDA 12 amd64: build/12/amd64/libparcagpucupti.so.12"
	@echo "CUDA 12 arm64: build/12/arm64/libparcagpucupti.so.12"
	@echo "CUDA 13 amd64: build/13/amd64/libparcagpucupti.so.13"
	@echo "CUDA 13 arm64: build/13/arm64/libparcagpucupti.so.13"

# Build runtime container image with both CUDA versions for both architectures
# Note: This builds a Docker image but doesn't extract local artifacts or load into Docker
# Multi-platform images stay in buildx cache. Use docker-push to push to registry.
cross:
	@echo "=== Building runtime container for AMD64 and ARM64 (includes CUDA 12 and 13) ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--build-arg CUDA_12_HEADERS=$(CUDA_12_HEADERS) \
		--build-arg CUDA_13_HEADERS=$(CUDA_13_HEADERS) \
		--target runtime \
		--platform linux/amd64,linux/arm64 \
		.
	@echo "Runtime container built for both platforms (cached, not loaded into Docker)"

# CUDA header image configuration
# Can be overridden to use local images (e.g., make cupti-all CUDA_12_HEADERS=cuda-headers:12)
CUDA_HEADERS_REGISTRY ?= ghcr.io/parca-dev/cuda-headers
CUDA_12_HEADERS ?= $(CUDA_HEADERS_REGISTRY):12
CUDA_13_HEADERS ?= $(CUDA_HEADERS_REGISTRY):13

# Build and push CUDA header images to registry
# These are lightweight images (~35MB each) containing only CUDA headers and libcupti
# Note: Only needs to be run manually when:
#   - CUDA versions are updated (12.9.1 -> 12.x.x, 13.0.2 -> 13.x.x)
#   - New CUDA major versions are added
#   - CUPTI API changes require header updates
push-cuda-headers:
	@echo "=== Building and pushing CUDA header images ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@echo "Building CUDA 12 headers..."
	@docker buildx build -f Dockerfile.cuda-headers \
		--build-arg CUDA_VERSION=12.9.1 \
		--platform linux/amd64,linux/arm64 \
		--tag $(CUDA_HEADERS_REGISTRY):12 \
		--push \
		.
	@echo "Building CUDA 13 headers..."
	@docker buildx build -f Dockerfile.cuda-headers \
		--build-arg CUDA_VERSION=13.0.2 \
		--platform linux/amd64,linux/arm64 \
		--tag $(CUDA_HEADERS_REGISTRY):13 \
		--push \
		.
	@echo "CUDA header images pushed to $(CUDA_HEADERS_REGISTRY):12 and :13"

# Build test infrastructure with CMake
test-infra:
	@echo "=== Building test infrastructure with CMake ==="
	@mkdir -p test/build
	@cd test/build && cmake .. -DCUDA_MAJOR_VERSION=$(CUDA_MAJOR) && make

# Run tests (using AMD64 library)
test: cupti-amd64 test-infra
	@./test.sh

# Run advanced test (8 GPUs @ 2500 launches/s, multi-threaded)
test-advanced: cupti-amd64 test-infra
	@echo "=== Running advanced test (8 GPUs @ 2500 launches/s) ==="
	@cd test/build && LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH \
		./test_cupti_prof ../../build/$(CUDA_MAJOR)/amd64/libparcagpucupti.so \
		--threads=4 --num-gpus=8 --launch-rate=2500 --duration=10

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	@rm -rf cupti/build cupti/build-amd64 cupti/build-arm64 build
	@rm -rf test/build
	@echo "Clean complete"

# Build and push multi-arch Docker images to ghcr.io
# Set IMAGE_TAG to override the default tag (e.g., make docker-push IMAGE_TAG=v1.0.0)
# Set IMAGE to override the image name (e.g., make docker-push IMAGE=ghcr.io/myuser/parcagpu)
# Set CUDA_12_HEADERS and CUDA_13_HEADERS to override header images (e.g., cuda-headers:12 for local)
# Note: Runtime image includes both CUDA 12 and 13
IMAGE ?= ghcr.io/parca-dev/parcagpu
IMAGE_TAG ?= latest
docker-push:
	@echo "=== Setting up buildx builder ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@echo "=== Building and pushing multi-arch Docker images to $(IMAGE):$(IMAGE_TAG) (includes CUDA 12 and 13) ==="
	@docker buildx build -f Dockerfile \
		--build-arg CUDA_12_HEADERS=$(CUDA_12_HEADERS) \
		--build-arg CUDA_13_HEADERS=$(CUDA_13_HEADERS) \
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
