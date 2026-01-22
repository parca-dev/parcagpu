.PHONY: all clean test build-amd64 build-arm64 build-all cross docker-push docker-test-build docker-test-run format local debug

LIB_NAME = libparcagpucupti.so

# Default target: build for both architectures
all: build-all

# Build libparcagpucupti.so for AMD64 using Docker
build-amd64:
	@echo "=== Building $(LIB_NAME) for AMD64 with Docker ==="
	@mkdir -p /tmp/parcagpu-build-amd64
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target export \
		--output type=local,dest=/tmp/parcagpu-build-amd64 \
		--platform linux/amd64 .
	@mkdir -p build/amd64
	@cp /tmp/parcagpu-build-amd64/$(LIB_NAME) build/amd64/
	@echo "AMD64 library built: build/amd64/$(LIB_NAME)"

# Build libparcagpucupti.so for ARM64 using Docker
build-arm64:
	@echo "=== Building $(LIB_NAME) for ARM64 with Docker ==="
	@mkdir -p /tmp/parcagpu-build-arm64
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target export \
		--output type=local,dest=/tmp/parcagpu-build-arm64 \
		--platform linux/arm64 .
	@mkdir -p build/arm64
	@cp /tmp/parcagpu-build-arm64/$(LIB_NAME) build/arm64/
	@echo "ARM64 library built: build/arm64/$(LIB_NAME)"

# Build both architectures
build-all: build-amd64 build-arm64
	@echo "=== All artifacts built ==="
	@echo "AMD64: build/amd64/$(LIB_NAME)"
	@echo "ARM64: build/arm64/$(LIB_NAME)"

# Build runtime container image for both architectures
# Multi-platform images stay in buildx cache. Use docker-push to push to registry.
cross:
	@echo "=== Building runtime container for AMD64 and ARM64 ==="
	@docker buildx create --name parcagpu-builder --use --bootstrap 2>/dev/null || docker buildx use parcagpu-builder
	@docker buildx build -f Dockerfile \
		--target runtime \
		--platform linux/amd64,linux/arm64 \
		.
	@echo "Runtime container built for both platforms (cached, not loaded into Docker)"

# Local build with CMake (for development/testing) - default is Release with symbols
local:
	@echo "=== Building locally with CMake (RelWithDebInfo) ==="
	@cmake -B build-local -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
	@cmake --build build-local
	@echo "Local build complete: build-local/lib/$(LIB_NAME)"

# Debug build with CMake (full debug, no optimizations)
debug:
	@echo "=== Building debug version with CMake ==="
	@cmake -B build-local -S . -DCMAKE_BUILD_TYPE=Debug
	@cmake --build build-local
	@echo "Debug build complete: build-local/lib/$(LIB_NAME)"

# Run local tests
test: local
	@echo "=== Running tests ==="
	@cd build-local && ctest --output-on-failure

# Clean build artifacts
clean:
	@echo "=== Cleaning build artifacts ==="
	@rm -rf build build-local bin lib zig-out .zig-cache
	@rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake compile_commands.json
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
docker-test-build: build-amd64
	@echo "=== Building test container image ==="
	@docker build -f Dockerfile.test -t parcagpu-test:latest .
	@echo "Test container built: parcagpu-test:latest"

# Run tests in container
# Pass arguments with ARGS variable (e.g., make docker-test-run ARGS="--forever")
docker-test-run: docker-test-build
	@echo "=== Running tests in container ==="
	@docker run --rm parcagpu-test:latest $(ARGS)

format:
	@echo "=== Formatting source files ==="
	@clang-format -i -style=file src/*.cpp src/*.h test/*.c
