.PHONY: all clean test cupti test-infra

# Default target: build everything
all: cupti test-infra

# Build libparcagpucupti.so with CMake
cupti:
	@echo "=== Building libparcagpucupti.so with CMake ==="
	@mkdir -p cupti/build
	@cd cupti/build && cmake .. && $(MAKE)

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
	@rm -rf cupti/build
	@rm -rf zig-out
	@rm -rf .zig-cache
	@echo "Clean complete"
