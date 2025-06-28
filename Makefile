.PHONY: all clean mock-cudart bpf usdt_tracer libparcagpu

CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

# eBPF settings
CLANG = clang
BPF_CFLAGS = -target bpf -Wall -O2 -g -D__TARGET_ARCH_x86
BPF_TARGET_DIR = target/bpf
BPF_SRC = trace_kernel_launch.bpf.c
BPF_OBJ = $(BPF_TARGET_DIR)/trace_kernel_launch.bpf.o

all: mock-cudart bpf usdt_tracer libparcagpu

mock-cudart: libcudart.so

libcudart.so: mock_cudart.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Mock libcudart.so built successfully"

test_mock: test_mock.c
	$(CC) $(CFLAGS) -o $@ $< -L. -lcudart
	@echo "Test mock program built successfully"

libparcagpu:
	cargo build --release

usdt_tracer:
	cd usdt_tracer && go build -o usdt_tracer main.go && cd ..

# eBPF targets
bpf: $(BPF_OBJ)

$(BPF_TARGET_DIR):
	mkdir -p $(BPF_TARGET_DIR)

$(BPF_OBJ): $(BPF_SRC) | $(BPF_TARGET_DIR)
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@
	@echo "eBPF program compiled: $@"

clean:
	rm -f libcudart.so
	rm -rf $(BPF_TARGET_DIR)
	rm usdt_tracer/usdt_tracer