.PHONY: all clean mock-cudart bpf

CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

# eBPF settings
CLANG = clang
BPF_CFLAGS = -target bpf -Wall -O2 -g -D__TARGET_ARCH_x86
BPF_TARGET_DIR = target/bpf
BPF_SRC = trace_kernel_launch.bpf.c
BPF_OBJ = $(BPF_TARGET_DIR)/trace_kernel_launch.bpf.o

all: mock-cudart bpf

mock-cudart: libcudart.so

libcudart.so: mock_cudart.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Mock libcudart.so built successfully"

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