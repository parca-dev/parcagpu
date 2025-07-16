.PHONY: all clean mock-cudart mock-cupti bpf gotracer usdt_tracer_libbpf libparcagpu cupti

CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

# eBPF settings
CLANG = clang
BPF_CFLAGS = -target bpf -Wall -O2 -g -D__TARGET_ARCH_x86
BPF_TARGET_DIR = target/bpf
BPF_SRC = trace_kernel_launch.bpf.c trace_cupti_launch.bpf.c
BPF_OBJ = $(BPF_TARGET_DIR)/trace_kernel_launch.bpf.o $(BPF_TARGET_DIR)/trace_cupti_launch.bpf.o

all: mock-cudart mock-cupti bpf gotracer usdt_tracer_libbpf libparcagpu test_mock

mock-cudart: libcudart.so

libcudart.so: mock_cudart.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Mock libcudart.so built successfully"

cupti: libparcagpucupti.so

libparcagpucupti.so: cupti/cupti-prof.c cupti/CMakeLists.txt
	@mkdir -p cupti/build
	@cd cupti/build && cmake .. && make
	@cp cupti/build/libcupti-prof.so libparcagpucupti.so
	@echo "libparcagpucupti.so built successfully using CMake"

test_mock: test_mock.c
	$(CC) $(CFLAGS) -o $@ $< -L. -lcudart
	@echo "Test mock program built successfully"

test_cupti_prof: test_cupti_prof.cu
	nvcc -ccbin clang-14 -g -G -o $@ $< -lcuda -lcudart
	@echo "Test CUPTI profiling program built successfully"

libparcagpu:
	cargo build --release

gotracer:
	cd gotracer && go build -o gotracer main.go && cd ..

usdt_tracer_libbpf:
	cd usdt_tracer_libbpf && CGO_CFLAGS="-I/home/tpr/src/libbpf/src -I/home/tpr/src/libbpf/include/uapi" CGO_LDFLAGS="-L/home/tpr/src/libbpf/src -lbpf" go build -o usdt_tracer_libbpf main_libbpf.go && cd ..

# eBPF targets
bpf: $(BPF_OBJ)

$(BPF_TARGET_DIR):
	mkdir -p $(BPF_TARGET_DIR)

$(BPF_TARGET_DIR)/%.bpf.o: %.bpf.c | $(BPF_TARGET_DIR)
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@
	@echo "eBPF program compiled: $@"

clean:
	rm -f libcudart.so libparcagpucupti.so test_mock test_cupti
	rm -rf $(BPF_TARGET_DIR)
	rm -f gotracer/gotracer
	rm -f usdt_tracer_libbpf/usdt_tracer_libbpf
	rm -rf cupti/build