.PHONY: all clean mock-cudart

CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

all: mock-cudart

mock-cudart: libcudart.so

libcudart.so: mock_cudart.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Mock libcudart.so built successfully"

clean:
	rm -f libcudart.so

test-with-mock: libcudart.so
	LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH) cargo test

run-with-mock: libcudart.so
	LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH) cargo build --release