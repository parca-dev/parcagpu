.PHONY: all clean cupti

all: cupti

cupti: libparcagpucupti.so

libparcagpucupti.so: cupti/cupti-prof.c cupti/CMakeLists.txt
	@mkdir -p cupti/build
	@cd cupti/build && cmake .. && make
	@cp cupti/build/libparcagpucupti.so libparcagpucupti.so
	@echo "libparcagpu.so built successfully using CMake"

clean:
	rm -f libparcagpucupti.so
	rm -rf cupti/build