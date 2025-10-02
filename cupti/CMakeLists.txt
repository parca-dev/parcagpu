cmake_minimum_required(VERSION 3.18)
project(parcagpucupti C)

# Detect cross-compilation
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for ${CMAKE_SYSTEM_PROCESSOR}")
    # For cross-compilation, use host CUDA headers (arch-independent) and target libraries
    if(NOT DEFINED CUDA_HOST_ROOT)
        set(CUDA_HOST_ROOT "/usr/local/cuda" CACHE PATH "Host CUDA installation for headers")
    endif()
    if(NOT DEFINED CUDA_CROSS_ROOT)
        set(CUDA_CROSS_ROOT "/cuda-arm64/usr/local/cuda-12.2" CACHE PATH "Target CUDA installation for libraries")
    endif()
    # Headers are architecture-independent, use host system's CUDA
    set(CUDAToolkit_INCLUDE_DIRS "${CUDA_HOST_ROOT}/include" "${CUDA_HOST_ROOT}/extras/CUPTI/include")
    # Libraries are architecture-specific, use ARM64 versions
    set(CUPTI_LIBRARY_DIR "${CUDA_CROSS_ROOT}/extras/CUPTI/lib64")
else()
    # Native build: find CUDA Toolkit normally
    find_package(CUDAToolkit REQUIRED)
    set(CUPTI_LIBRARY_DIR "/usr/local/cuda-12.9/extras/CUPTI/lib64")
endif()

# Add CUPTI library directory to link directories
link_directories(${CUPTI_LIBRARY_DIR})

# Create shared library
add_library(parcagpucupti SHARED cupti-prof.c)

# Set properties
set_target_properties(parcagpucupti PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Add debug symbols
target_compile_options(parcagpucupti PRIVATE -g)

# Include directories
target_include_directories(parcagpucupti PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# systemtap headers are now in /usr/include/sys/ (copied in Dockerfile for cross-compilation)

# Link libraries
if(CMAKE_CROSSCOMPILING)
    # For cross-compilation, link libraries by name
    target_link_libraries(parcagpucupti PRIVATE
        cupti
        cuda
        nvperf_host
    )
else()
    # Native build: use imported targets
    target_link_libraries(parcagpucupti PRIVATE
        CUDA::cupti
        CUDA::cuda_driver
        nvperf_host
    )
endif()

# Installation
install(TARGETS parcagpucupti
    LIBRARY DESTINATION lib
)