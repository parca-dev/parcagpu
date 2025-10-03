cmake_minimum_required(VERSION 3.18)
project(parcagpucupti C)

# CUDA root directory (can be overridden via -DCUDA_ROOT=...)
if(NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
endif()

# CUDA stub library directory (for linking without actual GPU libraries)
if(NOT DEFINED CUDA_STUB_LIBDIR)
    set(CUDA_STUB_LIBDIR "${CUDA_ROOT}/lib64/stubs" CACHE PATH "CUDA stub library directory")
endif()

# Set include directories for CUDA headers
set(CUDAToolkit_INCLUDE_DIRS
    "${CUDA_ROOT}/include"
    "${CUDA_ROOT}/extras/CUPTI/include"
)

# Add stub library directory to link directories
link_directories(${CUDA_STUB_LIBDIR})

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

# Link libraries by name (using stub libraries from CUDA_STUB_LIBDIR)
target_link_libraries(parcagpucupti PRIVATE
    cupti
)

# Installation
install(TARGETS parcagpucupti
    LIBRARY DESTINATION lib
)