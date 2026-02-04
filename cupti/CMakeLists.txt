cmake_minimum_required(VERSION 3.18)
project(parcagpucupti C CXX)

# CUDA root directory (can be overridden via -DCUDA_ROOT=...)
if(NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
endif()

# CUDA library directory
if(NOT DEFINED CUDA_LIBDIR)
    set(CUDA_LIBDIR "${CUDA_ROOT}/lib64" CACHE PATH "CUDA library directory")
endif()

# Set include directories for CUDA headers
set(CUDAToolkit_INCLUDE_DIRS
    "${CUDA_ROOT}/include"
    "${CUDA_ROOT}/extras/CUPTI/include"
)

# Add CUDA library directory to link directories
link_directories(${CUDA_LIBDIR})

# Create shared library with both C and C++ sources
add_library(parcagpucupti SHARED cupti-prof.c correlation_filter.cpp)

# Set properties
set_target_properties(parcagpucupti PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Add debug symbols and disable C++ exceptions
target_compile_options(parcagpucupti PRIVATE -g -fno-exceptions)

# Include directories
target_include_directories(parcagpucupti PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# systemtap headers are now in /usr/include/sys/ (copied in Dockerfile for cross-compilation)

# Find the CUPTI library explicitly
find_library(CUPTI_LIBRARY
    NAMES cupti
    PATHS ${CUDA_LIBDIR}
    REQUIRED
)

# Find the CUDA driver library
find_library(CUDA_LIBRARY
    NAMES cuda
    PATHS ${CUDA_LIBDIR} ${CUDA_LIBDIR}/stubs
    REQUIRED
)

# Link libraries
target_link_libraries(parcagpucupti PRIVATE
    ${CUPTI_LIBRARY}
    ${CUDA_LIBRARY}
)

# Installation
install(TARGETS parcagpucupti
    LIBRARY DESTINATION lib
)