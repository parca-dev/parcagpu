cmake_minimum_required(VERSION 3.18)
project(cupti-prof C)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Add CUPTI library directory to link directories
link_directories(/usr/local/cuda-12.9/extras/CUPTI/lib64)

# Create shared library
add_library(cupti-prof SHARED cupti-prof.c)

# Set properties
set_target_properties(cupti-prof PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Add debug symbols
target_compile_options(cupti-prof PRIVATE -g)

# Include directories
target_include_directories(cupti-prof PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/cuda-12.9/extras/CUPTI/include
)

# Link libraries
target_link_libraries(cupti-prof PRIVATE
    CUDA::cupti
    CUDA::cuda_driver
    nvperf_host
)

# Installation
install(TARGETS cupti-prof
    LIBRARY DESTINATION lib
)