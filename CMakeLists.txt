cmake_minimum_required(VERSION 3.18)
project(parcagpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release with debug symbols
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g -O2")

# Set proton directory (git submodule or Docker build)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/proton")
    set(PROTON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proton")
elseif(EXISTS "/build/proton")
    set(PROTON_DIR "/build/proton")
else()
    message(FATAL_ERROR "Could not find proton directory. Expected at ${CMAKE_CURRENT_SOURCE_DIR}/proton or /build/proton")
endif()
message(STATUS "Using Proton from: ${PROTON_DIR}")

# Find CUDA headers (allow fallback to manual path for header-only builds)
if(NOT DEFINED CUDA_INCLUDE_DIR)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
    else()
        # Fallback for header-only builds (e.g., Docker with ghcr.io/parca-dev/cuda-headers)
        set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
        if(NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda.h")
            message(FATAL_ERROR "Could not find CUDA headers. Set CUDA_INCLUDE_DIR or install CUDA Toolkit.")
        endif()
        message(STATUS "Using CUDA headers from: ${CUDA_INCLUDE_DIR}")
    endif()
endif()

# Include directories from proton
include_directories(
    ${PROTON_DIR}/csrc/include
    ${PROTON_DIR}/common/include
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Collect proton driver source files we need
# Note: We only link driver APIs and callback utilities (no profiler dependencies)
set(PROTON_SOURCES
    ${PROTON_DIR}/csrc/lib/Driver/GPU/CuptiApi.cpp
    ${PROTON_DIR}/csrc/lib/Driver/GPU/CudaApi.cpp
    ${PROTON_DIR}/csrc/lib/Profiler/Cupti/CuptiCallbacks.cpp
)

# Build the shared library
add_library(parcagpucupti SHARED
    src/parcagpu_cupti.cpp
    src/parcagpu_pc_sampling.cpp
    src/correlation_filter.cpp
    ${PROTON_SOURCES}
)

# Link against system libraries only (NOT libcupti - loaded dynamically by Proton)
target_link_libraries(parcagpucupti
    dl
    pthread
)

# Set output directory
set_target_properties(parcagpucupti PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "parcagpucupti"
)

# Install target
install(TARGETS parcagpucupti
    LIBRARY DESTINATION lib
)

# Build test infrastructure
option(BUILD_TESTS "Build test infrastructure" ON)

if(BUILD_TESTS)
    # Mock CUDA library for testing (minimal - just cuDriverGetVersion)
    add_library(cuda_mock SHARED
        test/mock_cuda.c
    )
    target_include_directories(cuda_mock PRIVATE
        ${CUDA_INCLUDE_DIR}
    )
    set_target_properties(cuda_mock PROPERTIES
        OUTPUT_NAME "cuda"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

    # Mock CUPTI library for testing
    add_library(cupti_mock SHARED
        test/mock_cupti.c
    )
    target_include_directories(cupti_mock PRIVATE
        ${CUDA_INCLUDE_DIR}
    )
    set_target_properties(cupti_mock PROPERTIES
        OUTPUT_NAME "cupti"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

    # Test executable
    add_executable(test_cupti_prof
        test/test_cupti_prof.c
    )
    target_include_directories(test_cupti_prof PRIVATE
        ${CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_cupti_prof
        dl
    )
    set_target_properties(test_cupti_prof PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Create libcuda.so.1 symlink for test (Proton expects libcuda.so.1)
    add_custom_command(TARGET cuda_mock POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libcuda.so
            ${CMAKE_BINARY_DIR}/lib/libcuda.so.1
        COMMENT "Creating libcuda.so.1 symlink"
    )

    # Create libcupti.so.12 symlink for test
    add_custom_command(TARGET cupti_mock POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libcupti.so
            ${CMAKE_BINARY_DIR}/lib/libcupti.so.12
        COMMENT "Creating libcupti.so.12 symlink"
    )
endif()
