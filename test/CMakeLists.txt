cmake_minimum_required(VERSION 3.18)
project(parcagpu_test C)

# CUDA include paths
if(NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
endif()

set(CUDA_INCLUDE_DIRS
    "${CUDA_ROOT}/include"
    "${CUDA_ROOT}/extras/CUPTI/include"
)

# Mock CUPTI library
add_library(cupti SHARED mock_cupti.c)
target_include_directories(cupti PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(cupti PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA major version - prefer explicitly passed value, fall back to detection
if(NOT DEFINED CUDA_MAJOR_VERSION)
    execute_process(
        COMMAND bash -c "ls ${CUDA_ROOT}/lib64/libcupti.so.* 2>/dev/null | grep -oE 'libcupti\\.so\\.[0-9]+$' | head -1 | grep -oE '[0-9]+$'"
        OUTPUT_VARIABLE CUDA_MAJOR_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT CUDA_MAJOR_VERSION)
        set(CUDA_MAJOR_VERSION "12")
    endif()
    message(STATUS "Detected CUDA major version: ${CUDA_MAJOR_VERSION}")
else()
    message(STATUS "Using specified CUDA major version: ${CUDA_MAJOR_VERSION}")
endif()

# Create versioned symlink matching installed CUDA
add_custom_command(TARGET cupti POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    $<TARGET_FILE_NAME:cupti>
    ${CMAKE_CURRENT_BINARY_DIR}/libcupti.so.${CUDA_MAJOR_VERSION}
)

# Test executable
add_executable(test_cupti_prof test_cupti_prof.c)
target_include_directories(test_cupti_prof PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_options(test_cupti_prof PRIVATE
    -D_POSIX_C_SOURCE=199309L
    -Wall
    -Wextra
)
target_link_libraries(test_cupti_prof PRIVATE dl pthread)
set_target_properties(test_cupti_prof PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Installation
install(TARGETS cupti test_cupti_prof
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
