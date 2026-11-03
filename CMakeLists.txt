cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(BareMetalInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA ("native" targets the installed GPU; needs CMake 3.24+)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Find sources (excluding main executable entry points)
file(GLOB_RECURSE CORE_SOURCES 
    "src/tokenizer.cpp" 
    "src/weights.cpp" 
    "src/memory_allocator.cu" 
    "src/kv_cache.cu" 
    "src/kernels.cu" 
    "src/execution_engine.cu"
    "src/server.cpp"
)

# Executables
add_executable(inference_engine src/main.cpp ${CORE_SOURCES})
add_executable(benchmark src/benchmark.cpp ${CORE_SOURCES})
add_executable(server src/server_main.cpp ${CORE_SOURCES})

# Checks and benchmarks. These need only the code they exercise.
enable_testing()

add_executable(tokenizer_test tests/tokenizer_test.cpp src/tokenizer.cpp)
add_test(NAME tokenizer COMMAND tokenizer_test)
set_tests_properties(tokenizer PROPERTIES LABELS cpu)

add_executable(kernel_tests tests/kernel_tests.cu src/kernels.cu)
add_test(NAME kernels COMMAND kernel_tests)
set_tests_properties(kernels PROPERTIES LABELS gpu)

add_executable(gemm_bench benchmarks/gemm_bench.cu src/kernels.cu)
