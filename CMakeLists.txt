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

add_executable(loader_test tests/loader_test.cpp src/weights.cpp)
add_test(NAME loader COMMAND loader_test)
set_tests_properties(loader PROPERTIES LABELS cpu)

# Compares logits with a reference model. Needs the export described in the README.
add_executable(logits_test tests/logits_test.cu src/weights.cpp src/memory_allocator.cu src/kv_cache.cu src/kernels.cu src/execution_engine.cu)
set(BAREMETAL_MODEL_DIR "${CMAKE_SOURCE_DIR}/models/tinyllama-15M/export" CACHE PATH "Exported model that the logits test uses")
if(EXISTS "${BAREMETAL_MODEL_DIR}/reference_logits.bin")
    add_test(NAME logits COMMAND logits_test "${BAREMETAL_MODEL_DIR}")
    set_tests_properties(logits PROPERTIES LABELS "gpu;model")
endif()
