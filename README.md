# BareMetal Inference Engine

A dependency-free, high-throughput LLM inference pipeline built entirely from scratch in C++ and CUDA. 

This project was designed as an educational, bare-metal approach to serving Large Language Models (like Llama-3) on NVIDIA GPUs without relying on massive Python ecosystems or heavy third-party C++ libraries (no PyTorch, no cuBLAS, no libTorch, no gRPC).

## Features

- **Custom BPE Tokenizer**: Reads vocabulary maps and performs Byte-Pair Encoding directly in C++.
- **Direct Weight Loading**: Maps raw binary tensor files straight into host memory.
- **VRAM Memory Management**: Custom RAII wrappers for `cudaMalloc` and device pointer lifecycle handling.
- **Static KV Caching**: Pre-allocates Keys and Values in VRAM based on the maximum sequence length to dramatically speed up token generation.
- **Custom CUDA Kernels**: Hand-written block-tiled `__global__` kernels for RMSNorm, RoPE, Matrix Multiplication (GEMM), and SwiGLU activations.
- **Dependency-Free API**: A bare-bones POSIX/Winsock server endpoint to accept inference requests.

## Project Structure

The project builds 3 distinct executable targets:

1. `inference_engine`: The core executable for testing tokenization and weight loading.
2. `benchmark`: A standalone tool that bypasses the network layer to stream 100 tokens as fast as possible, reporting raw `Tokens/Sec`.
3. `server`: A socket-based API endpoint that listens for incoming generation requests and routes them through the continuous batching/execution engine.

## Prerequisites

To compile and run this project, you must have the following installed and available in your system's `PATH`:

1.  **CMake** (version 3.18 or higher)
2.  **C++17 Compiler** (e.g., MSVC `cl.exe` for Windows, or `g++` for Linux)
3.  **NVIDIA CUDA Toolkit** (`nvcc` compiler)

> **Windows Users**: It is highly recommended to run the build commands from inside the **Developer Command Prompt for Visual Studio**, as this automatically configures the environment variables for `cl` and `cmake`.

### Common Visual Studio Troubleshooting
If CMake throws an error inside Visual Studio (`CMake Error at ... CMakeDetermineCompilerId.cmake` or failing to find `cuda_runtime.h`):
- **Missing C++ Workload**: Ensure you checked **"Desktop development with C++"** in the Visual Studio Installer.
- **Missing CUDA Toolkit**: Ensure you downloaded the **NVIDIA CUDA Toolkit**. Visual Studio does *not* come with it.
- **Stale CMake Cache**: If you installed CUDA *after* opening Visual Studio, CMake will remember its previous failure. Go to **Project > CMake Cache > Delete Cache** and regenerate the project.

## Building the Engine

Clone the repository and build using standard CMake commands:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

*(Note: If building via Visual Studio's CMake integration, simply open the folder and use the "Build All" or "F5" functionality).*

## Testing and Benchmarking

### 1. Run the Checks
The CPU checks run anywhere. The GPU checks compare each kernel with a CPU reference and need a CUDA device.

```bash
ctest --test-dir build -C Release -L cpu --output-on-failure
ctest --test-dir build -C Release -L gpu --output-on-failure
```

GitHub CI compiles every target and runs the CPU checks. Its standard runners have no GPU, so run the GPU checks locally. **6 of the 13 GPU checks currently fail**, for the reasons listed under Current Status.

### 2. Benchmark the GEMM Kernel Against PyTorch
`gemm_bench` and `benchmarks/pytorch_gemm.py` run the same matrix multiplies on the same GPU and write their results as JSON.

```bash
./build/Release/gemm_bench.exe baremetal.json
pip install torch --index-url https://download.pytorch.org/whl/cu128
python benchmarks/pytorch_gemm.py pytorch.json
```

RTX 50 series GPUs need a PyTorch build that supports `sm_120`, such as the `cu128` build above.

Results on an RTX 5080 with PyTorch 2.11.0, from `benchmarks/results/`:

| Workload | BareMetal | PyTorch |
| --- | --- | --- |
| Multiply two 256 x 256 matrices | 19 us | 24 us |
| Multiply two 4096 x 4096 matrices | 28.8 ms | 3.5 ms |
| 32 layers of vector x 4096 x 4096 matrix | 158 passes/s | 342 passes/s |

These numbers cover matrix multiplies only, so neither one is a token rate.

### 3. The `benchmark` Program
`benchmark` prints a tokens-per-second figure, but it does not measure inference. It loads no weights, so each token costs only 32 empty kernel launches, some console output and a vector of zeros. Don't use this figure as a measure of speed.

### 4. Run the API Server

```bash
./build/Release/server.exe
```

The server does not open a socket yet. It processes one simulated request through `forward()` and exits.

## Current Status

This project is a work in progress and does not run a model yet.

| Component | Status |
| --- | --- |
| GEMM kernel | Works with separate buffers. Gives wrong results in place, which is how `forward()` calls it. |
| RMSNorm kernel | Correct up to 256 elements. Larger vectors are wrong because the sum is per GPU block. |
| RoPE, softmax and SwiGLU kernels | Empty |
| Attention, embeddings and output projection | Not implemented |
| VRAM allocation and weight copy | Works |
| KV cache | Allocated, but never read or written |
| Weight loader | Fills tensors with zeros and does not read the file |
| Tokenizer | Converts characters to byte values. BPE is not implemented. |
| Server | Simulates one request. No socket is opened. |
