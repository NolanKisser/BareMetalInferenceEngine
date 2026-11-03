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

1.  **CMake** (version 3.24 or higher)
2.  **C++17 Compiler** (e.g., MSVC `cl.exe` for Windows, or `g++` for Linux)
3.  **NVIDIA CUDA Toolkit** (`nvcc` compiler). RTX 50-series (Blackwell, sm_120) GPUs need CUDA 12.8 or newer.

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

### 1. Run the BareMetal GPU Benchmark
Use the built-in benchmark tool to profile the speed of our custom CUDA kernels. It simulates a 32-layer transformer matrix multiplication loop to generate 100 tokens and outputs the raw `Tokens / Sec` hardware limit.

```bash
# Via command line
./build/Release/benchmark.exe

# Or via Visual Studio
Select "benchmark.exe" as your Startup Item and press F5
```
*Tip: To analyze the memory bandwidth and SM occupancy of the custom block-tiled GEMM kernels, run this under NVIDIA Nsight Compute:*
`ncu --set full ./benchmark.exe`

### 2. Compare Efficiency against PyTorch
To truly prove the efficiency of this BareMetal engine, we have included a `pytorch_benchmark.py` script. This script simulates the exact same 32-layer matrix multiplications using the Python/PyTorch backend.

To compare the speed:
1. Ensure you have Python and PyTorch installed (`pip install torch`).
2. Run the baseline comparison:
```bash
python pytorch_benchmark.py
```
By comparing the `Tokens / Sec` of `pytorch_benchmark.py` against `benchmark.exe`, you can quantify the efficiency gained by stripping out the Python Global Interpreter Lock (GIL) and framework overhead!

### 3. Start the API Server
Launch the lightweight socket server to listen for inference requests.

```bash
./build/Release/server.exe
```
Once running, the server will block and listen on port `8080`. You can simulate sending a payload to it, and the server will execute the forward pass and print the generation status.
