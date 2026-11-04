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

1. `inference_engine`: Generates tokens greedily from a prompt of token IDs.
2. `benchmark`: A standalone tool that generates 100 tokens with a loaded model, reporting raw `Tokens/Sec`.
3. `server`: A placeholder for the API endpoint. It runs one simulated request through the engine and exits.

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

## Running a Model

The engine runs Llama-architecture models whose weights are in `model.safetensors`. These steps use `nickypro/tinyllama-15M`, an MIT-licensed model with 6 layers, 288 dimensions and a 32,000-token vocabulary. It is trained on children's stories.

```bash
pip install torch transformers safetensors numpy huggingface_hub

# Download the weights, pinned to one revision. Avoid the repository's pickle file.
hf download nickypro/tinyllama-15M model.safetensors config.json tokenizer.json tokenizer_config.json --revision a97e01fa7d54c088df9dc1b53d581a863ce08cd6 --local-dir models/tinyllama-15M

# Convert the weights to the engine's format
python scripts/export_llama.py models/tinyllama-15M models/tinyllama-15M/export

# Generate 40 tokens from "Once upon a time"
./build/Release/inference_engine.exe models/tinyllama-15M/export 40 1 9038 2501 263 931
```

The tokenizer does not implement BPE yet, so prompts and output are token IDs.

## Testing and Benchmarking

### 1. Run the Checks
The CPU checks run anywhere. The GPU checks compare each kernel with a CPU reference and need a CUDA device.

To also compare the engine's logits with Hugging Face `transformers`, write the reference file and then re-run CMake's configure step, which adds the `logits` test:

```bash
python scripts/reference_logits.py models/tinyllama-15M models/tinyllama-15M/export/reference_logits.bin
cmake -S . -B build
```

Then run the checks:

```bash
ctest --test-dir build -C Release -L cpu --output-on-failure
ctest --test-dir build -C Release -L gpu --output-on-failure
```

GitHub CI compiles every target and runs the CPU checks. Its standard runners have no GPU, so run the GPU checks locally.

The `logits` test feeds a whole sequence through the engine, using the KV cache, and compares every logit at every position with the reference. With the default 45-token sequence, the largest difference is 0.00005 and the most likely next token always matches.

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
| Multiply two 256 x 256 matrices | 18 us | 22 us |
| Multiply two 4096 x 4096 matrices | 28.8 ms | 3.5 ms |
| 32 layers of vector x 4096 x 4096 matrix | 156 passes/s | 347 passes/s |

These numbers cover matrix multiplies only, so neither one is a token rate.

### 3. The `benchmark` Program
`benchmark` generates 100 tokens greedily, one at a time, with the KV cache.

```bash
./build/Release/benchmark.exe models/tinyllama-15M/export
```

With `tinyllama-15M` on an RTX 5080, it reports about 800 tokens per second, or 1.2 ms per token. This has not been compared with another runtime on the same model yet.

### 4. Run the API Server

```bash
./build/Release/server.exe models/tinyllama-15M/export
```

The server does not open a socket yet. It processes one simulated request through `forward()` and exits.

## Current Status

The engine runs small Llama-architecture models, and its logits match Hugging Face `transformers`. It is still a work in progress.

| Component | Status |
| --- | --- |
| Forward pass and greedy generation | Works, one token at a time, for a batch of one |
| GEMM kernel | Works. Input and output must be separate buffers. |
| RMSNorm, RoPE, softmax, SwiGLU and attention kernels | Work |
| KV cache | Works |
| Weight loading | Works for 32-bit float exports. Quantized weights are not supported. |
| Tokenizer | Converts characters to byte values. BPE is not implemented. |
| Server | Simulates one request. No socket is opened. |
