#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Launch configurations and math kernels

// 1. RMSNorm (single vector of `size` elements; in-place is allowed)
void launch_rms_norm(float* out, float* in, float* weight, int size, float eps = 1e-5f, cudaStream_t stream = 0);

// 2. Rotary Positional Embeddings (RoPE)
// Rotates q [num_heads x head_dim] and k [num_kv_heads x head_dim] in place, for
// one token at seq_pos. head_dim must be even. This follows the Hugging Face
// Llama convention, which pairs element i with element i + head_dim / 2.
void launch_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, float theta = 10000.0f, cudaStream_t stream = 0);

// 3. Matrix Multiplication (GEMM) - block tiled
// C = A * B. A is [M x K], B is [K x N], C is [M x N]
// C must not alias A or B: tiles of A are re-read after other blocks write C.
void launch_gemm(float* C, float* A, float* B, int M, int N, int K, cudaStream_t stream = 0);

// 4. Softmax, in place
void launch_softmax(float* x, int size, cudaStream_t stream = 0);

// Applies softmax to num_vectors vectors of `size` elements, `stride` elements apart.
void launch_softmax_batch(float* x, int size, int num_vectors, int stride, cudaStream_t stream = 0);

// 5. SwiGLU Activation (for FFN): out = silu(gate) * up
void launch_swiglu(float* out, float* gate, float* up, int size, cudaStream_t stream = 0);

// 6. Element-wise add: out = a + b. out may alias a or b.
void launch_add(float* out, float* a, float* b, int size, cudaStream_t stream = 0);

// 7. Writes one token's keys or values, src [num_heads x head_dim], into a cache
// laid out as [num_heads x max_seq_len x head_dim].
void launch_kv_store(float* cache, float* src, int seq_pos, int num_heads, int head_dim, int max_seq_len, cudaStream_t stream = 0);

// 8. Scaled dot-product attention for one query per head, over the first seq_len
// cached positions. out and q are [num_heads x head_dim]. scores is scratch space
// of at least [num_heads x seq_len]. out must not alias q or the caches.
void launch_attention(float* out, float* q, float* k_cache, float* v_cache, float* scores, int num_heads, int head_dim, int max_seq_len, int seq_len, cudaStream_t stream = 0);

#endif // KERNELS_CUH
