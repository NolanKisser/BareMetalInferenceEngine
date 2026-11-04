#include "kernels.cuh"
#include "cuda_utils.h"
#include <float.h>
#include <math.h>

// --- Block-wide reductions ---
// These need a block whose thread count is a multiple of warpSize, so that
// every warp is full, and a shared array with one element per warp.
struct SumOp {
    __device__ static float apply(float a, float b) { return a + b; }
    __device__ static float identity() { return 0.0f; }
};

struct MaxOp {
    __device__ static float apply(float a, float b) { return fmaxf(a, b); }
    __device__ static float identity() { return -FLT_MAX; }
};

// Only lane 0 holds the complete result.
template <typename Op>
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = Op::apply(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Combines one value from every thread and returns the result to all of them.
template <typename Op>
__device__ __forceinline__ float block_reduce(float val, float* s_warp_values) {
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    val = warp_reduce<Op>(val);
    if (lane == 0) s_warp_values[warp] = val;
    __syncthreads();

    if (warp == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        val = (lane < num_warps) ? s_warp_values[lane] : Op::identity();
        val = warp_reduce<Op>(val);
        if (lane == 0) s_warp_values[0] = val;
    }
    __syncthreads();
    return s_warp_values[0];
}

static const int kReductionThreads = 1024;
static const int kElementThreads = 256;

static int blocks_for(int elements) {
    return (elements + kElementThreads - 1) / kElementThreads;
}


// --- RMSNorm ---
// A single block normalizes the whole vector so every thread sees the full
// sum of squares. Safe to run in place (out == in): each element is read and
// written by the same thread, and all reads of `in` finish before the writes.
__global__ void rms_norm_kernel(float* out, float* in, float* weight, int size, float eps) {
    __shared__ float s_warp_values[kReductionThreads / 32];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = in[i];
        sum += val * val;
    }
    sum = block_reduce<SumOp>(sum, s_warp_values);

    float rms = rsqrtf((sum / size) + eps);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = in[i] * rms * weight[i];
    }
}

void launch_rms_norm(float* out, float* in, float* weight, int size, float eps, cudaStream_t stream) {
    rms_norm_kernel<<<1, kReductionThreads, 0, stream>>>(out, in, weight, size, eps);
    CUDA_CHECK(cudaGetLastError());
}


// --- GEMM ---
#define TILE_SIZE 16
__global__ void gemm_kernel(float* C, float* A, float* B, int M, int N, int K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_gemm(float* C, float* A, float* B, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<blocks, threads, 0, stream>>>(C, A, B, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}


// --- RoPE ---
// One thread rotates one (x[i], x[i + head_dim / 2]) pair.
__global__ void rope_kernel(float* x, int seq_pos, int head_dim, int num_heads, float theta) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * half) return;

    int i = idx % half;
    float* head = x + (idx / half) * head_dim;
    float angle = seq_pos / powf(theta, 2.0f * i / head_dim);
    float s = sinf(angle);
    float c = cosf(angle);
    float first = head[i];
    float second = head[i + half];
    head[i] = first * c - second * s;
    head[i + half] = second * c + first * s;
}

void launch_rope(float* q, float* k, int seq_pos, int head_dim, int num_heads, int num_kv_heads, float theta, cudaStream_t stream) {
    int half = head_dim / 2;
    rope_kernel<<<blocks_for(num_heads * half), kElementThreads, 0, stream>>>(q, seq_pos, head_dim, num_heads, theta);
    CUDA_CHECK(cudaGetLastError());
    rope_kernel<<<blocks_for(num_kv_heads * half), kElementThreads, 0, stream>>>(k, seq_pos, head_dim, num_kv_heads, theta);
    CUDA_CHECK(cudaGetLastError());
}


// --- Softmax ---
// One block per vector. Subtracting the maximum keeps exp() from overflowing.
__global__ void softmax_kernel(float* x, int size, int stride) {
    __shared__ float s_warp_values[kReductionThreads / 32];
    float* vec = x + blockIdx.x * stride;

    float largest = MaxOp::identity();
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        largest = fmaxf(largest, vec[i]);
    }
    largest = block_reduce<MaxOp>(largest, s_warp_values);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        sum += expf(vec[i] - largest);
    }
    sum = block_reduce<SumOp>(sum, s_warp_values);

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        vec[i] = expf(vec[i] - largest) / sum;
    }
}

void launch_softmax(float* x, int size, cudaStream_t stream) {
    launch_softmax_batch(x, size, 1, size, stream);
}

void launch_softmax_batch(float* x, int size, int num_vectors, int stride, cudaStream_t stream) {
    softmax_kernel<<<num_vectors, kReductionThreads, 0, stream>>>(x, size, stride);
    CUDA_CHECK(cudaGetLastError());
}


// --- SwiGLU ---
__global__ void swiglu_kernel(float* out, float* gate, float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float g = gate[idx];
    out[idx] = g / (1.0f + expf(-g)) * up[idx];  // silu(g) = g * sigmoid(g)
}

void launch_swiglu(float* out, float* gate, float* up, int size, cudaStream_t stream) {
    swiglu_kernel<<<blocks_for(size), kElementThreads, 0, stream>>>(out, gate, up, size);
    CUDA_CHECK(cudaGetLastError());
}


// --- Element-wise add ---
__global__ void add_kernel(float* out, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

void launch_add(float* out, float* a, float* b, int size, cudaStream_t stream) {
    add_kernel<<<blocks_for(size), kElementThreads, 0, stream>>>(out, a, b, size);
    CUDA_CHECK(cudaGetLastError());
}


// --- KV cache write ---
__global__ void kv_store_kernel(float* cache, float* src, int seq_pos, int num_heads, int head_dim, int max_seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * head_dim) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    cache[(static_cast<size_t>(head) * max_seq_len + seq_pos) * head_dim + d] = src[idx];
}

void launch_kv_store(float* cache, float* src, int seq_pos, int num_heads, int head_dim, int max_seq_len, cudaStream_t stream) {
    kv_store_kernel<<<blocks_for(num_heads * head_dim), kElementThreads, 0, stream>>>(cache, src, seq_pos, num_heads, head_dim, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
}


// --- Scaled dot-product attention ---
// One block per head. Each thread scores a share of the cached positions.
__global__ void attention_scores_kernel(float* scores, float* q, float* k_cache, int head_dim, int max_seq_len, int seq_len) {
    int head = blockIdx.x;
    const float* query = q + head * head_dim;
    const float* keys = k_cache + static_cast<size_t>(head) * max_seq_len * head_dim;
    float scale = rsqrtf(static_cast<float>(head_dim));

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* key = keys + static_cast<size_t>(j) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * key[d];
        }
        scores[head * seq_len + j] = dot * scale;
    }
}

// One block per head. Each thread computes a share of the output elements.
__global__ void attention_mix_kernel(float* out, float* weights, float* v_cache, int head_dim, int max_seq_len, int seq_len) {
    int head = blockIdx.x;
    const float* head_weights = weights + head * seq_len;
    const float* values = v_cache + static_cast<size_t>(head) * max_seq_len * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            sum += head_weights[j] * values[static_cast<size_t>(j) * head_dim + d];
        }
        out[head * head_dim + d] = sum;
    }
}

void launch_attention(float* out, float* q, float* k_cache, float* v_cache, float* scores, int num_heads, int head_dim, int max_seq_len, int seq_len, cudaStream_t stream) {
    attention_scores_kernel<<<num_heads, kElementThreads, 0, stream>>>(scores, q, k_cache, head_dim, max_seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
    launch_softmax_batch(scores, seq_len, num_heads, seq_len, stream);
    attention_mix_kernel<<<num_heads, kElementThreads, 0, stream>>>(out, scores, v_cache, head_dim, max_seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
}
