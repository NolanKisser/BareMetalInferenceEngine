#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "memory_allocator.h"
#include "kv_cache.h"
#include "weights.h"
#include <vector>
#include <memory>
#include <string>

// Runs a Llama-architecture model. The constructor throws std::runtime_error if
// the allocator is missing a tensor the model needs.
class ExecutionEngine {
public:
    ExecutionEngine(std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<KVCache> kv_cache, const ModelConfig& config);
    ~ExecutionEngine();

    // Feeds tokens through the model, extending the sequence held in the KV
    // cache, and returns the logits that follow the last token. Throws
    // std::invalid_argument for bad tokens, or std::runtime_error when the
    // sequence would exceed the KV cache.
    std::vector<float> forward(const std::vector<int>& input_tokens);

    // Starts a new sequence.
    void reset();

    // Toggle per-pass logging
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    struct LayerWeights {
        float* attention_norm;
        float* q_proj;
        float* k_proj;
        float* v_proj;
        float* o_proj;
        float* ffn_norm;
        float* gate_proj;
        float* up_proj;
        float* down_proj;
    };

    float* weight(const std::string& name) const;
    float* new_buffer(size_t elements);
    void run_layers(int token);

    std::shared_ptr<MemoryAllocator> allocator_;
    std::shared_ptr<KVCache> kv_cache_;
    ModelConfig config_;

    float* token_embeddings_ = nullptr;
    float* final_norm_ = nullptr;
    float* lm_head_ = nullptr;
    std::vector<LayerWeights> layers_;

    // Intermediate activation buffers (FP32)
    std::vector<std::shared_ptr<DeviceBuffer<float>>> buffers_;
    float* hidden_ = nullptr;
    float* normed_ = nullptr;
    float* q_ = nullptr;
    float* k_ = nullptr;
    float* v_ = nullptr;
    float* attention_out_ = nullptr;
    float* scores_ = nullptr;
    float* projected_ = nullptr;
    float* gate_ = nullptr;
    float* up_ = nullptr;
    float* activated_ = nullptr;
    float* logits_ = nullptr;

    bool verbose_ = false;
};

#endif // EXECUTION_ENGINE_H
