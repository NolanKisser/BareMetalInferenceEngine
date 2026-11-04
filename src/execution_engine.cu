#include "execution_engine.h"
#include "kernels.cuh"
#include <iostream>
#include <stdexcept>

ExecutionEngine::ExecutionEngine(std::shared_ptr<MemoryAllocator> allocator, std::shared_ptr<KVCache> kv_cache, const ModelConfig& config)
    : allocator_(allocator), kv_cache_(kv_cache), config_(config)
{
    token_embeddings_ = weight("model.embed_tokens.weight");
    final_norm_ = weight("model.norm.weight");
    lm_head_ = weight("lm_head.weight");

    for (int l = 0; l < config_.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        layers_.push_back({
            weight(prefix + "input_layernorm.weight"),
            weight(prefix + "self_attn.q_proj.weight"),
            weight(prefix + "self_attn.k_proj.weight"),
            weight(prefix + "self_attn.v_proj.weight"),
            weight(prefix + "self_attn.o_proj.weight"),
            weight(prefix + "post_attention_layernorm.weight"),
            weight(prefix + "mlp.gate_proj.weight"),
            weight(prefix + "mlp.up_proj.weight"),
            weight(prefix + "mlp.down_proj.weight"),
        });
    }

    const size_t dim = config_.dim;
    const size_t hidden_dim = config_.hidden_dim;
    hidden_ = new_buffer(dim);
    normed_ = new_buffer(dim);
    q_ = new_buffer(dim);
    k_ = new_buffer(dim);
    v_ = new_buffer(dim);
    attention_out_ = new_buffer(dim);
    scores_ = new_buffer(static_cast<size_t>(config_.num_heads) * kv_cache_->get_max_seq_len());
    projected_ = new_buffer(dim);
    gate_ = new_buffer(hidden_dim);
    up_ = new_buffer(hidden_dim);
    activated_ = new_buffer(hidden_dim);
    logits_ = new_buffer(config_.vocab_size);
}

ExecutionEngine::~ExecutionEngine() {}

float* ExecutionEngine::weight(const std::string& name) const {
    float* tensor = static_cast<float*>(allocator_->get_device_tensor(name));
    if (!tensor) throw std::runtime_error("Model is missing tensor " + name);
    return tensor;
}

float* ExecutionEngine::new_buffer(size_t elements) {
    buffers_.push_back(std::make_shared<DeviceBuffer<float>>(elements));
    return buffers_.back()->get();
}

void ExecutionEngine::reset() {
    kv_cache_->reset();
}

std::vector<float> ExecutionEngine::forward(const std::vector<int>& input_tokens) {
    if (input_tokens.empty()) throw std::invalid_argument("forward() needs at least one token");
    for (int token : input_tokens) {
        if (token < 0 || token >= config_.vocab_size) throw std::invalid_argument("Token " + std::to_string(token) + " is outside the vocabulary");
    }
    const size_t end_pos = static_cast<size_t>(kv_cache_->get_current_seq_pos()) + input_tokens.size();
    if (end_pos > static_cast<size_t>(kv_cache_->get_max_seq_len())) {
        throw std::runtime_error("Sequence would exceed the KV cache's " + std::to_string(kv_cache_->get_max_seq_len()) + " positions");
    }

    if (verbose_) std::cout << "Forward pass for " << input_tokens.size() << " tokens" << std::endl;

    // One token at a time: the KV cache supplies earlier positions, so a token can
    // only attend to itself and what came before it. No attention mask is needed.
    for (int token : input_tokens) {
        run_layers(token);
        kv_cache_->increment_seq_pos(1);
    }

    launch_rms_norm(normed_, hidden_, final_norm_, config_.dim, config_.norm_eps);
    launch_gemm(logits_, normed_, lm_head_, 1, config_.vocab_size, config_.dim);

    // A synchronous copy, so the GPU has finished everything queued above.
    std::vector<float> logits(config_.vocab_size);
    CUDA_CHECK(cudaMemcpy(logits.data(), logits_, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return logits;
}

void ExecutionEngine::run_layers(int token) {
    const int dim = config_.dim;
    const int hidden_dim = config_.hidden_dim;
    const int heads = config_.num_heads;
    const int head_dim = config_.head_dim();
    const int max_seq_len = kv_cache_->get_max_seq_len();
    const int seq_pos = kv_cache_->get_current_seq_pos();
    const float eps = config_.norm_eps;

    const float* embedding = token_embeddings_ + static_cast<size_t>(token) * dim;
    CUDA_CHECK(cudaMemcpy(hidden_, embedding, dim * sizeof(float), cudaMemcpyDeviceToDevice));

    // The export stores each projection as [in x out], so a row vector times the
    // weight matrix gives the same result as the usual W * x.
    for (int l = 0; l < config_.num_layers; ++l) {
        const LayerWeights& w = layers_[l];
        float* k_cache = kv_cache_->get_key_cache(l);
        float* v_cache = kv_cache_->get_value_cache(l);

        // Attention, with a residual connection
        launch_rms_norm(normed_, hidden_, w.attention_norm, dim, eps);
        launch_gemm(q_, normed_, w.q_proj, 1, dim, dim);
        launch_gemm(k_, normed_, w.k_proj, 1, dim, dim);
        launch_gemm(v_, normed_, w.v_proj, 1, dim, dim);
        launch_rope(q_, k_, seq_pos, head_dim, heads, heads, config_.rope_theta);
        launch_kv_store(k_cache, k_, seq_pos, heads, head_dim, max_seq_len);
        launch_kv_store(v_cache, v_, seq_pos, heads, head_dim, max_seq_len);
        launch_attention(attention_out_, q_, k_cache, v_cache, scores_, heads, head_dim, max_seq_len, seq_pos + 1);
        launch_gemm(projected_, attention_out_, w.o_proj, 1, dim, dim);
        launch_add(hidden_, hidden_, projected_, dim);

        // Feed-forward, with a residual connection
        launch_rms_norm(normed_, hidden_, w.ffn_norm, dim, eps);
        launch_gemm(gate_, normed_, w.gate_proj, 1, hidden_dim, dim);
        launch_gemm(up_, normed_, w.up_proj, 1, hidden_dim, dim);
        launch_swiglu(activated_, gate_, up_, hidden_dim);
        launch_gemm(projected_, activated_, w.down_proj, 1, dim, hidden_dim);
        launch_add(hidden_, hidden_, projected_, dim);
    }
}
