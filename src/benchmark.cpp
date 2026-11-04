#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include "execution_engine.h"

// Measures greedy generation with the KV cache, one token at a time.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: benchmark <model_dir>" << std::endl;
        return 1;
    }
    std::cout << "--- BareMetal Inference Engine Benchmark ---" << std::endl;

    try {
        ModelLoader loader;
        if (!loader.load(argv[1])) return 1;
        const ModelConfig& config = loader.get_config();

        auto allocator = std::make_shared<MemoryAllocator>();
        allocator->allocate_and_copy_weights(loader.get_tensors());
        auto kv_cache = std::make_shared<KVCache>(1, config.max_seq_len, config.num_layers, config.num_heads, config.head_dim());
        ExecutionEngine engine(allocator, kv_cache, config);

        const auto next_token = [](const std::vector<float>& logits) {
            return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        };
        const int bos_token = 1;
        const int generated_tokens = std::min(100, config.max_seq_len);

        // Warm up: first launches pay for context setup and module loading
        int token = bos_token;
        for (int i = 0; i < 10; ++i) token = next_token(engine.forward({token}));
        engine.reset();

        // forward() copies the logits back to the host, which waits for the GPU,
        // so the host clock covers all of the work.
        token = bos_token;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < generated_tokens; ++i) token = next_token(engine.forward({token}));
        const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::cout << "\n--- Benchmark Results ---" << std::endl;
        std::cout << "Total Time: " << elapsed_s << " s" << std::endl;
        std::cout << "Tokens Generated: " << generated_tokens << std::endl;
        std::cout << "Latency / Token: " << elapsed_s * 1000.0 / generated_tokens << " ms" << std::endl;
        std::cout << "Tokens / Sec: " << generated_tokens / elapsed_s << " t/s" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
