#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include "execution_engine.h"

// Measures greedy generation with the KV cache, one token at a time.
// benchmarks/pytorch_tokens.py runs the same workload and writes the same JSON layout.
//
// Usage: benchmark <model_dir> [output.json]
int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: benchmark <model_dir> [output.json]" << std::endl;
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
        const int repetitions = 20;

        // Warm up: first launches pay for context setup and module loading
        int token = bos_token;
        for (int i = 0; i < 10; ++i) token = next_token(engine.forward({token}));
        engine.reset();

        std::vector<int> tokens(generated_tokens);
        std::vector<double> samples;
        for (int rep = 0; rep < repetitions; ++rep) {
            // forward() copies the logits back to the host, which waits for the GPU,
            // so the host clock covers all of the work.
            token = bos_token;
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < generated_tokens; ++i) {
                token = next_token(engine.forward({token}));
                tokens[i] = token;
            }
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            engine.reset();
        }

        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double elapsed_s = (sorted[(repetitions - 1) / 2] + sorted[repetitions / 2]) / 2.0;

        std::cout << "\n--- Benchmark Results ---" << std::endl;
        std::cout << "Median of " << repetitions << " repetitions" << std::endl;
        std::cout << "Total Time: " << elapsed_s << " s" << std::endl;
        std::cout << "Tokens Generated: " << generated_tokens << std::endl;
        std::cout << "Latency / Token: " << elapsed_s * 1000.0 / generated_tokens << " ms" << std::endl;
        std::cout << "Tokens / Sec: " << generated_tokens / elapsed_s << " t/s" << std::endl;

        if (argc == 3) {
            cudaDeviceProp device{};
            CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
            FILE* file = std::fopen(argv[2], "w");
            if (!file) {
                std::cerr << "Cannot write " << argv[2] << std::endl;
                return 1;
            }
            std::fprintf(file, "{\n  \"engine\": \"baremetal\",\n  \"device\": \"%s\",\n  \"generated_tokens\": %d,\n", device.name, generated_tokens);
            std::fprintf(file, "  \"median_s\": %.9f,\n  \"tokens_per_s\": %.3f,\n  \"samples\": [", elapsed_s, generated_tokens / elapsed_s);
            for (size_t i = 0; i < samples.size(); ++i) std::fprintf(file, "%s%.9f", i ? ", " : "", samples[i]);
            std::fprintf(file, "],\n  \"tokens\": [");
            for (size_t i = 0; i < tokens.size(); ++i) std::fprintf(file, "%s%d", i ? ", " : "", tokens[i]);
            std::fprintf(file, "]\n}\n");
            std::fclose(file);
            std::cout << "Wrote " << argv[2] << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
