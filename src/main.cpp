#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include "execution_engine.h"

// Generates tokens greedily from a prompt of token IDs. The tokenizer does not
// implement BPE yet, so prompts and output are token IDs.
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: inference_engine <model_dir> <new_tokens> <token_id> [token_id ...]" << std::endl;
        return 1;
    }

    try {
        ModelLoader loader;
        if (!loader.load(argv[1])) return 1;
        const ModelConfig& config = loader.get_config();

        const int new_tokens = std::stoi(argv[2]);
        std::vector<int> prompt;
        for (int i = 3; i < argc; ++i) prompt.push_back(std::stoi(argv[i]));

        auto allocator = std::make_shared<MemoryAllocator>();
        allocator->allocate_and_copy_weights(loader.get_tensors());
        auto kv_cache = std::make_shared<KVCache>(1, config.max_seq_len, config.num_layers, config.num_heads, config.head_dim());
        ExecutionEngine engine(allocator, kv_cache, config);

        std::vector<float> logits = engine.forward(prompt);
        std::cout << "Generated token IDs:";
        for (int i = 0; i < new_tokens; ++i) {
            const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::cout << " " << next;
            if (i + 1 < new_tokens) logits = engine.forward({next});
        }
        std::cout << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
