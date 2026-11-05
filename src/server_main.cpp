#include "server.h"
#include "weights.h"
#include "memory_allocator.h"
#include "kv_cache.h"
#include <exception>
#include <memory>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: server <model_dir>" << std::endl;
        return 1;
    }
    int port = 8080;
    std::cout << "--- BareMetal Inference API Server ---" << std::endl;

    try {
        ModelLoader loader;
        if (!loader.load(argv[1])) return 1;
        const ModelConfig& config = loader.get_config();

        auto allocator = std::make_shared<MemoryAllocator>();
        allocator->allocate_and_copy_weights(loader.get_tensors());
        auto kv_cache = std::make_shared<KVCache>(1, config.max_seq_len, config.num_layers, config.num_heads, config.head_dim());
        auto engine = std::make_shared<ExecutionEngine>(allocator, kv_cache, config);

        // Start Phase 4 Server
        InferenceServer server(engine, port);
        server.start();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
