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
#include "tokenizer.h"

// Generates tokens greedily from a text prompt, printing the text as it goes, or
// from token IDs with --ids, printing token IDs.
namespace {

const char* kUsage =
    "Usage: inference_engine <model_dir> <new_tokens> <prompt text ...>\n"
    "       inference_engine <model_dir> <new_tokens> --ids <token_id> [token_id ...]";

int argmax(const std::vector<float>& logits) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << kUsage << std::endl;
        return 1;
    }
    const std::string model_dir = argv[1];
    const bool ids_mode = std::string(argv[3]) == "--ids";
    if (ids_mode && argc < 5) {
        std::cerr << kUsage << std::endl;
        return 1;
    }

    try {
        const int new_tokens = std::stoi(argv[2]);

        Tokenizer tokenizer;
        std::vector<int> prompt;
        if (ids_mode) {
            for (int i = 4; i < argc; ++i) prompt.push_back(std::stoi(argv[i]));
        } else {
            if (!tokenizer.load(model_dir + "/tokenizer.txt")) {
                std::cerr << "Re-run scripts/export_llama.py to write tokenizer.txt, or pass token IDs with --ids." << std::endl;
                return 1;
            }
            std::string text = argv[3];
            for (int i = 4; i < argc; ++i) text += std::string(" ") + argv[i];
            prompt = tokenizer.encode(text);
            if (prompt.empty()) {
                std::cerr << "Error: the prompt is empty" << std::endl;
                return 1;
            }
        }

        ModelLoader loader;
        if (!loader.load(model_dir)) return 1;
        const ModelConfig& config = loader.get_config();
        if (!ids_mode && tokenizer.vocab_size() != config.vocab_size) {
            std::cerr << "Error: the tokenizer has " << tokenizer.vocab_size() << " tokens but the model has " << config.vocab_size << std::endl;
            return 1;
        }

        auto allocator = std::make_shared<MemoryAllocator>();
        allocator->allocate_and_copy_weights(loader.get_tensors());
        auto kv_cache = std::make_shared<KVCache>(1, config.max_seq_len, config.num_layers, config.num_heads, config.head_dim());
        ExecutionEngine engine(allocator, kv_cache, config);

        std::vector<float> logits = engine.forward(prompt);

        if (ids_mode) {
            std::cout << "Generated token IDs:";
            for (int i = 0; i < new_tokens; ++i) {
                const int next = argmax(logits);
                std::cout << " " << next;
                if (i + 1 < new_tokens) logits = engine.forward({next});
            }
            std::cout << std::endl;
            return 0;
        }

        // Decodes the whole sequence after each token and prints what is new, so
        // spaces come out right. A run of byte tokens decodes as a unit, so text
        // is printed only after an ordinary token ends the run.
        auto ends_run = [&](int id) { return !tokenizer.is_byte(id) && !tokenizer.is_special(id); };
        std::vector<int> sequence = prompt;
        size_t stable = sequence.size();
        while (stable > 0 && !ends_run(sequence[stable - 1])) --stable;
        size_t printed = 0;
        auto print_new_text = [&](size_t tokens) {
            const std::string text = tokenizer.decode(std::vector<int>(sequence.begin(), sequence.begin() + tokens));
            if (text.size() > printed) std::cout << text.substr(printed) << std::flush;
            printed = std::max(printed, text.size());
        };
        print_new_text(stable);

        for (int i = 0; i < new_tokens; ++i) {
            const int next = argmax(logits);
            if (next == tokenizer.eos_id()) break;
            sequence.push_back(next);
            if (ends_run(next)) print_new_text(sequence.size());
            if (i + 1 < new_tokens) logits = engine.forward({next});
        }
        print_new_text(sequence.size());
        std::cout << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
