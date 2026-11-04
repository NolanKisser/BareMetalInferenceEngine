// Compares the engine's logits with a Hugging Face reference for a whole sequence.
//
// Usage: logits_test <model_dir>
// Expects <model_dir>/reference_logits.bin from scripts/reference_logits.py.
#include "execution_engine.h"
#include "kv_cache.h"
#include "memory_allocator.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// A limit on the largest absolute difference in any logit. It allows for float
// rounding between two implementations, and is far below the gaps that decide
// which token is chosen.
const float kMaxLogitError = 0.01f;

struct Reference {
    int vocab_size = 0;
    std::vector<int> tokens;
    std::vector<float> logits;
    const float* at(size_t position) const { return logits.data() + position * vocab_size; }
};

bool read_reference(const std::string& path, Reference& reference) {
    std::ifstream file(path, std::ios::binary);
    int32_t header[2];
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) || header[0] <= 0 || header[1] <= 0) return false;
    reference.vocab_size = header[0];
    reference.tokens.resize(header[1]);
    reference.logits.resize(static_cast<size_t>(header[0]) * header[1]);
    file.read(reinterpret_cast<char*>(reference.tokens.data()), reference.tokens.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(reference.logits.data()), reference.logits.size() * sizeof(float));
    return static_cast<bool>(file);
}

float worst_error(const std::vector<float>& got, const float* want) {
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float error = std::fabs(got[i] - want[i]);
        if (!(error <= worst)) worst = error;  // a NaN must count as a failure
    }
    return worst;
}

int argmax(const float* values, int count) {
    return static_cast<int>(std::max_element(values, values + count) - values);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: logits_test <model_dir>\n");
        return 2;
    }
    const std::string model_dir = argv[1];

    try {
        Reference reference;
        if (!read_reference(model_dir + "/reference_logits.bin", reference)) {
            std::fprintf(stderr, "Could not read reference_logits.bin in %s\n", model_dir.c_str());
            return 2;
        }
        ModelLoader loader;
        if (!loader.load(model_dir)) return 2;
        const ModelConfig& config = loader.get_config();
        if (config.vocab_size != reference.vocab_size) {
            std::fprintf(stderr, "The reference and model vocabulary sizes differ\n");
            return 2;
        }

        auto allocator = std::make_shared<MemoryAllocator>();
        allocator->allocate_and_copy_weights(loader.get_tensors());
        auto kv_cache = std::make_shared<KVCache>(1, config.max_seq_len, config.num_layers, config.num_heads, config.head_dim());
        ExecutionEngine engine(allocator, kv_cache, config);

        int failures = 0;
        const size_t count = reference.tokens.size();

        // One token at a time, so every position's logits can be checked.
        float worst = 0.0f;
        int wrong_tokens = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto logits = engine.forward({reference.tokens[i]});
            const float error = worst_error(logits, reference.at(i));
            worst = (error > worst || error != error) ? error : worst;
            const int got = argmax(logits.data(), config.vocab_size);
            if (got != argmax(reference.at(i), config.vocab_size)) ++wrong_tokens;
            if (i < 3 || i + 1 == count) std::printf("  position %3zu: token %5d, worst logit error %.6f\n", i, reference.tokens[i], error);
        }
        bool passed = worst <= kMaxLogitError;
        if (!passed) ++failures;
        std::printf("%s  logits at all %zu positions: worst error %.6f, limit %.3f\n", passed ? "PASS" : "FAIL", count, worst, kMaxLogitError);
        passed = wrong_tokens == 0;
        if (!passed) ++failures;
        std::printf("%s  most likely next token: %d of %zu positions differ\n", passed ? "PASS" : "FAIL", wrong_tokens, count);

        // The whole sequence in one call, after a reset.
        engine.reset();
        const float batch_error = worst_error(engine.forward(reference.tokens), reference.at(count - 1));
        passed = batch_error <= kMaxLogitError;
        if (!passed) ++failures;
        std::printf("%s  reset, then the whole sequence in one call: worst error %.6f\n", passed ? "PASS" : "FAIL", batch_error);

        std::printf("\n%d check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
