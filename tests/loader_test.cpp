// CPU-only checks of ModelLoader, using small exports written here.
#include "weights.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool passed, const char* name) {
    if (!passed) ++failures;
    std::printf("%s  %s\n", passed ? "PASS" : "FAIL", name);
}

const char* const kSettings =
    "baremetal-llama 1\n"
    "dim 8\nhidden_dim 16\nvocab_size 10\nnum_layers 1\nnum_heads 2\n"
    "max_seq_len 32\nnorm_eps 1e-05\nrope_theta 10000.0\n";

void write_export(const fs::path& dir, const std::string& index, const std::vector<float>& weights) {
    fs::create_directories(dir);
    std::ofstream(dir / "model_index.txt") << index;
    std::ofstream binary(dir / "model.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
}

bool loads(const fs::path& dir, const std::string& index, const std::vector<float>& weights) {
    write_export(dir, index, weights);
    ModelLoader loader;
    return loader.load(dir.string());
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "baremetal_loader_test";
    fs::remove_all(dir);
    const std::vector<float> weights = {1, 2, 3, 4, 5, 6};
    const std::string settings = kSettings;

    write_export(dir, settings + "tensors 2\nfirst 0 8 1 2\nsecond 8 16 2 2\n", weights);
    ModelLoader loader;
    const bool loaded = loader.load(dir.string());
    check(loaded, "loads a valid export");
    if (loaded) {
        const ModelConfig& config = loader.get_config();
        check(config.dim == 8 && config.hidden_dim == 16 && config.vocab_size == 10 && config.num_layers == 1 &&
                  config.num_heads == 2 && config.max_seq_len == 32 && config.head_dim() == 4,
              "reads the model settings");
        const auto& tensors = loader.get_tensors();
        const float* second = tensors.size() == 2 ? static_cast<const float*>(tensors[1]->data) : nullptr;
        check(second && tensors[1]->name == "second" && tensors[1]->shape == std::vector<int>({2, 2}) && second[0] == 3 && second[3] == 6,
              "reads each tensor's name, shape and values");
    }

    ModelLoader missing;
    check(!missing.load((dir / "does_not_exist").string()), "reports a missing export");
    check(!loads(dir, "something-else 1\n", weights), "rejects an unknown format");
    check(!loads(dir, settings + "tensors 1\nfirst 0 12 1 2\n", weights), "rejects a size that does not match the shape");
    check(!loads(dir, settings + "tensors 1\nfirst 16 16 2 2\n", weights), "rejects a tensor past the end of the file");
    check(!loads(dir, settings + "tensors 2\nfirst 0 4 1 1\n", weights), "rejects a truncated tensor list");
    check(!loads(dir, settings + "tensors 2\nfirst 0 4 1 1\nfirst 4 4 1 1\n", weights), "rejects a repeated tensor name");

    std::string odd_head = settings;
    odd_head.replace(odd_head.find("dim 8"), 5, "dim 6");
    check(!loads(dir, odd_head + "tensors 1\nfirst 0 4 1 1\n", weights), "rejects an odd head size");

    fs::remove_all(dir);
    std::printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
