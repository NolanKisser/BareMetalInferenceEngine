#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// Basic tensor structure. Data is always 32-bit float, row-major.
struct Tensor {
    std::string name;
    std::vector<int> shape;
    void* data = nullptr;
    size_t size_bytes = 0;

    ~Tensor() {
        if (data) {
            free(data);
        }
    }
};

// Hyperparameters of a Llama-architecture model.
struct ModelConfig {
    int dim = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
    int num_layers = 0;
    int num_heads = 0;
    int max_seq_len = 0;
    float norm_eps = 0.0f;
    float rope_theta = 0.0f;

    int head_dim() const { return num_heads > 0 ? dim / num_heads : 0; }
};

// Reads a directory written by scripts/export_llama.py, which contains
// model_index.txt and model.bin.
class ModelLoader {
public:
    // Returns false and prints the reason if the export is missing or invalid.
    bool load(const std::string& model_dir);

    const ModelConfig& get_config() const { return config; }
    const std::vector<std::shared_ptr<Tensor>>& get_tensors() const { return tensors; }

private:
    ModelConfig config;
    std::vector<std::shared_ptr<Tensor>> tensors;
};

#endif // WEIGHTS_H
