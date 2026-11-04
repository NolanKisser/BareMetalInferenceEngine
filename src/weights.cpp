#include "weights.h"

#include <fstream>
#include <iostream>
#include <set>
#include <utility>

namespace {

bool fail(const std::string& message) {
    std::cerr << "Cannot load model: " << message << std::endl;
    return false;
}

bool valid(const ModelConfig& c) {
    return c.dim > 0 && c.hidden_dim > 0 && c.vocab_size > 0 && c.num_layers > 0 && c.num_heads > 0 &&
           c.max_seq_len > 0 && c.norm_eps > 0.0f && c.rope_theta > 0.0f && c.dim % c.num_heads == 0 &&
           c.head_dim() % 2 == 0;
}

}  // namespace

bool ModelLoader::load(const std::string& model_dir) {
    tensors.clear();
    config = ModelConfig{};

    const std::string index_path = model_dir + "/model_index.txt";
    std::ifstream index(index_path);
    if (!index) return fail("could not open " + index_path);

    std::string format;
    int version = 0;
    if (!(index >> format >> version) || format != "baremetal-llama" || version != 1) {
        return fail(index_path + " is not a version 1 export");
    }

    ModelConfig parsed;
    long long tensor_count = -1;
    std::string key;
    while (tensor_count < 0 && index >> key) {
        bool ok = true;
        if (key == "dim") ok = static_cast<bool>(index >> parsed.dim);
        else if (key == "hidden_dim") ok = static_cast<bool>(index >> parsed.hidden_dim);
        else if (key == "vocab_size") ok = static_cast<bool>(index >> parsed.vocab_size);
        else if (key == "num_layers") ok = static_cast<bool>(index >> parsed.num_layers);
        else if (key == "num_heads") ok = static_cast<bool>(index >> parsed.num_heads);
        else if (key == "max_seq_len") ok = static_cast<bool>(index >> parsed.max_seq_len);
        else if (key == "norm_eps") ok = static_cast<bool>(index >> parsed.norm_eps);
        else if (key == "rope_theta") ok = static_cast<bool>(index >> parsed.rope_theta);
        else if (key == "tensors") ok = static_cast<bool>(index >> tensor_count);
        else return fail("unknown setting '" + key + "' in " + index_path);
        if (!ok) return fail("invalid value for '" + key + "' in " + index_path);
    }
    if (tensor_count < 0) return fail(index_path + " does not list its tensors");
    if (!valid(parsed)) return fail(index_path + " has invalid or missing model settings");

    const std::string weights_path = model_dir + "/model.bin";
    std::ifstream weights(weights_path, std::ios::binary | std::ios::ate);
    if (!weights) return fail("could not open " + weights_path);
    const unsigned long long file_size = static_cast<unsigned long long>(weights.tellg());

    std::vector<std::shared_ptr<Tensor>> loaded;
    std::set<std::string> names;
    for (long long i = 0; i < tensor_count; ++i) {
        std::string name;
        long long offset = -1, bytes = -1;
        int rows = 0, cols = 0;
        if (!(index >> name >> offset >> bytes >> rows >> cols)) return fail(index_path + " ends before listing every tensor");
        if (!names.insert(name).second) return fail("tensor " + name + " is listed twice");
        if (rows <= 0 || cols <= 0 || offset < 0 || bytes != static_cast<long long>(rows) * cols * static_cast<long long>(sizeof(float))) {
            return fail("tensor " + name + " has an inconsistent shape or size");
        }
        if (static_cast<unsigned long long>(offset) + static_cast<unsigned long long>(bytes) > file_size) {
            return fail("tensor " + name + " extends past the end of " + weights_path);
        }

        auto tensor = std::make_shared<Tensor>();
        tensor->name = name;
        tensor->shape = {rows, cols};
        tensor->size_bytes = static_cast<size_t>(bytes);
        tensor->data = malloc(tensor->size_bytes);
        if (!tensor->data) return fail("out of host memory reading " + name);
        weights.seekg(offset);
        weights.read(static_cast<char*>(tensor->data), bytes);
        if (!weights) return fail("could not read " + name + " from " + weights_path);
        loaded.push_back(tensor);
    }

    config = parsed;
    tensors = std::move(loaded);
    std::cout << "Loaded " << tensors.size() << " tensors from " << weights_path << std::endl;
    return true;
}
