// Times the custom GEMM kernel doing real work. pytorch_gemm.py runs the same
// workloads, and both write the same JSON layout so the results can be compared.
//
// Usage: gemm_bench [output.json]
#include "cuda_utils.h"
#include "kernels.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Windows resets the display driver if one kernel runs for about 2 seconds, so
// a size is skipped when the previous size predicts a run longer than this.
const double kMaxSingleLaunchSeconds = 1.0;

std::vector<float> normal_floats(size_t count, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(count);
    for (float& value : values) value = dist(rng);
    return values;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double square_gemm_seconds(int n, std::mt19937& rng, int reps, std::vector<double>& samples) {
    const size_t count = static_cast<size_t>(n) * n;
    DeviceBuffer<float> a(count), b(count), c(count);
    const auto host_a = normal_floats(count, 1.0f, rng);
    const auto host_b = normal_floats(count, 1.0f, rng);
    CUDA_CHECK(cudaMemcpy(a.get(), host_a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.get(), host_b.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    launch_gemm(c.get(), a.get(), b.get(), n, n, n);  // warm-up, not timed
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int rep = 0; rep < reps; ++rep) {
        const auto start = Clock::now();
        launch_gemm(c.get(), a.get(), b.get(), n, n, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return median(samples);
}

// A vector passes through `layers` separate weight matrices. This is the matrix
// work in a transformer's feed-forward steps, and nothing else from inference.
double chain_seconds(int dim, int layers, int passes, int reps, std::mt19937& rng, std::vector<double>& samples) {
    const size_t weight_count = static_cast<size_t>(dim) * dim;
    const float weight_stddev = 1.0f / std::sqrt(static_cast<float>(dim));  // keeps values from overflowing
    std::vector<DeviceBuffer<float>> weights;
    weights.reserve(layers);
    const auto host_weight = normal_floats(weight_count, weight_stddev, rng);
    for (int layer = 0; layer < layers; ++layer) {
        weights.emplace_back(weight_count);
        CUDA_CHECK(cudaMemcpy(weights.back().get(), host_weight.data(), weight_count * sizeof(float), cudaMemcpyHostToDevice));
    }

    const auto input = normal_floats(dim, 1.0f, rng);
    DeviceBuffer<float> first(dim), second(dim);
    for (int rep = -1; rep < reps; ++rep) {  // rep -1 is an untimed warm-up
        CUDA_CHECK(cudaMemcpy(first.get(), input.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
        const auto start = Clock::now();
        float* in = first.get();
        float* out = second.get();
        for (int pass = 0; pass < passes; ++pass) {
            for (int layer = 0; layer < layers; ++layer) {
                launch_gemm(out, in, weights[layer].get(), 1, dim, dim);
                std::swap(in, out);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (rep >= 0) samples.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    return median(samples);
}

void write_samples(FILE* file, const std::vector<double>& samples) {
    std::fprintf(file, "[");
    for (size_t i = 0; i < samples.size(); ++i) std::fprintf(file, "%s%.9f", i ? ", " : "", samples[i]);
    std::fprintf(file, "]");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output_path = argc > 1 ? argv[1] : "baremetal.json";
    std::mt19937 rng(42);

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("Device: %s\n", device.name);

    FILE* file = std::fopen(output_path.c_str(), "w");
    if (!file) {
        std::fprintf(stderr, "Cannot write %s\n", output_path.c_str());
        return 1;
    }
    std::fprintf(file, "{\n  \"engine\": \"baremetal\",\n  \"device\": \"%s\",\n  \"square\": [", device.name);

    double previous_seconds = 0.0;
    bool first_row = true;
    for (int n : {256, 512, 1024, 2048, 4096}) {
        // Cost grows with the cube of n, so doubling n costs about 8 times as much.
        if (previous_seconds * 8.0 > kMaxSingleLaunchSeconds) {
            std::printf("Skipping n=%d: predicted %.2f s per launch risks a display driver reset\n", n, previous_seconds * 8.0);
            continue;
        }
        std::vector<double> samples;
        previous_seconds = square_gemm_seconds(n, rng, 7, samples);
        const double gflops = 2.0 * n * static_cast<double>(n) * n / previous_seconds / 1e9;
        std::printf("n=%-5d median %.6f s  %.1f GFLOP/s\n", n, previous_seconds, gflops);
        std::fprintf(file, "%s\n    {\"n\": %d, \"median_s\": %.9f, \"gflops\": %.3f, \"samples\": ", first_row ? "" : ",", n, previous_seconds, gflops);
        write_samples(file, samples);
        std::fprintf(file, "}");
        first_row = false;
    }

    const int dim = 4096, layers = 32, passes = 100;
    std::vector<double> samples;
    const double seconds = chain_seconds(dim, layers, passes, 5, rng, samples);
    std::printf("chain dim=%d layers=%d: median %.4f s for %d passes = %.1f passes/s\n", dim, layers, seconds, passes, passes / seconds);
    std::fprintf(file, "\n  ],\n  \"chain\": {\"dim\": %d, \"layers\": %d, \"passes\": %d, \"median_s\": %.9f, \"passes_per_s\": %.3f, \"samples\": ",
                 dim, layers, passes, seconds, passes / seconds);
    write_samples(file, samples);
    std::fprintf(file, "}\n}\n");
    std::fclose(file);
    std::printf("Wrote %s\n", output_path.c_str());
    return 0;
}
