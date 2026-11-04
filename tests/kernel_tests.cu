// GPU correctness checks: every kernel is compared with a CPU reference.
// Requires a CUDA device. Exits non-zero if any check fails.
#include "cuda_utils.h"
#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(42);
int failures = 0;

std::vector<float> random_floats(size_t count, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> values(count);
    for (float& value : values) value = dist(rng);
    return values;
}

void upload(const DeviceBuffer<float>& device, const std::vector<float>& host) {
    CUDA_CHECK(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> download(const DeviceBuffer<float>& device, size_t count) {
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device.get(), count * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

void report(const std::string& name, const std::vector<float>& got, const std::vector<float>& want) {
    const float tolerance = 1e-3f;
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double error = std::fabs(static_cast<double>(got[i]) - want[i]) / (1.0 + std::fabs(want[i]));
        // A NaN must count as a failure, so test for "not within tolerance".
        if (!(error <= worst)) worst = error;
    }
    const bool passed = worst <= tolerance;
    if (!passed) ++failures;
    std::printf("%s  %-54s worst relative error %.3g\n", passed ? "PASS" : "FAIL", name.c_str(), worst);
}

std::vector<float> reference_gemm(const std::vector<float>& a, const std::vector<float>& b, int m, int n, int k) {
    std::vector<float> c(static_cast<size_t>(m) * n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += static_cast<double>(a[static_cast<size_t>(row) * k + i]) * b[static_cast<size_t>(i) * n + col];
            c[static_cast<size_t>(row) * n + col] = static_cast<float>(sum);
        }
    }
    return c;
}

void test_gemm(int m, int n, int k) {
    const auto a = random_floats(static_cast<size_t>(m) * k);
    const auto b = random_floats(static_cast<size_t>(k) * n);
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(static_cast<size_t>(m) * n);
    upload(da, a);
    upload(db, b);
    launch_gemm(dc.get(), da.get(), db.get(), m, n, k);
    report("gemm " + std::to_string(m) + "x" + std::to_string(k) + " * " + std::to_string(k) + "x" + std::to_string(n),
           download(dc, static_cast<size_t>(m) * n), reference_gemm(a, b, m, n, k));
}

void test_rms_norm(int size) {
    const auto in = random_floats(size);
    const auto weight = random_floats(size, 0.5f, 1.5f);
    double sum_of_squares = 0.0;
    for (float value : in) sum_of_squares += static_cast<double>(value) * value;
    const double scale = 1.0 / std::sqrt(sum_of_squares / size + 1e-5);
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = static_cast<float>(in[i] * scale * weight[i]);

    DeviceBuffer<float> din(size), dweight(size), dout(size);
    upload(din, in);
    upload(dweight, weight);
    launch_rms_norm(dout.get(), din.get(), dweight.get(), size);
    report("rms_norm size " + std::to_string(size), download(dout, size), want);
}

// ExecutionEngine::forward normalizes the hidden state in place.
void test_rms_norm_in_place(int size) {
    const auto in = random_floats(size);
    const auto weight = random_floats(size, 0.5f, 1.5f);
    double sum_of_squares = 0.0;
    for (float value : in) sum_of_squares += static_cast<double>(value) * value;
    const double scale = 1.0 / std::sqrt(sum_of_squares / size + 1e-5);
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = static_cast<float>(in[i] * scale * weight[i]);

    DeviceBuffer<float> dx(size), dweight(size);
    upload(dx, in);
    upload(dweight, weight);
    launch_rms_norm(dx.get(), dx.get(), dweight.get(), size);
    report("rms_norm in place, as forward() calls it, size " + std::to_string(size), download(dx, size), want);
}

void test_softmax(int size) {
    const auto in = random_floats(size, -4.0f, 4.0f);
    double total = 0.0;
    for (float value : in) total += std::exp(static_cast<double>(value));
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = static_cast<float>(std::exp(static_cast<double>(in[i])) / total);

    DeviceBuffer<float> dx(size);
    upload(dx, in);
    launch_softmax(dx.get(), size);
    report("softmax size " + std::to_string(size), download(dx, size), want);
}

void test_swiglu(int size) {
    const auto gate = random_floats(size, -3.0f, 3.0f);
    const auto up = random_floats(size, -3.0f, 3.0f);
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = static_cast<float>(gate[i] / (1.0 + std::exp(-static_cast<double>(gate[i]))) * up[i]);

    DeviceBuffer<float> dgate(size), dup(size), dout(size);
    upload(dgate, gate);
    upload(dup, up);
    CUDA_CHECK(cudaMemset(dout.get(), 0, size * sizeof(float)));
    launch_swiglu(dout.get(), dgate.get(), dup.get(), size);
    report("swiglu size " + std::to_string(size), download(dout, size), want);
}

void test_softmax_large_values(int size) {
    auto in = random_floats(size, 995.0f, 1005.0f);
    const float largest = *std::max_element(in.begin(), in.end());
    double total = 0.0;
    for (float value : in) total += std::exp(static_cast<double>(value) - largest);
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = static_cast<float>(std::exp(static_cast<double>(in[i]) - largest) / total);

    DeviceBuffer<float> dx(size);
    upload(dx, in);
    launch_softmax(dx.get(), size);
    report("softmax does not overflow on values near 1000", download(dx, size), want);
}

void test_softmax_batch(int size, int vectors, int stride) {
    const size_t count = static_cast<size_t>(vectors) * stride;
    const auto in = random_floats(count, -4.0f, 4.0f);
    std::vector<float> want = in;
    for (int v = 0; v < vectors; ++v) {
        double total = 0.0;
        for (int i = 0; i < size; ++i) total += std::exp(static_cast<double>(in[static_cast<size_t>(v) * stride + i]));
        for (int i = 0; i < size; ++i) {
            const size_t at = static_cast<size_t>(v) * stride + i;
            want[at] = static_cast<float>(std::exp(static_cast<double>(in[at])) / total);
        }
    }
    DeviceBuffer<float> dx(count);
    upload(dx, in);
    launch_softmax_batch(dx.get(), size, vectors, stride);
    report("softmax batch, leaves the gaps between vectors alone", download(dx, count), want);
}

void test_swiglu_extreme_values() {
    const std::vector<float> gate = {-1000.0f, -100.0f, 0.0f, 100.0f, 1000.0f};
    const std::vector<float> up = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> want = {0.0f, 0.0f, 0.0f, 200.0f, 2000.0f};
    DeviceBuffer<float> dgate(gate.size()), dup(up.size()), dout(gate.size());
    upload(dgate, gate);
    upload(dup, up);
    launch_swiglu(dout.get(), dgate.get(), dup.get(), static_cast<int>(gate.size()));
    report("swiglu stays finite for very large inputs", download(dout, gate.size()), want);
}

std::vector<float> reference_rope(const std::vector<float>& x, int seq_pos, int head_dim, int heads, double theta) {
    std::vector<float> out = x;
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < half; ++i) {
            const double angle = seq_pos / std::pow(theta, 2.0 * i / head_dim);
            const double first = x[h * head_dim + i], second = x[h * head_dim + i + half];
            out[h * head_dim + i] = static_cast<float>(first * std::cos(angle) - second * std::sin(angle));
            out[h * head_dim + i + half] = static_cast<float>(second * std::cos(angle) + first * std::sin(angle));
        }
    }
    return out;
}

void test_rope(int seq_pos, int head_dim, int heads, int kv_heads) {
    const size_t q_count = static_cast<size_t>(head_dim) * heads, k_count = static_cast<size_t>(head_dim) * kv_heads;
    const auto q = random_floats(q_count);
    const auto k = random_floats(k_count);
    DeviceBuffer<float> dq(q_count), dk(k_count);
    upload(dq, q);
    upload(dk, k);
    launch_rope(dq.get(), dk.get(), seq_pos, head_dim, heads, kv_heads);
    const std::string where = " at position " + std::to_string(seq_pos) + ", " + std::to_string(heads) + " heads";
    report("rope q" + where, download(dq, q_count), reference_rope(q, seq_pos, head_dim, heads, 10000.0));
    report("rope k, " + std::to_string(kv_heads) + " heads, position " + std::to_string(seq_pos), download(dk, k_count), reference_rope(k, seq_pos, head_dim, kv_heads, 10000.0));
}

void test_add(int size) {
    const auto a = random_floats(size);
    const auto b = random_floats(size);
    std::vector<float> want(size);
    for (int i = 0; i < size; ++i) want[i] = a[i] + b[i];
    DeviceBuffer<float> da(size), db(size);
    upload(da, a);
    upload(db, b);
    launch_add(da.get(), da.get(), db.get(), size);
    report("add in place, size " + std::to_string(size), download(da, size), want);
}

void test_kv_store() {
    const int heads = 3, head_dim = 8, max_seq = 5, pos = 2;
    const size_t cache_count = static_cast<size_t>(heads) * max_seq * head_dim;
    auto want = random_floats(cache_count);
    const auto src = random_floats(static_cast<size_t>(heads) * head_dim);
    DeviceBuffer<float> dcache(cache_count), dsrc(src.size());
    upload(dcache, want);
    upload(dsrc, src);
    launch_kv_store(dcache.get(), dsrc.get(), pos, heads, head_dim, max_seq);
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < head_dim; ++d) want[(static_cast<size_t>(h) * max_seq + pos) * head_dim + d] = src[h * head_dim + d];
    }
    report("kv store writes one position and leaves the rest alone", download(dcache, cache_count), want);
}

void test_attention(int heads, int head_dim, int max_seq, int seq_len) {
    const size_t cache_count = static_cast<size_t>(heads) * max_seq * head_dim, query_count = static_cast<size_t>(heads) * head_dim;
    const auto q = random_floats(query_count);
    const auto k = random_floats(cache_count);
    const auto v = random_floats(cache_count);

    std::vector<float> want(query_count);
    for (int h = 0; h < heads; ++h) {
        std::vector<double> weights(seq_len);
        double total = 0.0;
        for (int j = 0; j < seq_len; ++j) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(q[h * head_dim + d]) * k[(static_cast<size_t>(h) * max_seq + j) * head_dim + d];
            weights[j] = std::exp(dot / std::sqrt(static_cast<double>(head_dim)));
            total += weights[j];
        }
        for (int d = 0; d < head_dim; ++d) {
            double sum = 0.0;
            for (int j = 0; j < seq_len; ++j) sum += weights[j] / total * v[(static_cast<size_t>(h) * max_seq + j) * head_dim + d];
            want[h * head_dim + d] = static_cast<float>(sum);
        }
    }

    DeviceBuffer<float> dq(query_count), dk(cache_count), dv(cache_count), dout(query_count), dscores(static_cast<size_t>(heads) * max_seq);
    upload(dq, q);
    upload(dk, k);
    upload(dv, v);
    launch_attention(dout.get(), dq.get(), dk.get(), dv.get(), dscores.get(), heads, head_dim, max_seq, seq_len);
    report("attention, " + std::to_string(heads) + " heads, " + std::to_string(seq_len) + " of " + std::to_string(max_seq) + " positions", download(dout, query_count), want);
}

}  // namespace

int main() {
    test_gemm(16, 16, 16);
    test_gemm(128, 128, 128);
    test_gemm(3, 5, 7);
    test_gemm(17, 33, 65);
    test_gemm(1, 4096, 4096);
    test_rms_norm(64);
    test_rms_norm(256);
    test_rms_norm(300);
    test_rms_norm(4096);
    test_rms_norm_in_place(4096);
    test_softmax(1);
    test_softmax(100);
    test_softmax(4096);
    test_softmax_large_values(256);
    test_softmax_batch(50, 6, 64);
    test_swiglu(4096);
    test_swiglu_extreme_values();
    test_rope(0, 48, 6, 6);
    test_rope(1, 48, 6, 3);
    test_rope(37, 128, 8, 8);
    test_add(1000);
    test_kv_store();
    test_attention(1, 16, 1, 1);
    test_attention(6, 48, 256, 1);
    test_attention(6, 48, 256, 200);

    CUDA_CHECK(cudaGetLastError());
    std::printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
