// GPU correctness checks: every kernel is compared with a CPU reference.
// Requires a CUDA device. Exits non-zero if any check fails.
#include "cuda_utils.h"
#include "kernels.cuh"

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

// Rotation preserves each query's length, and at position 1 it must change the
// values. The tensor layout is not defined yet, so this checks only for any change.
void test_rope_changes_query() {
    const int head_dim = 128, heads = 8;
    const size_t count = static_cast<size_t>(head_dim) * heads;
    const auto q = random_floats(count);
    const auto k = random_floats(count);
    DeviceBuffer<float> dq(count), dk(count);
    upload(dq, q);
    upload(dk, k);
    launch_rope(dq.get(), dk.get(), 1, head_dim, heads, heads);
    const auto got = download(dq, count);
    const bool changed = got != q;
    if (!changed) ++failures;
    std::printf("%s  %-54s %s\n", changed ? "PASS" : "FAIL", "rope rotates the query at position 1",
                changed ? "values changed" : "values unchanged");
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
    test_softmax(4096);
    test_swiglu(4096);
    test_rope_changes_query();

    CUDA_CHECK(cudaGetLastError());
    std::printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
