"""Runs the gemm_bench.cu workloads with PyTorch and writes the same JSON layout.

Usage: python pytorch_gemm.py [output.json]
"""
import json
import statistics
import sys
import time

import torch


def timed(work):
    torch.cuda.synchronize()
    start = time.perf_counter()
    work()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def square_gemm(n, reps=7):
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    c = torch.empty(n, n, device="cuda")
    work = lambda: torch.matmul(a, b, out=c)
    work()  # warm-up, not timed
    torch.cuda.synchronize()
    return [timed(work) for _ in range(reps)]


def chain(dim, layers, passes, reps=5):
    host_weight = torch.randn(dim, dim) / dim**0.5  # keeps values from overflowing
    weights = [host_weight.to("cuda") for _ in range(layers)]
    first = torch.randn(1, dim, device="cuda")
    second = torch.empty(1, dim, device="cuda")

    def work():
        current, target = first.clone(), second
        for _ in range(passes):
            for weight in weights:
                torch.matmul(current, weight, out=target)
                current, target = target, current
            torch.cuda.synchronize()

    work()  # warm-up, not timed
    return [timed(work) for _ in range(reps)]


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        sys.exit("A CUDA device is required: a CPU result would not be comparable.")
    output_path = sys.argv[1] if len(sys.argv) > 1 else "pytorch.json"
    torch.manual_seed(42)

    result = {
        "engine": "pytorch",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "square": [],
    }
    print(f"Device: {result['device']}  torch {torch.__version__}")

    for n in (256, 512, 1024, 2048, 4096):
        samples = square_gemm(n)
        seconds = statistics.median(samples)
        gflops = 2 * n**3 / seconds / 1e9
        print(f"n={n:<5d} median {seconds:.6f} s  {gflops:.1f} GFLOP/s")
        result["square"].append({"n": n, "median_s": seconds, "gflops": gflops, "samples": samples})

    dim, layers, passes = 4096, 32, 100
    samples = chain(dim, layers, passes)
    seconds = statistics.median(samples)
    print(f"chain dim={dim} layers={layers}: median {seconds:.4f} s for {passes} passes = {passes / seconds:.1f} passes/s")
    result["chain"] = {
        "dim": dim,
        "layers": layers,
        "passes": passes,
        "median_s": seconds,
        "passes_per_s": passes / seconds,
        "samples": samples,
    }

    with open(output_path, "w") as file:
        json.dump(result, file, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
