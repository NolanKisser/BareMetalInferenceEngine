"""Runs the src/benchmark.cpp workload with Hugging Face transformers on the GPU.

Generates greedily from the same start token, one token at a time with a KV cache,
copies the logits to the host and picks the next token there, as the engine does.
Both programs write the same JSON layout, including the tokens they generated, so
plot_benchmarks.py can confirm that they did the same work.

Usage: python pytorch_tokens.py <hf_model_dir> [output.json]
"""
import json
import statistics
import sys
import time

import torch
import transformers
from transformers import DynamicCache, LlamaForCausalLM

BOS_TOKEN = 1
GENERATED_TOKENS = 100
REPETITIONS = 20


def generate(model, count):
    cache = DynamicCache()
    token = BOS_TOKEN
    tokens = []
    for _ in range(count):
        input_ids = torch.tensor([[token]], device="cuda")
        logits = model(input_ids=input_ids, past_key_values=cache, use_cache=True).logits
        token = logits[0, -1].cpu().argmax().item()
        tokens.append(token)
    return tokens


@torch.inference_mode()
def main():
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    if not torch.cuda.is_available():
        sys.exit("A CUDA device is required: a CPU result would not be comparable.")
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "pytorch_tokens.json"

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    ).float().to("cuda").eval()

    generate(model, 10)  # warm-up, not timed

    samples = []
    tokens = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        tokens = generate(model, GENERATED_TOKENS)
        samples.append(time.perf_counter() - start)

    seconds = statistics.median(samples)
    print(f"Device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  transformers {transformers.__version__}")
    print(f"Attention: {model.config._attn_implementation}")
    print(f"Median of {REPETITIONS} repetitions: {seconds * 1000 / GENERATED_TOKENS:.3f} ms per token, {GENERATED_TOKENS / seconds:.1f} tokens/s")

    with open(output_path, "w") as file:
        json.dump({
            "engine": "pytorch",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "attention": model.config._attn_implementation,
            "device": torch.cuda.get_device_name(0),
            "generated_tokens": GENERATED_TOKENS,
            "median_s": seconds,
            "tokens_per_s": GENERATED_TOKENS / seconds,
            "samples": samples,
            "tokens": tokens,
        }, file, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
