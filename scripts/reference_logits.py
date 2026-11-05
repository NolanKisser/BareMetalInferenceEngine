"""Writes reference logits from Hugging Face transformers for tests/logits_test.cu.

Usage: python reference_logits.py <hf_model_dir> <output_file> [prompt] [new_tokens]

Generates greedily on the CPU in 32-bit floats. It then runs the whole sequence
once without a KV cache, so the reference does not depend on any caching.

Output format, little-endian:
  int32 vocab_size, int32 token_count, int32 tokens[token_count],
  float32 logits[token_count][vocab_size]
"""
import struct
import sys

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM


@torch.inference_mode()
def main():
    if not 3 <= len(sys.argv) <= 5:
        sys.exit(__doc__)
    model_dir, output_path = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Once upon a time"
    new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
    ).float().eval()

    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(new_tokens):
        next_token = model(tokens, use_cache=False).logits[0, -1].argmax()
        tokens = torch.cat([tokens, next_token.reshape(1, 1)], dim=1)

    # The last position's logits predict a token beyond the sequence, which is also checked.
    logits = model(tokens, use_cache=False).logits[0]
    vocab_size = logits.shape[1]
    ids = tokens[0].tolist()
    with open(output_path, "wb") as file:
        file.write(struct.pack("<ii", vocab_size, len(ids)))
        file.write(struct.pack(f"<{len(ids)}i", *ids))
        file.write(logits.numpy().astype("<f4").tobytes())

    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    print(f"Token IDs: {ids}")
    print(f"Text: {tokenizer.decode(ids, skip_special_tokens=True)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
