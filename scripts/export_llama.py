"""Converts a Hugging Face Llama model to the format ModelLoader reads.

Usage: python export_llama.py <hf_model_dir> <output_dir>

Reads config.json and model.safetensors only, and never loads pickle files.
Writes model_index.txt and model.bin, with every tensor as 32-bit float. Linear
projections are transposed to [in, out], because the engine multiplies a row
vector by the weight matrix.
"""
import json
import pathlib
import re
import sys

import numpy as np
from safetensors import safe_open

LAYER_TENSORS = {
    "input_layernorm.weight": False,
    "post_attention_layernorm.weight": False,
    "self_attn.q_proj.weight": True,
    "self_attn.k_proj.weight": True,
    "self_attn.v_proj.weight": True,
    "self_attn.o_proj.weight": True,
    "mlp.gate_proj.weight": True,
    "mlp.up_proj.weight": True,
    "mlp.down_proj.weight": True,
}


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    unsupported = []
    if config.get("model_type") != "llama":
        unsupported.append(f"model_type {config.get('model_type')}")
    if config.get("hidden_act") != "silu":
        unsupported.append(f"hidden_act {config.get('hidden_act')}")
    if config.get("rope_scaling"):
        unsupported.append("rope_scaling")
    if config.get("attention_bias") or config.get("mlp_bias"):
        unsupported.append("bias terms")
    if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
        unsupported.append("separate key and value head counts")
    if config.get("pretraining_tp", 1) != 1:
        unsupported.append("pretraining_tp")
    if unsupported:
        sys.exit("Unsupported model settings: " + ", ".join(unsupported))
    return config


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir = pathlib.Path(sys.argv[1])
    output_dir = pathlib.Path(sys.argv[2])
    config = read_config(model_dir / "config.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(model_dir / "model.safetensors", framework="np") as weights:
        names = set(weights.keys())
        tensors = [("model.embed_tokens.weight", "model.embed_tokens.weight", False)]
        for layer in range(config["num_hidden_layers"]):
            for suffix, transpose in LAYER_TENSORS.items():
                name = f"model.layers.{layer}.{suffix}"
                tensors.append((name, name, transpose))
        tensors.append(("model.norm.weight", "model.norm.weight", False))

        if "lm_head.weight" in names:
            tensors.append(("lm_head.weight", "lm_head.weight", True))
        elif config.get("tie_word_embeddings"):
            tensors.append(("lm_head.weight", "model.embed_tokens.weight", True))
        else:
            sys.exit("The model has no lm_head.weight and does not tie its embeddings")

        known = {source for _, source, _ in tensors}
        # Rotary embedding frequencies are computed, so older exports may store them.
        ignored = {name for name in names if re.search(r"rotary_emb\.inv_freq$", name)}
        unknown = names - known - ignored
        if unknown:
            sys.exit("Unsupported tensors: " + ", ".join(sorted(unknown)))
        missing = known - names
        if missing:
            sys.exit("Missing tensors: " + ", ".join(sorted(missing)))

        lines = [
            "baremetal-llama 1",
            f"dim {config['hidden_size']}",
            f"hidden_dim {config['intermediate_size']}",
            f"vocab_size {config['vocab_size']}",
            f"num_layers {config['num_hidden_layers']}",
            f"num_heads {config['num_attention_heads']}",
            f"max_seq_len {config['max_position_embeddings']}",
            f"norm_eps {config['rms_norm_eps']}",
            f"rope_theta {config.get('rope_theta', 10000.0)}",
            f"tensors {len(tensors)}",
        ]
        offset = 0
        with open(output_dir / "model.bin", "wb") as binary:
            for name, source, transpose in tensors:
                data = weights.get_tensor(source).astype("<f4")
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim != 2:
                    sys.exit(f"{source} has {data.ndim} dimensions")
                if transpose:
                    data = data.T
                binary.write(np.ascontiguousarray(data).tobytes())
                lines.append(f"{name} {offset} {data.size * 4} {data.shape[0]} {data.shape[1]}")
                offset += data.size * 4

    (output_dir / "model_index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(tensors)} tensors, {offset / 1e6:.1f} MB, to {output_dir}")


if __name__ == "__main__":
    main()
