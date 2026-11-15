"""Converts a Hugging Face Llama model to the format ModelLoader reads.

Usage: python export_llama.py <hf_model_dir> <output_dir>

Reads config.json and model.safetensors only, and never loads pickle files.
Writes model_index.txt and model.bin, with every tensor as 32-bit float. Linear
projections are transposed to [in, out], because the engine multiplies a row
vector by the weight matrix. Also writes tokenizer.txt from tokenizer.json when
the tokenizer is the SentencePiece-style BPE that src/tokenizer.cpp implements.
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


class UnsupportedTokenizer(Exception):
    pass


SPACE_MARKER = "▁"
SENTENCEPIECE_NORMALIZER = {
    "type": "Sequence",
    "normalizers": [
        {"type": "Prepend", "prepend": SPACE_MARKER},
        {"type": "Replace", "pattern": {"String": " "}, "content": SPACE_MARKER},
    ],
}
SENTENCEPIECE_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": SPACE_MARKER}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}


def hex_string(text):
    return text.encode("utf-8").hex() or "-"


def write_tokenizer(json_path, config, output_path):
    """Writes the SentencePiece-style BPE tokenizer that src/tokenizer.cpp reads.

    Raises UnsupportedTokenizer for anything else, such as Llama 3's byte-level BPE.
    Strings are written as UTF-8 hex, or "-" when empty, so each fits in one field.
    """
    if not json_path.exists():
        raise UnsupportedTokenizer(f"{json_path} does not exist")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    model = data.get("model", {})

    problems = []
    if model.get("type") != "BPE":
        problems.append(f"model type {model.get('type')}")
    if data.get("normalizer") != SENTENCEPIECE_NORMALIZER:
        problems.append("a normalizer other than SentencePiece's")
    if data.get("pre_tokenizer") is not None:
        problems.append("a pre-tokenizer")
    if data.get("decoder") != SENTENCEPIECE_DECODER:
        problems.append("a decoder other than SentencePiece's")
    if model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        problems.append("BPE dropout or subword affixes")
    if model.get("ignore_merges"):
        problems.append("ignore_merges")
    for token in data.get("added_tokens", []):
        if token["lstrip"] or token["rstrip"] or token["single_word"] or not token["normalized"]:
            problems.append(f"added token {token['content']!r} with matching options")
    if problems:
        raise UnsupportedTokenizer("it uses " + ", ".join(problems))

    vocab = model["vocab"]
    pieces = [None] * len(vocab)
    for piece, index in vocab.items():
        if not 0 <= index < len(pieces) or pieces[index] is not None:
            raise UnsupportedTokenizer("its vocabulary ids are not 0 to n - 1")
        pieces[index] = piece
    if len(pieces) != config["vocab_size"]:
        raise UnsupportedTokenizer(f"it has {len(pieces)} tokens but the model has {config['vocab_size']}")

    def token_id(content):
        for token in data.get("added_tokens", []):
            if token["content"] == content:
                return token["id"]
        if content in vocab:
            return vocab[content]
        raise UnsupportedTokenizer(f"{content!r} is not in the vocabulary")

    bos = config.get("bos_token_id", -1)
    eos = config.get("eos_token_id", -1)
    if isinstance(eos, list):
        eos = eos[0]
    unk = token_id(model["unk_token"]) if model.get("unk_token") else -1

    # Hugging Face adds the start token through the post-processor.
    single = (data.get("post_processor") or {}).get("single")
    if data.get("post_processor") is None:
        add_bos = 0
    elif data["post_processor"].get("type") == "TemplateProcessing" and single == [
        {"SpecialToken": {"id": pieces[bos] if 0 <= bos < len(pieces) else None, "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
    ]:
        add_bos = 1
    else:
        raise UnsupportedTokenizer("its post-processor does something other than add the start token")

    lines = [
        "baremetal-tokenizer 1",
        f"bos {bos}",
        f"eos {eos}",
        f"unk {unk}",
        f"add_bos {add_bos}",
        f"byte_fallback {int(bool(model.get('byte_fallback')))}",
        f"vocab {len(pieces)}",
    ]
    lines += [hex_string(piece) for piece in pieces]

    added = data.get("added_tokens", [])
    lines.append(f"added {len(added)}")
    for token in added:
        if not 0 <= token["id"] < len(pieces):
            raise UnsupportedTokenizer(f"added token {token['content']!r} is outside the vocabulary")
        # Added tokens are matched after normalization, so match their normalized text.
        match = SPACE_MARKER + token["content"].replace(" ", SPACE_MARKER)
        lines.append(f"{token['id']} {int(token['special'])} {hex_string(match)}")

    merges = model["merges"]
    lines.append(f"merges {len(merges)}")
    for merge in merges:
        # Older files write a merge as "left right", newer ones as a pair.
        parts = merge.split(" ") if isinstance(merge, str) else merge
        if len(parts) != 2:
            raise UnsupportedTokenizer(f"merge {merge!r} is not a pair")
        left, right = parts
        lines.append(f"{hex_string(left)} {hex_string(right)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote a tokenizer with {len(pieces)} tokens and {len(merges)} merges to {output_path}")


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

    tokenizer_path = output_dir / "tokenizer.txt"
    try:
        write_tokenizer(model_dir / "tokenizer.json", config, tokenizer_path)
    except UnsupportedTokenizer as error:
        tokenizer_path.unlink(missing_ok=True)
        print(f"Skipped the tokenizer, so prompts must be token IDs: {error}")


if __name__ == "__main__":
    main()
