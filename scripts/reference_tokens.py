"""Writes tokenizer cases from Hugging Face tokenizers for tests/tokenizer_test.cpp.

Usage: python reference_tokens.py <hf_model_dir> <output_file> [random_cases]

Each case is one line. Strings are UTF-8 hex, or "-" when empty:
  encode <text> <count> <ids...>
  decode <count> <ids...> <text with added tokens> <text without them>

The random cases mix spaces, newlines, accented letters, CJK, emoji and added
tokens. They use a fixed seed, so the file is the same on every run.
"""
import json
import pathlib
import random
import sys

import tokenizers
from tokenizers import Tokenizer

FIXED_TEXTS = [
    "",
    " ",
    "  ",
    "\n",
    "Once upon a time",
    "Once upon a time, there was a little girl named Lily.",
    "  leading spaces",
    "trailing spaces  ",
    "multiple   spaces\tand\ttabs",
    "\n\nHi  there\r\n",
    "café naïve résumé",
    "你好，世界",
    "rocket \U0001F680 and family \U0001F468‍\U0001F469‍\U0001F467",
    "a </s> b",
    "a</s>b",
    "</s>",
    "<s>hi",
    "x <unk> y",
    "<s></s><unk>",
    "▁already marked",
    "Numbers 0123456789 and symbols !@#$%^&*()[]{}<>",
    "It's \"quoted\" -- and (bracketed).",
    "The quick brown fox jumps over the lazy dog. " * 20,
]

ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-")
    + [" "] * 12
    + ["\n", "\t", "é", "ü", "ß", "你", "好", "あ", "ж", "\U0001F600", "\U0001F680", "▁"]
    + ["<s>", "</s>", "<unk>", " </s>", " <s>"]
)


def hex_string(text):
    return text.encode("utf-8").hex() or "-"


def main():
    if not 3 <= len(sys.argv) <= 4:
        sys.exit(__doc__)
    model_dir = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    random_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    added = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8")).get("added_tokens", [])
    special_ids = {token["id"] for token in added if token["special"]}
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    rng = random.Random(1234)

    texts = list(FIXED_TEXTS)
    for _ in range(random_cases):
        texts.append("".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 60))))

    # Decoding cases: the encoded texts, plus random ids that favor byte tokens,
    # so some byte runs are not valid UTF-8.
    id_lists = []
    for _ in range(random_cases):
        ids = []
        for _ in range(rng.randint(1, 20)):
            roll = rng.random()
            if roll < 0.4:
                ids.append(rng.randint(3, 258))  # <0x00> to <0xFF> in Llama's vocabulary
            elif roll < 0.5:
                ids.append(rng.choice(sorted(special_ids)))
            else:
                ids.append(rng.randrange(vocab_size))
        id_lists.append(ids)

    lines = ["baremetal-tokenizer-reference 1"]
    for text in texts:
        ids = tokenizer.encode(text).ids
        lines.append(f"encode {hex_string(text)} {len(ids)} " + " ".join(map(str, ids)))
        id_lists.append(ids)
    for ids in id_lists:
        # Drop added tokens explicitly. tokenizers' own skip_special_tokens keeps
        # Llama's <s>, because the token's stored text is normalized.
        with_added = tokenizer.decode(ids, skip_special_tokens=False)
        without_added = tokenizer.decode([i for i in ids if i not in special_ids], skip_special_tokens=False)
        lines.append(f"decode {len(ids)} " + " ".join(map(str, ids)) + f" {hex_string(with_added)} {hex_string(without_added)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"tokenizers {tokenizers.__version__}")
    print(f"Wrote {len(texts)} encoding and {len(id_lists)} decoding cases to {output_path}")


if __name__ == "__main__":
    main()
