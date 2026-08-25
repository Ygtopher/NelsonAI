"""
Nelson LLM — Dataset Tokenization
Converts clean text files into binary token ID arrays for fast training.
Runs once before training starts.
"""

import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
TOKENIZED_DIR = Path("data/tokenized")
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = Path("tokenizer/nelson_tokenizer/tokenizer.json")
CHUNK_SIZE = 512  # context length


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}.\n"
            "Run: python tokenizer/train_tokenizer.py"
        )
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok.enable_truncation(max_length=100_000)  # Disable truncation for full docs
    tok.no_padding()
    return tok


def tokenize_file(tokenizer: Tokenizer, input_path: Path, output_path: Path):
    """Tokenize a text file and save as numpy uint16 array."""
    if output_path.exists():
        n_tokens = np.load(output_path, mmap_mode="r").shape[0]
        print(f"  ✓ Already tokenized: {output_path.name} ({n_tokens:,} tokens)")
        return n_tokens

    print(f"  Tokenizing {input_path.name} ...")
    all_ids = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    eos_id = tokenizer.token_to_id("<|eos|>")

    for line in tqdm(lines, desc=f"  {input_path.name}", ncols=80):
        line = line.strip()
        if not line:
            continue
        encoded = tokenizer.encode(line)
        # Append token IDs + EOS separator between documents
        all_ids.extend(encoded.ids)
        all_ids.append(eos_id)

    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  ✓ {output_path.name}: {len(arr):,} tokens ({size_mb:.1f} MB)")
    return len(arr)


def main():
    print("=" * 60)
    print("  NELSON — DATASET TOKENIZATION")
    print("=" * 60)

    tokenizer = load_tokenizer()
    print(f"  Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size():,}")

    total_tokens = 0
    for split in ["train", "val", "test"]:
        in_path  = PROCESSED_DIR / f"{split}.txt"
        out_path = TOKENIZED_DIR / f"{split}.npy"
        if in_path.exists() and in_path.stat().st_size > 0:
            n = tokenize_file(tokenizer, in_path, out_path)
            total_tokens += n

    print(f"\n  Total tokens: {total_tokens:,}")
    print(f"  Estimated training steps (batch=4, seq=512, grad_acc=16):")
    steps = total_tokens // (4 * 512 * 16)
    print(f"    ~{steps:,} steps per epoch")

    print("\n✓ Tokenization complete!")
    print("Next step: python training/trainer.py")


if __name__ == "__main__":
    main()
