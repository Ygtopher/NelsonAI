"""
Nelson LLM — Multilingual BPE Tokenizer Training
Trains a single shared Byte Pair Encoding tokenizer on:
  - Kinyarwanda (primary, ~60% of corpus)
  - English     (secondary, ~25%)
  - French      (secondary, ~15%)
Vocabulary: 32,000 tokens (covers all 3 languages efficiently)
Uses HuggingFace `tokenizers` library.
"""

import os
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence

PROCESSED_DIR  = Path("data/processed")
TOKENIZER_DIR  = Path("tokenizer/nelson_tokenizer")
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# ── Special Tokens ─────────────────────────────────────────────────
SPECIAL_TOKENS = [
    "<|pad|>",       # 0  — padding
    "<|bos|>",       # 1  — beginning of sequence
    "<|eos|>",       # 2  — end of sequence
    "<|unk|>",       # 3  — unknown token
    "<|user|>",      # 4  — user turn marker
    "<|nelson|>",    # 5  — Nelson's turn marker
    "<|system|>",    # 6  — system prompt marker
    # ── Language control tokens ──────────────────────
    "<|lang_rw|>",   # 7  — Kinyarwanda language tag
    "<|lang_en|>",   # 8  — English language tag
    "<|lang_fr|>",   # 9  — French language tag
    # ── Tool-calling tokens ───────────────────────
    "<|search|>",    # 10 — start web search
    "<|/search|>",   # 11 — end web search
    "<|wiki|>",      # 12 — start Wikipedia lookup
    "<|/wiki|>",     # 13 — end Wikipedia lookup
    "<|fetch|>",     # 14 — start URL fetch
    "<|/fetch|>",    # 15 — end URL fetch
    "<|result|>",    # 16 — start tool result
    "<|/result|>",   # 17 — end tool result
]

VOCAB_SIZE = 32_000   # Covers Kinyarwanda + English + French efficiently


def get_training_files() -> list[str]:
    """Return list of text files to train the tokenizer on."""
    files = []
    for split in ["train.txt", "val.txt"]:
        p = PROCESSED_DIR / split
        if p.exists() and p.stat().st_size > 0:
            files.append(str(p))

    if not files:
        print("✗ No processed data found. Run data_collection/clean_text.py first.")
        raise FileNotFoundError("No training data for tokenizer.")

    total_mb = sum(Path(f).stat().st_size for f in files) / 1e6
    print(f"  Training tokenizer on {len(files)} files ({total_mb:.1f} MB)")
    return files


def train_tokenizer():
    print("=" * 60)
    print("  NELSON — KINYARWANDA TOKENIZER TRAINING")
    print("=" * 60)

    # ── Build BPE tokenizer ──────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Byte-level pre-tokenization: splits on whitespace + handles special chars
    # This ensures every possible byte can be represented (no true UNKs)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # ── Trainer ──────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=3,           # Token must appear ≥3 times
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # ── Train ────────────────────────────────────────────────────
    files = get_training_files()
    print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE}) ...")
    tokenizer.train(files=files, trainer=trainer)
    print(f"  ✓ Training complete. Vocabulary size: {tokenizer.get_vocab_size():,}")

    # ── Post-processor: add BOS/EOS automatically ────────────────
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<|pad|>"),
        pad_token="<|pad|>",
    )

    # ── Save ─────────────────────────────────────────────────────
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  ✓ Tokenizer saved: {tokenizer_path}")

    # Also save a human-readable vocab map
    vocab = tokenizer.get_vocab()
    vocab_path = TOKENIZER_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Vocabulary saved: {vocab_path}")

    # Save token ID map for easy reference
    special_map = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_map,
        "model_type": "bpe",
        "tokenizer_class": "NelsonTokenizer",
    }
    with open(TOKENIZER_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return tokenizer


def verify_tokenizer(tokenizer: Tokenizer):
    """Quick sanity check across all three languages."""
    print("\n" + "=" * 60)
    print("  TOKENIZER VERIFICATION (Trilingual)")
    print("=" * 60)

    test_sentences = [
        ("rw", "Muraho! Amakuru yawe?"),
        ("rw", "Ndi mwiza, urakoze. Nagufasha bite?"),
        ("rw", "Igihugu cya Rwanda ni icyiza cyane."),
        ("en", "Hello! How can I help you today?"),
        ("en", "Rwanda is a country in East Africa with a rich culture."),
        ("fr", "Bonjour! Comment puis-je vous aider?"),
        ("fr", "Le Rwanda est un beau pays en Afrique de l'Est."),
    ]

    for lang, sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        n_tokens = len(encoded.ids)
        efficiency = len(sentence) / max(n_tokens, 1)
        print(f"\n  [{lang.upper()}] {sentence}")
        print(f"  Tokens  : {n_tokens} ({efficiency:.1f} chars/token)")
        print(f"  Decoded : {decoded}")

    print("\n  Special token IDs:")
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        print(f"    {tok:<20} → {tid}")


if __name__ == "__main__":
    tokenizer = train_tokenizer()
    verify_tokenizer(tokenizer)
    print("\n✓ Tokenizer ready!")
    print("Next step: python training/tokenize_dataset.py")
