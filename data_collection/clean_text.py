"""
Nelson LLM — Multilingual Text Cleaning Pipeline
Cleans and balances Kinyarwanda + English + French text.

Key changes vs. monolingual version:
  - Per-language quality filtering
  - Language detection gating
  - Weighted sampling to maintain 60% rw / 25% en / 15% fr ratio
  - Outputs clean train/val/test splits
"""

import re
import os
import random
import hashlib
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Import language detector
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.lang_detect import detect_language

RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/processed")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Target corpus mix (by character count)
LANG_RATIOS = {"rw": 0.60, "en": 0.25, "fr": 0.15}

# Noise patterns (language-agnostic)
NOISE_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+\.\S+|<[^>]+>|"
    r"\{[^}]+\}|\[[^\]]*\]|={2,}.*?={2,}|#\w+|@\w+|\d{10,}",
    re.IGNORECASE,
)

# Per-language min/max line length
LANG_LENGTH = {
    "rw": (30, 2000),
    "en": (40, 2000),
    "fr": (40, 2000),
}


def clean_line(line: str) -> str:
    """Universal text cleaner."""
    line = unicodedata.normalize("NFC", line)
    line = NOISE_RE.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip()

    # Must be mostly alphabetic
    if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.45:
        return ""
    return line


def process_file(path: Path) -> dict[str, list[str]]:
    """
    Load and clean a text file, group lines by detected language.
    Returns dict: {lang_code: [lines]}
    """
    results = {"rw": [], "en": [], "fr": []}
    if not path.exists() or path.stat().st_size < 100:
        return results

    # Infer expected language from filename
    if "_rw" in path.name:
        expected_lang = "rw"
    elif "_en" in path.name:
        expected_lang = "en"
    elif "_fr" in path.name:
        expected_lang = "fr"
    else:
        expected_lang = None  # Detect dynamically

    print(f"\n  Processing {path.name} ({path.stat().st_size/1e6:.1f} MB) ...")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    kept = 0
    for line in tqdm(raw_lines, desc=f"  {path.stem}", ncols=80, leave=False):
        line = line.strip()
        if not line:
            continue
        cleaned = clean_line(line)
        if not cleaned:
            continue

        # Language detection
        if expected_lang:
            lang = expected_lang
            # Spot-check: if clearly wrong language, skip
            detected = detect_language(cleaned)
            if detected != expected_lang and len(cleaned) > 80:
                continue  # Discard lines that don't match expected lang
        else:
            lang = detect_language(cleaned)
            if lang not in ("rw", "en", "fr"):
                continue

        # Length filter
        min_len, max_len = LANG_LENGTH[lang]
        if len(cleaned) < min_len or len(cleaned) > max_len:
            continue

        results[lang].append(cleaned)
        kept += 1

    print(f"  → rw={len(results['rw']):,}  en={len(results['en']):,}  fr={len(results['fr']):,}  (kept {kept:,})")
    return results


def deduplicate(lines: list[str]) -> list[str]:
    seen = set()
    unique = []
    for line in lines:
        h = hashlib.md5(line.lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(line)
    removed = len(lines) - len(unique)
    if removed:
        print(f"  Removed {removed:,} duplicates → {len(unique):,} unique")
    return unique


def balance_corpus(lang_buckets: dict[str, list[str]]) -> list[str]:
    """
    Sample from each language bucket according to LANG_RATIOS.
    Strictly anchors the total size to ensure the ratio is maintained,
    usually bottlenecked by the primary language (Kinyarwanda).
    """
    totals = {lang: len(lines) for lang, lines in lang_buckets.items()}
    print(f"\n  Available before balancing:")
    for lang, n in totals.items():
        print(f"    {lang}: {n:,} lines")

    # Anchor the total dataset size based on Kinyarwanda to force the 60% ratio
    rw_available = totals.get("rw", 0)
    rw_ratio = LANG_RATIOS["rw"]
    
    # Calculate what the total corpus size SHOULD be to maintain the ratio
    max_total = int(rw_available / rw_ratio)
    
    targets = {}
    for lang, ratio in LANG_RATIOS.items():
        want = int(max_total * ratio)
        # If we don't have enough of a secondary language, we cap it
        targets[lang] = min(want, totals.get(lang, 0))

    print(f"\n  After balancing (Target Ratio: {LANG_RATIOS}):")
    all_lines = []
    for lang, n in targets.items():
        sample = random.sample(lang_buckets[lang], n) if n > 0 else []
        # Tag each line with language token for the model to learn
        tagged = [f"<|lang_{lang}|> {line}" for line in sample]
        all_lines.extend(tagged)
        
        actual_ratio = (n / max(sum(targets.values()), 1)) * 100
        print(f"    {lang}: {n:,} lines ({actual_ratio:.0f}%)")

    return all_lines


def split_and_save(lines: list[str]):
    random.seed(42)
    random.shuffle(lines)

    n = len(lines)
    n_val  = max(1000, int(n * 0.01))
    n_test = max(500,  int(n * 0.005))
    n_train = n - n_val - n_test

    train = lines[:n_train]
    val   = lines[n_train:n_train + n_val]
    test  = lines[n_train + n_val:]

    for split, data in [("train", train), ("val", val), ("test", test)]:
        out = CLEAN_DIR / f"{split}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(data))
        print(f"  ✓ {split}.txt : {len(data):,} lines ({out.stat().st_size/1e6:.1f} MB)")

    return train, val, test


def main():
    print("=" * 65)
    print("  NELSON AI — MULTILINGUAL TEXT CLEANING")
    print("  Languages: Kinyarwanda + English + French")
    print("=" * 65)

    lang_buckets: dict[str, list[str]] = {"rw": [], "en": [], "fr": []}

    # Process all raw files
    for raw_file in sorted(RAW_DIR.glob("*.txt")):
        file_results = process_file(raw_file)
        for lang, lines in file_results.items():
            lang_buckets[lang].extend(lines)

    # Deduplicate per language
    print("\n  Deduplicating ...")
    for lang in lang_buckets:
        lang_buckets[lang] = deduplicate(lang_buckets[lang])

    if all(len(v) == 0 for v in lang_buckets.values()):
        print("\n✗ No data found! Run data_collection/download_data.py first.")
        return

    # Balance and merge
    all_lines = balance_corpus(lang_buckets)

    # Split and save
    print(f"\n  Saving splits ...")
    train, val, test = split_and_save(all_lines)

    # Stats
    total = len(train)
    total_chars = sum(len(l) for l in train)
    print(f"\n  Training corpus:")
    print(f"    Lines      : {total:,}")
    print(f"    Characters : {total_chars:,}")
    print(f"    Avg length : {total_chars // max(total, 1)} chars/line")

    print("\nNext: python tokenizer/train_tokenizer.py")


if __name__ == "__main__":
    main()
