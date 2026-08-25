"""
Nelson AI — Multilingual Data Collection (v3 — Robust)
Uses only methods proven to work with datasets v5+.

Strategy:
  - Small/medium datasets: load_dataset() non-streaming
  - Large datasets: download Parquet files directly via huggingface_hub
  - Skip anything that uses loading scripts (.py)

Already collected from previous run:
  ✓ wikipedia_rw.txt (11.3 MB)
  ✓ opus_rw.txt (11.6 MB)
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_texts(texts: list[str], out_path: Path, label: str) -> int:
    """Write texts to file. Skips if already exists and non-empty."""
    if out_path.exists() and out_path.stat().st_size > 5_000:
        size = out_path.stat().st_size / 1e6
        print(f"  ✓ Already exists: {out_path.name} ({size:.1f} MB)")
        return 0
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            t = t.strip()
            if len(t) > 50:
                f.write(t + "\n\n")
                count += 1
    size = out_path.stat().st_size / 1e6
    print(f"  ✓ {label}: {count:,} texts → {out_path.name} ({size:.1f} MB)")
    return count


def download_parquet_texts(repo_id: str, config: str, split: str,
                           text_col: str, max_files: int = 5,
                           max_texts: int = 200_000) -> list[str]:
    """
    Download Parquet files directly from HuggingFace Hub and extract text.
    This bypasses the datasets library entirely — no script issues.
    """
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    texts = []

    try:
        # List parquet files in the repo
        files = api.list_repo_tree(repo_id, path_in_repo=f"data/{split}",
                                    repo_type="dataset")
        parquet_files = [f.rfile_path for f in files
                         if hasattr(f, 'rfile_path') and f.rfile_path.endswith(".parquet")]

        if not parquet_files:
            # Try config subfolder
            files = api.list_repo_tree(repo_id, path_in_repo=f"{config}/{split}",
                                        repo_type="dataset")
            parquet_files = [f.rfile_path for f in files
                             if hasattr(f, 'rfile_path') and f.rfile_path.endswith(".parquet")]

        if not parquet_files:
            return texts

        for pf in parquet_files[:max_files]:
            local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
            df = pd.read_parquet(local_path)
            if text_col in df.columns:
                for text in df[text_col].dropna():
                    if isinstance(text, str) and len(text) > 50:
                        texts.append(text)
                        if len(texts) >= max_texts:
                            return texts
    except Exception as e:
        print(f"    Parquet download failed: {e}")

    return texts


# ─────────────────────────────────────────────────────────────────
# KINYARWANDA
# ─────────────────────────────────────────────────────────────────

def download_kinyarwanda():
    print("\n" + "─" * 60)
    print("  [RW] Kinyarwanda Sources")
    print("─" * 60)

    # 1. Wikipedia RW — already collected, check
    wiki_rw = RAW_DIR / "wikipedia_rw.txt"
    if wiki_rw.exists() and wiki_rw.stat().st_size > 5_000:
        print(f"  ✓ Wikipedia RW: already downloaded ({wiki_rw.stat().st_size/1e6:.1f} MB)")
    else:
        print("  Loading wikimedia/wikipedia [20231101.rw] ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikimedia/wikipedia", "20231101.rw", split="train")
            texts = [row["text"] for row in ds if len(row.get("text", "")) > 50]
            save_texts(texts, wiki_rw, "Wikipedia RW")
        except Exception as e:
            print(f"  ✗ Wikipedia RW: {e}")

    # 2. OPUS-100 RW — already collected, check
    opus_rw = RAW_DIR / "opus_rw.txt"
    if opus_rw.exists() and opus_rw.stat().st_size > 5_000:
        print(f"  ✓ OPUS-100 RW: already downloaded ({opus_rw.stat().st_size/1e6:.1f} MB)")
    else:
        print("  Loading Helsinki-NLP/opus-100 [en-rw] ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Helsinki-NLP/opus-100", "en-rw", split="train",
                              streaming=True)
            texts = []
            for i, row in enumerate(tqdm(ds, total=200_000, ncols=80, leave=False)):
                tr = row.get("translation", {})
                rw = tr.get("rw", "").strip()
                if len(rw) > 20:
                    texts.append(rw)
                if i >= 200_000:
                    break
            save_texts(texts, opus_rw, "OPUS-100 RW")
        except Exception as e:
            print(f"  ✗ OPUS-100 RW: {e}")

    # 3. Extra: Flores-200 Kinyarwanda benchmark (small but high quality)
    flores_rw = RAW_DIR / "flores_rw.txt"
    if not (flores_rw.exists() and flores_rw.stat().st_size > 1_000):
        print("  Loading facebook/flores [kin_Latn] ...")
        try:
            from datasets import load_dataset
            texts = []
            for split_name in ["dev", "devtest"]:
                ds = load_dataset("facebook/flores", "kin_Latn", split=split_name)
                texts.extend([row["sentence"] for row in ds if len(row.get("sentence", "")) > 10])
            save_texts(texts, flores_rw, "Flores RW")
        except Exception as e:
            print(f"  ✗ Flores RW: {e}")


# ─────────────────────────────────────────────────────────────────
# ENGLISH
# ─────────────────────────────────────────────────────────────────

def download_english():
    print("\n" + "─" * 60)
    print("  [EN] English Sources")
    print("─" * 60)

    # 1. Wikitext-103 (clean, curated English prose — perfect for LM training)
    wikitext = RAW_DIR / "wikitext_en.txt"
    if not (wikitext.exists() and wikitext.stat().st_size > 5_000):
        print("  Loading Salesforce/wikitext [wikitext-103-raw-v1] ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
            texts = [row["text"] for row in ds if len(row.get("text", "").strip()) > 50]
            save_texts(texts, wikitext, "Wikitext-103 EN")
        except Exception as e:
            print(f"  ✗ Wikitext: {e}")
    else:
        print(f"  ✓ Wikitext EN: already downloaded ({wikitext.stat().st_size/1e6:.1f} MB)")

    # 2. OPUS-100 English side (from en-fr pairs)
    opus_en = RAW_DIR / "opus_en.txt"
    if not (opus_en.exists() and opus_en.stat().st_size > 5_000):
        print("  Loading Helsinki-NLP/opus-100 [en-fr] → English side ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train",
                              streaming=True)
            texts = []
            for i, row in enumerate(tqdm(ds, total=200_000, ncols=80, leave=False)):
                en = row.get("translation", {}).get("en", "").strip()
                if len(en) > 20:
                    texts.append(en)
                if i >= 200_000:
                    break
            save_texts(texts, opus_en, "OPUS-100 EN")
        except Exception as e:
            print(f"  ✗ OPUS-100 EN: {e}")
    else:
        print(f"  ✓ OPUS-100 EN: already downloaded ({opus_en.stat().st_size/1e6:.1f} MB)")

    # 3. Tiny Stories (very clean, simple English — good for small LMs)
    tiny = RAW_DIR / "tinystories_en.txt"
    if not (tiny.exists() and tiny.stat().st_size > 5_000):
        print("  Loading roneneldan/TinyStories ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            texts = []
            for i, row in enumerate(tqdm(ds, total=100_000, ncols=80, leave=False)):
                text = row.get("text", "").strip()
                if len(text) > 50:
                    texts.append(text)
                if i >= 100_000:
                    break
            save_texts(texts, tiny, "TinyStories EN")
        except Exception as e:
            print(f"  ✗ TinyStories: {e}")
    else:
        print(f"  ✓ TinyStories EN: already downloaded ({tiny.stat().st_size/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────
# FRENCH
# ─────────────────────────────────────────────────────────────────

def download_french():
    print("\n" + "─" * 60)
    print("  [FR] French Sources")
    print("─" * 60)

    # 1. OPUS-100 French side (from en-fr pairs)
    opus_fr = RAW_DIR / "opus_fr.txt"
    if not (opus_fr.exists() and opus_fr.stat().st_size > 5_000):
        print("  Loading Helsinki-NLP/opus-100 [en-fr] → French side ...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train",
                              streaming=True)
            texts = []
            for i, row in enumerate(tqdm(ds, total=200_000, ncols=80, leave=False)):
                fr = row.get("translation", {}).get("fr", "").strip()
                if len(fr) > 20:
                    texts.append(fr)
                if i >= 200_000:
                    break
            save_texts(texts, opus_fr, "OPUS-100 FR")
        except Exception as e:
            print(f"  ✗ OPUS-100 FR: {e}")
    else:
        print(f"  ✓ OPUS-100 FR: already downloaded ({opus_fr.stat().st_size/1e6:.1f} MB)")

    # 2. Flores-200 French (small but very high quality)
    flores_fr = RAW_DIR / "flores_fr.txt"
    if not (flores_fr.exists() and flores_fr.stat().st_size > 1_000):
        print("  Loading facebook/flores [fra_Latn] ...")
        try:
            from datasets import load_dataset
            texts = []
            for split_name in ["dev", "devtest"]:
                ds = load_dataset("facebook/flores", "fra_Latn", split=split_name)
                texts.extend([row["sentence"] for row in ds if len(row.get("sentence", "")) > 10])
            save_texts(texts, flores_fr, "Flores FR")
        except Exception as e:
            print(f"  ✗ Flores FR: {e}")

    # 3. French Wikipedia via direct Parquet download (bypass datasets library)
    wiki_fr = RAW_DIR / "wikipedia_fr.txt"
    if not (wiki_fr.exists() and wiki_fr.stat().st_size > 5_000):
        print("  Downloading Wikipedia FR via HuggingFace Hub (Parquet) ...")
        texts = download_parquet_texts(
            "wikimedia/wikipedia", "20231101.fr", "train",
            text_col="text", max_files=2, max_texts=80_000
        )
        if texts:
            save_texts(texts, wiki_fr, "Wikipedia FR (Parquet)")
        else:
            print("  ✗ Wikipedia FR Parquet failed — OPUS-100 FR is enough")
    else:
        print(f"  ✓ Wikipedia FR: already downloaded ({wiki_fr.stat().st_size/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 65)
    print("  DATA COLLECTION SUMMARY")
    print("=" * 65)

    total_mb = 0
    rw_mb = en_mb = fr_mb = 0

    for f in sorted(RAW_DIR.glob("*.txt")):
        size = f.stat().st_size / 1e6
        if size < 0.001:
            continue
        total_mb += size

        name = f.name.lower()
        if "_rw" in name or "flores_rw" in name:
            rw_mb += size
        elif "_en" in name or "wikitext" in name or "tinystories" in name:
            en_mb += size
        elif "_fr" in name or "flores_fr" in name:
            fr_mb += size

        try:
            lines = sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
        except:
            lines = 0
        print(f"  {f.name:<35} {size:>7.1f} MB   {lines:>10,} lines")

    total = rw_mb + en_mb + fr_mb or 1
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {total_mb:.1f} MB")
    print(f"\n  Language mix:")
    print(f"    Kinyarwanda : {rw_mb:.1f} MB ({rw_mb/total*100:.0f}%)")
    print(f"    English     : {en_mb:.1f} MB ({en_mb/total*100:.0f}%)")
    print(f"    French      : {fr_mb:.1f} MB ({fr_mb/total*100:.0f}%)")

    if total_mb < 10:
        print("\n  ⚠ Very small corpus. Expect basic quality.")
    elif total_mb < 100:
        print("\n  ⚠ Moderate corpus. Quality will be limited but functional.")
    else:
        print("\n  ✓ Good corpus size!")

    print("=" * 65)
    print("\nNext: python data_collection/clean_text.py")


if __name__ == "__main__":
    print("=" * 65)
    print("  NELSON AI — MULTILINGUAL DATA COLLECTION (v3)")
    print("  Kinyarwanda + English + French")
    print("=" * 65)

    download_kinyarwanda()
    download_english()
    download_french()
    print_summary()
