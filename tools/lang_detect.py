"""
Nelson AI — Language Detection Utility
Identifies whether text is Kinyarwanda, English, or French.
Used during data collection, cleaning, and chat routing.
"""

import re
from typing import Optional


# ── Word frequency heuristics per language ────────────────────────

LANG_WORD_SETS = {
    "rw": {  # Kinyarwanda
        "ndi", "ni", "na", "mu", "ku", "wa", "ya", "za", "ba", "ko",
        "kandi", "ariko", "ngo", "kuko", "naho", "cyane", "cyangwa",
        "ubwo", "iyo", "nibwo", "nubwo", "aho", "ubundi", "maze",
        "umuntu", "abantu", "inzira", "amakuru", "akazi", "abana",
        "igihugu", "gutuza", "gukora", "kugenda", "kubona", "kwiga",
        "umuryango", "imyaka", "ibirori", "abakozi", "Rwanda",
        "muraho", "murakoze", "yego", "oya", "bite", "amakori",
        "ubu", "aho", "hano", "iki", "nki", "iki", "icyo",
    },
    "en": {  # English
        "the", "and", "is", "in", "it", "of", "to", "a", "that",
        "was", "for", "on", "are", "with", "as", "at", "be", "by",
        "this", "which", "from", "or", "an", "but", "not", "have",
        "he", "she", "they", "we", "you", "I", "said", "about",
        "there", "when", "who", "will", "more", "been", "one",
        "would", "their", "were", "has", "its", "also", "into",
    },
    "fr": {  # French
        "le", "la", "les", "de", "du", "des", "un", "une", "et",
        "en", "à", "que", "qui", "il", "elle", "ils", "elles",
        "nous", "vous", "ce", "se", "son", "sa", "ses", "dans",
        "sur", "par", "pour", "avec", "est", "sont", "mais",
        "ou", "donc", "ni", "car", "plus", "aussi", "tout",
        "comme", "bien", "très", "même", "fait", "peut", "être",
    },
}

# Script-level markers (fast heuristic before word-level check)
FRENCH_MARKERS = re.compile(r"[àâäéèêëîïôùûüœæç]", re.IGNORECASE)
NON_LATIN = re.compile(r"[^\x00-\x024F\s]")  # Non-Latin script


def detect_language(text: str, threshold: float = 0.04) -> str:
    """
    Detect the primary language of a text snippet.
    Returns ISO 639-1 code: 'rw', 'en', or 'fr'. Defaults to 'rw'.
    """
    if not text or len(text.strip()) < 10:
        return "rw"

    words = re.findall(r"\b[a-zA-ZàâäéèêëîïôùûüœæçÀÂÄÉÈÊËÎÏÔÙÛÜŒÆÇ']+\b", text.lower())
    if not words:
        return "rw"

    word_set = set(words)
    total = len(words)

    scores = {}
    for lang, vocab in LANG_WORD_SETS.items():
        overlap = len(word_set & {w.lower() for w in vocab})
        scores[lang] = overlap / total

    # Boost French if accented characters present
    if FRENCH_MARKERS.search(text):
        scores["fr"] = scores.get("fr", 0) + 0.05

    # Return the highest-scoring language above threshold
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] >= threshold:
        return best_lang

    return "rw"  # Default to Kinyarwanda


def is_accepted_language(text: str) -> bool:
    """Return True if text is in Kinyarwanda, English, or French."""
    return detect_language(text) in ("rw", "en", "fr")


def get_lang_label(lang_code: str) -> str:
    return {"rw": "Kinyarwanda", "en": "English", "fr": "French"}.get(lang_code, "Unknown")


if __name__ == "__main__":
    tests = [
        ("Muraho! Amakuru yawe?", "rw"),
        ("Hello, how are you doing today?", "en"),
        ("Bonjour! Comment allez-vous aujourd'hui?", "fr"),
        ("The government of Rwanda announced new policies.", "en"),
        ("Le gouvernement du Rwanda a annoncé de nouvelles politiques.", "fr"),
        ("Igihugu cya Rwanda gifite abantu benshi.", "rw"),
    ]
    print("Language Detection Test:")
    for text, expected in tests:
        detected = detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} [{detected}] expected=[{expected}]  {text[:50]}")
