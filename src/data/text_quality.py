"""Detect garbled OCR text.

quality(text, known) = share of the words (3+ letters) that are 'known', i.e. that occur in at
least MIN_DF different questions of the whole question bank. Real questions score about 1.0,
OCR noise ("Total tres delay f GONS sews t ARTT+ Sil trams ...") scores about 0.3.
Questions below LOW_QUALITY are never used to build papers (they are still listed with a
warning so that nothing silently disappears)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWN_WORDS_PATH = ROOT / "data" / "processed" / "known_words.txt"
TOKEN = re.compile(r"[A-Za-z]{3,}")
MIN_DF = 4
LOW_QUALITY = 0.6


def build_known_words(texts, min_df: int = MIN_DF) -> set:
    df = Counter()
    for t in texts:
        df.update({w.lower() for w in TOKEN.findall(t or "")})
    return {w for w, n in df.items() if n >= min_df}


def quality(text: str, known) -> float:
    toks = [w.lower() for w in TOKEN.findall(text or "")]
    if len(toks) < 4:
        return 1.0                      # too short to judge
    return sum(w in known for w in toks) / len(toks)


def save_known_words(known, path: Path = KNOWN_WORDS_PATH):
    path.write_text("\n".join(sorted(known)), encoding="utf-8")


def load_known_words(path: Path = KNOWN_WORDS_PATH):
    return set(path.read_text(encoding="utf-8").split()) if path.exists() else None
