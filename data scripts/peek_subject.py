"""Show the first lines of a couple of readable papers/outlines per folder name,
so you can check whether two folder names are really the same course.

Usage:  python "data scripts/peek_subject.py" "DW" "Data Warehousing and Data Mining"
"""
import sys

import pandas as pd

from common import PROCESSED_DIR, ROOT

tables = [pd.read_csv(PROCESSED_DIR / name, keep_default_na=False) for name in ("papers.csv", "outlines.csv")]
everything = pd.concat(tables, ignore_index=True)

for name in sys.argv[1:]:
    rows = everything[(everything["subject_raw"] == name) & (everything["has_text"].astype(str) == "True")]
    print("=" * 72)
    print(f"{name}   ({len(rows)} readable files)")
    for row in rows.head(2).itertuples():
        text = (ROOT / "data" / "interim" / "text" / f"{row.file_id}.txt").read_text(encoding="utf-8")
        print("-" * 72)
        print(row.repo_path)
        print(" ".join(text.split())[:350])