"""Load the extracted text of the clean exam papers."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "interim" / "text"
PAPERS_CLEAN = ROOT / "data" / "processed" / "papers_clean.csv"
PAPER_TYPES = ["quiz", "mid", "final"]


def build_text_index(text_dir: Path = TEXT_DIR) -> dict:
    """Map file name (without extension) -> path, for every file under text_dir."""
    return {p.stem: p for p in text_dir.rglob("*") if p.is_file()}


def find_text_path(file_id: str, index: dict):
    """Find the text file of a paper. Exact match first, then 'contains file_id'."""
    if file_id in index:
        return index[file_id]
    for stem, path in index.items():
        if file_id in stem:
            return path
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_papers_with_text(require_year: bool = True,
                          include_solutions: bool = True) -> pd.DataFrame:
    """Clean papers that have readable text, with a new 'text' column.

    require_year: drop papers without a year (needed for a time-based split).
    include_solutions: solution files without a plain companion are kept by
        default, because they are often the only copy of the paper.
    """
    papers = pd.read_csv(PAPERS_CLEAN, dtype={"file_id": str})
    papers = papers[papers["has_text"]]
    papers = papers[papers["assessment_type"].isin(PAPER_TYPES)]
    if require_year:
        papers = papers[papers["year"].notna()]
    if not include_solutions:
        papers = papers[~papers["is_solution"]]
    papers = papers.copy()
    papers["year"] = papers["year"].astype(int)

    index = build_text_index()
    paths = papers["file_id"].map(lambda f: find_text_path(f, index))
    missing = int(paths.isna().sum())
    if missing:
        print(f"[load_text] WARNING: {missing} papers have no text file and were dropped")
    keep = paths.notna()
    papers = papers[keep].copy()
    papers["text"] = [read_text(p) for p in paths[keep]]
    return papers.reset_index(drop=True)


if __name__ == "__main__":
    df = load_papers_with_text()
    print("papers with text and year:", len(df))
    print(df["assessment_type"].value_counts())
    print("subjects:", df["subject"].nunique())
    print(df["text"].str.len().describe())