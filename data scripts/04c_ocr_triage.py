"""Step 4c - what did the OCR reject, and does it matter?

Reads : data/processed/ocr_results.csv, data/processed/papers_clean.csv
Writes: data/processed/ocr_triage.csv   (rejected files that look rescuable)
Prints: where the rejected files are (confidence band, type, year, subject).

Rejected files are mostly handwritten answer sheets (not useful as questions anyway) or
poor photographs. 'rescuable' = printed-looking text (enough words, medium confidence).
"""
import pandas as pd

from common import PROCESSED_DIR

ocr = pd.read_csv(PROCESSED_DIR / "ocr_results.csv", dtype={"file_id": str})
papers = pd.read_csv(PROCESSED_DIR / "papers_clean.csv", dtype={"file_id": str})
rej = ocr[ocr["status"] != "ok"].merge(
    papers[["file_id", "subject", "year", "assessment_type", "extension"]], on="file_id", how="left")
rej["in_clean_set"] = rej["subject"].notna()
rej["conf_band"] = pd.cut(rej["mean_conf"], [-1, 30, 45, 55, 101],
                          labels=["<30 (handwriting?)", "30-45", "45-55 (borderline)", ">=55"])
rej["rescuable"] = (rej["mean_conf"] >= 40) & (rej["words"] >= 60)

print(f"OCR files: {len(ocr)} | ok {(ocr.status == 'ok').sum()} | rejected {len(rej)}")
print("\nrejected by confidence band:\n", rej["conf_band"].value_counts().sort_index().to_string())
print("\nrejected by file type:\n", rej["extension"].value_counts().to_string())
clean = rej[rej["in_clean_set"]]
print(f"\nrejected files that are in the clean paper set: {len(clean)}")
print(clean["assessment_type"].value_counts().to_string())
print("recent (2023+):", int((clean["year"] >= 2023).sum()), "| rescuable:", int(clean["rescuable"].sum()))
print("\nsubjects losing most papers:\n", clean["subject"].value_counts().head(12).to_string())
rej[rej["rescuable"]].sort_values("mean_conf", ascending=False).to_csv(
    PROCESSED_DIR / "ocr_triage.csv", index=False, encoding="utf-8")