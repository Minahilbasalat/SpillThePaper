"""Split the 100 hand-labelling rows into 3 files (one per team member).
    python scripts/label_split.py
Each file has ONLY the question text (no predictions, so nobody is biased).
Open labeler_1.csv ... labeler_3.csv in Excel, fill the two empty columns, save as CSV."""
from pathlib import Path

import pandas as pd

P = Path("data/processed")
d = pd.read_csv(P / "question_label_sample.csv")
d["row_id"] = range(len(d))
out = P / "labeling"
out.mkdir(exist_ok=True)
for i in range(3):
    part = d[d["row_id"] % 3 == i][["row_id", "subject", "question_text"]].copy()
    part["true_qtype"] = ""
    part["true_marks"] = ""
    part.to_csv(out / f"labeler_{i + 1}.csv", index=False, encoding="utf-8-sig")
print("created data/processed/labeling/labeler_1.csv, labeler_2.csv, labeler_3.csv")
print("true_qtype must be ONE of: mcq, true_false, fill_blank, code, numerical, "
      "proof_derivation, diagram_design, theory, scenario, fragment, other")
print("true_marks: the marks written in the question as a number, or leave EMPTY if none is written")