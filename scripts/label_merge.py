"""Put the 3 filled files back together.   python scripts/label_merge.py"""
from pathlib import Path

import pandas as pd

P = Path("data/processed")
d = pd.read_csv(P / "question_label_sample.csv")
d["row_id"] = range(len(d))
lab = pd.concat([pd.read_csv(P / "labeling" / f"labeler_{i}.csv") for i in (1, 2, 3)])
lab = lab.set_index("row_id")[["true_qtype", "true_marks"]]
d = d.drop(columns=["true_qtype", "true_marks"]).merge(lab, left_on="row_id", right_index=True, how="left")
d.drop(columns="row_id").to_csv(P / "question_label_sample.csv", index=False, encoding="utf-8")
print("labelled rows:", int(d["true_qtype"].notna().sum()), "of", len(d))