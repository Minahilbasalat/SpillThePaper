"""Accuracy of the rule-based question features against HAND labels.

1. python -m src.data.build_question_table      -> data/processed/question_label_sample.csv
2. Team members fill the columns `true_qtype` (one of: mcq, true_false, fill_blank, code,
   numerical, proof_derivation, diagram_design, theory, scenario, fragment, other) and
   `true_marks` (number, or empty if the question states no marks). Split the 100 rows
   between the three of you; label WITHOUT looking at the predicted columns.
3. python -m src.evaluation.question_type_check
"""
import numpy as np
import pandas as pd

PATH = "data/processed/question_label_sample.csv"
OUT = "results/question_features_check.csv"


def main():
    d = pd.read_csv(PATH)
    lab = d[d["true_qtype"].notna() & (d["true_qtype"].astype(str).str.strip() != "")].copy()
    if lab.empty:
        raise SystemExit("no labelled rows yet: fill true_qtype first")
    lab["true_qtype"] = lab["true_qtype"].str.strip().str.lower()
    n = len(lab)
    acc = (lab["qtype"] == lab["true_qtype"]).mean()
    print(f"question type: {acc:.1%} correct on {n} hand-labelled questions")
    per = (lab.assign(ok=lab["qtype"] == lab["true_qtype"])
           .groupby("true_qtype")["ok"].agg(["mean", "size"]).round(2))
    print(per.to_string())
    print("\nmost frequent confusions (true -> predicted):")
    wrong = lab[lab["qtype"] != lab["true_qtype"]]
    print(wrong.groupby(["true_qtype", "qtype"]).size().sort_values(ascending=False).head(8).to_string())

    # marks: compare parsed marks with hand-read marks (both missing counts as agreement)
    tm = pd.to_numeric(lab["true_marks"], errors="coerce")
    pm = pd.to_numeric(lab["marks"], errors="coerce")
    agree = ((tm.isna() & pm.isna()) | np.isclose(tm.fillna(-1), pm.fillna(-2)))
    print(f"\nmarks: {agree.mean():.1%} agree "
          f"(has marks: parsed {pm.notna().mean():.0%}, hand-read {tm.notna().mean():.0%})")
    pd.DataFrame([{"n_labelled": n, "qtype_accuracy": round(acc, 3),
                   "marks_agreement": round(agree.mean(), 3)}]).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()