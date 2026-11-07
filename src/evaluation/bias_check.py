"""Bias Check: does the best learned model perform unevenly across groups?

There are no demographic attributes in this data, so the subgroups are
data-coverage and assessment groups: assessment type, subject coverage tier
(training exams per subject), lab vs theory subjects, target year and amount
of history. All metrics are on the TEST set with the model and threshold fixed
on the VALIDATION set. Each group has a 95% cluster-bootstrap CI (exams are the
resampling unit). A gap counts as larger than the uncertainty only when the CIs
of the best and the worst group do not overlap (a conservative test).

Run:  python -m src.evaluation.bias_check --dummy   (dummy data, results/dummy/)
      python -m src.evaluation.bias_check           (real data, results/)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.error_analysis import (  # noqa: E402
    HISTORY_BINS,
    HISTORY_LABELS,
    MIN_ROWS,
    select_model,
)
from src.evaluation.evaluate import get_paths, load_predictions  # noqa: E402
from src.evaluation.metrics import evaluate_with_ci, exam_group_indices  # noqa: E402
from src.evaluation.predictions_io import REPO_ROOT, read_split  # noqa: E402

PAPERS_CLEAN = REPO_ROOT / "data" / "processed" / "papers_clean.csv"
METRICS = ["recall", "precision", "f1", "auc_roc"]
TIER_LABELS = ["small", "medium", "large"]
MIN_EXAMS = 3  # a group needs at least this many exams to count as reliable
GROUPINGS = [
    ("target_type", "Assessment type"),
    ("coverage_tier", "Subject coverage tier (training exams)"),
    ("subject_kind", "Lab vs theory subject"),
    ("target_year", "Target year"),
    ("history", "History size (prior papers)"),
]


def coverage_tiers(train: pd.DataFrame) -> pd.DataFrame:
    """Small / medium / large by the number of TRAINING exams per subject."""
    n_exams = (
        train.drop_duplicates(["subject", "target_year", "target_type"])
        .groupby("subject")
        .size()
    )
    low, high = n_exams.quantile([1 / 3, 2 / 3])
    tier = np.where(n_exams <= low, "small", np.where(n_exams <= high, "medium", "large"))
    return pd.DataFrame(
        {"train_exams": n_exams, "tier": pd.Categorical(tier, categories=TIER_LABELS)}
    )

def lab_lookup() -> pd.Series:
    """Share of each subject's papers flagged is_lab_subject (from papers_clean)."""
    papers = pd.read_csv(PAPERS_CLEAN, usecols=["subject", "is_lab_subject"])
    flag = papers["is_lab_subject"].astype(str).str.lower().eq("true")
    return flag.groupby(papers["subject"]).mean()


def add_groups(test: pd.DataFrame, info: pd.DataFrame, lab_share: pd.Series) -> pd.DataFrame:
    df = test.copy()
    tier = df["subject"].map(info["tier"]).astype(object).fillna("none")
    df["coverage_tier"] = pd.Categorical(tier, categories=TIER_LABELS + ["none"])
    share = df["subject"].map(lab_share)
    kind = np.where(share.isna(), "unknown", np.where(share >= 0.5, "lab", "theory"))
    df["subject_kind"] = pd.Categorical(kind, categories=["theory", "lab", "unknown"])
    df["history"] = pd.cut(df["prior_papers"], HISTORY_BINS, labels=HISTORY_LABELS)
    return df


def group_rows(df: pd.DataFrame, by: str, threshold: float, n_boot: int, seed: int):
    rows = []
    for key, g in df.groupby(by, observed=True, sort=True):
        res = evaluate_with_ci(g, threshold, n_boot=n_boot, seed=seed)
        row = {
            "grouping": by,
            "group": str(key),
            "n_rows": res["n_rows"],
            "n_exams": res["n_exams"],
            "positive_rate": res["positive_rate"],
            "mean_score": float(g["score"].mean()),
            "reliable": res["n_rows"] >= MIN_ROWS and res["n_exams"] >= MIN_EXAMS,
        }
        for m in METRICS:
            row[m] = res[m]
            row[f"{m}_lo"] = res[f"{m}_lo"]
            row[f"{m}_hi"] = res[f"{m}_hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def gap_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Largest gap per grouping and metric, among reliable groups only."""
    rows = []
    for by, t in table.groupby("grouping", sort=False):
        t = t[t["reliable"]]
        for m in METRICS:
            valid = t.dropna(subset=[m])
            if len(valid) < 2:
                continue
            best = valid.loc[valid[m].idxmax()]
            worst = valid.loc[valid[m].idxmin()]
            rows.append(
                {
                    "grouping": by,
                    "metric": m,
                    "best_group": best["group"],
                    "worst_group": worst["group"],
                    "best_value": best[m],
                    "worst_value": worst[m],
                    "gap": best[m] - worst[m],
                    "ci_separated": bool(best[f"{m}_lo"] > worst[f"{m}_hi"]),
                }
            )
    return pd.DataFrame(rows)


def text_availability() -> pd.DataFrame:
    """Readable-text share by assessment type, from the cleaned papers table."""
    p = pd.read_csv(PAPERS_CLEAN, usecols=["assessment_type", "has_text"])
    p["has_text"] = p["has_text"].astype(str).str.lower().eq("true")
    g = p.groupby("assessment_type")["has_text"].agg(n_papers="size", n_readable="sum")
    g["n_unreadable"] = g["n_papers"] - g["n_readable"]
    g["pct_unreadable"] = 100 * g["n_unreadable"] / g["n_papers"]
    return g.reset_index()


def split_composition(data_dir: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        d = read_split(split, data_dir)
        for t, g in d.groupby("target_type"):
            rows.append(
                {
                    "split": split,
                    "target_type": t,
                    "n_rows": len(g),
                    "share_of_rows": len(g) / len(d),
                    "n_exams": len(exam_group_indices(g)),
                    "positive_rate": float(g["label"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_bias(table: pd.DataFrame, path, model: str, threshold: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    width = 0.27
    for ax, (by, title) in zip(axes, GROUPINGS):
        t = table[table["grouping"] == by].reset_index(drop=True)
        x = np.arange(len(t))
        for j, (m, label) in enumerate(
            [("recall", "Recall"), ("precision", "Precision"), ("auc_roc", "AUC-ROC")]
        ):
            vals = t[m].to_numpy(float)
            err = np.vstack(
                [
                    np.nan_to_num(vals - t[f"{m}_lo"].to_numpy(float)),
                    np.nan_to_num(t[f"{m}_hi"].to_numpy(float) - vals),
                ]
            )
            bars = ax.bar(
                x + (j - 1) * width, vals, width, yerr=np.clip(err, 0, None),
                capsize=2, label=label, error_kw={"linewidth": 0.8},
            )
            for rect, ok in zip(bars, t["reliable"]):
                rect.set_alpha(1.0 if ok else 0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{g}\n(n={r}, {e} ex.)" for g, r, e in zip(t["group"], t["n_rows"], t["n_exams"])],
            fontsize=8,
        )
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if by == "target_type":
            ax.legend(loc="lower left", fontsize=8)
    note = axes[5]
    note.axis("off")
    note.text(
        0.0, 0.9,
        f"Bias check, test set\nModel: {model}\nThreshold: {threshold:.3g} (fixed on validation)\n\n"
        "Error bars: 95% cluster bootstrap\n(whole exams resampled).\n\n"
        f"Faded bars: fewer than {MIN_ROWS} rows or\nfewer than {MIN_EXAMS} exams (unreliable).\n\n"
        "n = test rows, ex. = test exams.",
        va="top", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="use dummy predictions")
    parser.add_argument("--model", default=None, help="analyse this learned model")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_paths(args.dummy)
    if args.dummy:
        print("*** DUMMY RUN: fake data, never report these numbers ***")

    preds = load_predictions(paths)
    model, threshold, val_f1 = select_model(preds, args.model)
    print(f"model: {model} | threshold {threshold:.4f} (chosen on val, val F1 {val_f1:.3f})")

    train = read_split("train", paths["data_dir"])
    test = read_split("test", paths["data_dir"]).reset_index(drop=True)
    test["score"] = preds[(model, "test")]["score"].to_numpy()

    info = coverage_tiers(train)
    print("\ncoverage tiers (training exams per subject; tertile cutoffs, equal counts share a tier):")
    for tier in TIER_LABELS:
        sub = info[info["tier"] == tier]["train_exams"]
        span = f"{sub.min()}-{sub.max()} training exams" if len(sub) else "no subjects"
        print(f"  {tier}: {len(sub)} subjects, {span}")
    lab_share = lab_lookup()
    df = add_groups(test, info, lab_share)
    subj = pd.Series(df["subject"].unique())
    share = subj.map(lab_share)
    print("\nlab subjects (>= half of papers flagged):",
          sorted(subj[share >= 0.5]) or "none among the test subjects")
    mixed = sorted(subj[(share > 0) & (share < 1)])
    if mixed:
        print("subjects with MIXED lab flags (classified by majority):", mixed)
    unknown = sorted(subj[share.isna()])
    if unknown:
        print("subjects NOT found in papers_clean.csv (grouped as 'unknown'):", unknown)

    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(
        [group_rows(df, by, threshold, args.n_boot, args.seed) for by, _ in GROUPINGS],
        ignore_index=True,
    )
    path = out_dir / "bias_check.csv"
    table.to_csv(path, index=False)
    show = ["grouping", "group", "n_rows", "n_exams", "recall", "precision", "f1",
            "auc_roc", "reliable"]
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    print(table[show].round(3).to_string(index=False))

    gaps = gap_summary(table)
    path = out_dir / "bias_check_gaps.csv"
    gaps.to_csv(path, index=False)
    print(f"\nwrote {path.relative_to(REPO_ROOT)} (largest gap per grouping and metric)")
    print(gaps.round(3).to_string(index=False))

    availability = text_availability()
    path = out_dir / "bias_data_text_availability.csv"
    availability.to_csv(path, index=False)
    print(f"\nwrote {path.relative_to(REPO_ROOT)} (papers_clean.csv, by assessment type)")
    print(availability.round(1).to_string(index=False))

    composition = split_composition(paths["data_dir"])
    path = out_dir / "bias_split_composition.csv"
    composition.to_csv(path, index=False)
    print(f"\nwrote {path.relative_to(REPO_ROOT)} (assessment types per split)")
    print(composition.round(3).to_string(index=False))

    paths["fig_dir"].mkdir(parents=True, exist_ok=True)
    fig_path = paths["fig_dir"] / "eval_bias_by_group.png"
    plot_bias(table, fig_path, model, threshold)
    print(f"\nfigure written: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()