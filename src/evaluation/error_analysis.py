"""Error analysis: where does the best learned model fail on the test set?

The model and its threshold are fixed from the VALIDATION set (best F1). The
test rows are only broken down into groups; nothing is tuned on them.

Run:  python -m src.evaluation.error_analysis --dummy   (dummy data, results/dummy/)
      python -m src.evaluation.error_analysis           (real data, results/)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.evaluate import (  # noqa: E402
    LEARNED,
    choose_threshold,
    get_paths,
    load_predictions,
)
from src.evaluation.metrics import compute_metrics, exam_group_indices  # noqa: E402
from src.evaluation.predictions_io import REPO_ROOT, read_split  # noqa: E402

MIN_ROWS = 20  # groups with fewer test rows are flagged as unreliable
HISTORY_BINS = [-np.inf, 4, 9, 19, np.inf]
HISTORY_LABELS = ["0-4", "5-9", "10-19", "20+"]
FAMILIARITY_BINS = [-np.inf, 0, 2, 5, np.inf]
FAMILIARITY_LABELS = ["0 (new)", "1-2", "3-5", "6+"]
TOPIC_KEYS = ["subject", "topic_id", "topic_label"]
SHOW = ["n_rows", "n_exams", "positive_rate", "mean_score", "precision",
        "recall", "f1", "auc_roc", "reliable"]


def select_model(preds: dict, forced: str | None) -> tuple[str, float, float]:
    """Best learned model by validation F1. Returns (model, threshold, val_f1)."""
    candidates = [m for m in LEARNED if (m, "val") in preds and (m, "test") in preds]
    if forced:
        if forced not in candidates:
            raise SystemExit(f"model '{forced}' has no val+test predictions")
        candidates = [forced]
    if not candidates:
        raise SystemExit(
            "No learned-model predictions found (logreg / random_forest / mlp). "
            "Use --dummy for now, or wait until Khadija's predictions are on main."
        )
    scored = []
    for model in candidates:
        val = preds[(model, "val")]
        threshold, _ = choose_threshold(model, val)
        f1 = compute_metrics(val["label"], val["score"], threshold)["f1"]
        scored.append((f1, model, threshold))
    f1, model, threshold = max(scored)
    return model, threshold, f1


def group_table(df: pd.DataFrame, by: str, threshold: float) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True, sort=True):
        m = compute_metrics(g["label"], g["score"], threshold)
        rows.append(
            {
                by: key,
                "n_rows": len(g),
                "n_exams": len(exam_group_indices(g)),
                "positive_rate": m["positive_rate"],
                "mean_score": float(g["score"].mean()),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc_roc": m["auc_roc"],
                "fn": m["fn"],
                "fp": m["fp"],
                "reliable": len(g) >= MIN_ROWS,
            }
        )
    return pd.DataFrame(rows)


def top_errors(df: pd.DataFrame, threshold: float, kind: str, k: int) -> pd.DataFrame:
    """Topics with the most false negatives ('fn') or false positives ('fp')."""
    predicted = (df["score"] >= threshold).astype(int)
    if kind == "fn":
        errors = df[(predicted == 0) & (df["label"] == 1)]
    else:
        errors = df[(predicted == 1) & (df["label"] == 0)]
    counts = errors.groupby(TOPIC_KEYS, dropna=False).size().rename(f"n_{kind}")
    totals = df.groupby(TOPIC_KEYS, dropna=False).size().rename("n_test_rows")
    out = pd.concat([counts, totals], axis=1, join="inner").reset_index()
    return out.sort_values([f"n_{kind}", "n_test_rows"], ascending=False).head(k)


def _bars(ax, table: pd.DataFrame, key: str, title: str, rotate: bool = False) -> None:
    x = np.arange(len(table))
    width = 0.27
    for j, (col, label) in enumerate(
        [("recall", "Recall"), ("precision", "Precision"), ("auc_roc", "AUC-ROC")]
    ):
        bars = ax.bar(x + (j - 1) * width, table[col].to_numpy(float), width, label=label)
        for rect, ok in zip(bars, table["reliable"]):
            rect.set_alpha(1.0 if ok else 0.35)
    names = [str(k)[:22] for k in table[key]]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(n={r})" for n, r in zip(names, table["n_rows"])],
        rotation=45 if rotate else 0,
        ha="right" if rotate else "center",
        fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def plot_errors(tables: dict, path, model: str, threshold: float) -> None:
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.2)
    panels = [
        ("target_type", "Assessment type", gs[0, 0], False),
        ("target_year", "Target year", gs[0, 1], False),
        ("history", "History size (prior papers)", gs[1, 0], False),
        ("familiarity", "Topic familiarity (times seen before)", gs[1, 1], False),
        ("subject", f"Subject (at least {MIN_ROWS} test rows)", gs[2, :], True),
    ]
    first = True
    for key, title, spec, rotate in panels:
        table = tables[key]
        ax = fig.add_subplot(spec)
        if table.empty:
            ax.set_title(f"{title}: no group with enough rows", fontsize=10)
            ax.axis("off")
            continue
        _bars(ax, table, key, title, rotate)
        if first:
            ax.legend(loc="lower left", fontsize=8)
            first = False
    fig.suptitle(
        f"Errors by group, test set: {model} (threshold {threshold:.3g}); "
        f"faded bars = fewer than {MIN_ROWS} rows",
        fontsize=11,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="use dummy predictions")
    parser.add_argument("--model", default=None, help="analyse this learned model")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    paths = get_paths(args.dummy)
    if args.dummy:
        print("*** DUMMY RUN: fake data, never report these numbers ***")

    preds = load_predictions(paths)
    model, threshold, val_f1 = select_model(preds, args.model)
    print(f"model: {model} | threshold {threshold:.4f} (chosen on val, val F1 {val_f1:.3f})")

    df = read_split("test", paths["data_dir"]).reset_index(drop=True)
    df["score"] = preds[(model, "test")]["score"].to_numpy()  # same row order (validated)
    df["history"] = pd.cut(df["prior_papers"], HISTORY_BINS, labels=HISTORY_LABELS)
    df["familiarity"] = pd.cut(
        df["topic_count_prior"].fillna(0), FAMILIARITY_BINS, labels=FAMILIARITY_LABELS
    )

    overall = compute_metrics(df["label"], df["score"], threshold)
    print(
        f"overall test: n={overall['n_rows']} precision {overall['precision']:.3f} "
        f"recall {overall['recall']:.3f} f1 {overall['f1']:.3f} auc {overall['auc_roc']:.3f}"
    )

    out_dir = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key in ["subject", "target_type", "target_year", "history", "familiarity"]:
        table = group_table(df, key, threshold)
        if key == "subject":
            kept = table[table["n_rows"] >= MIN_ROWS]
            print(
                f"\nsubjects: {len(kept)} with >= {MIN_ROWS} test rows "
                f"({len(table) - len(kept)} excluded as too small)"
            )
            table = kept.reset_index(drop=True)
        tables[key] = table
        name = "history" if key == "history" else key
        path = out_dir / f"error_analysis_{name}.csv"
        table.to_csv(path, index=False)
        print(f"\n[{key}] wrote {path.relative_to(REPO_ROOT)}")
        if not table.empty:
            print(table[[key] + SHOW].round(3).to_string(index=False))

    for kind, label in [("fn", "false_negatives"), ("fp", "false_positives")]:
        errors = top_errors(df, threshold, kind, args.top_k)
        path = out_dir / f"error_analysis_{label}.csv"
        errors.to_csv(path, index=False)
        print(f"\n[{label}] wrote {path.relative_to(REPO_ROOT)} (top {args.top_k} topics)")
        print(errors.to_string(index=False, max_colwidth=45))

    paths["fig_dir"].mkdir(parents=True, exist_ok=True)
    fig_path = paths["fig_dir"] / "eval_errors_by_group.png"
    plot_errors(tables, fig_path, model, threshold)
    print(f"\nfigure written: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()