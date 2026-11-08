"""Report figures: ROC, precision-recall, calibration and confusion matrices.

Called by evaluate.py. Dummy runs write to results/dummy/figures only, so
they can never overwrite the real figures in report/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

LEARNED = ["logreg", "random_forest", "mlp"]
RULES = ["rule_count_ge5", "rule_last_year", "topic_rate_only"]
DPI = 200


def _line_style(model: str) -> str:
    return "-" if model in LEARNED else "--"


def _ops(rows: list[dict]) -> dict:
    return {r["model"]: r for r in rows}


def _models(preds: dict, ops: dict, split: str, names: list[str]) -> list[str]:
    return [m for m in names if m in ops and (m, split) in preds]


def pick_best(val_rows: list[dict], candidates: list[str]):
    """Model with the highest VALIDATION F1 among the candidates (or None)."""
    scored = [(r["f1"], r["model"]) for r in val_rows if r["model"] in candidates]
    if not scored:
        return None
    return max(scored, key=lambda t: t[0])[1]


def plot_roc(preds: dict, test_rows: list[dict], path: Path) -> None:
    ops = _ops(test_rows)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for model in _models(preds, ops, "test", LEARNED + RULES):
        d = preds[(model, "test")]
        fpr, tpr, _ = roc_curve(d["label"], d["score"])
        r = ops[model]
        (line,) = ax.plot(
            fpr, tpr, _line_style(model), linewidth=1.6,
            label=f"{model} (AUC {r['auc_roc']:.3f})",
        )
        ax.plot(
            r["fp"] / max(r["fp"] + r["tn"], 1),
            r["tp"] / max(r["tp"] + r["fn"], 1),
            "o", color=line.get_color(),
        )
    ax.plot([0, 1], [0, 1], ":", color="grey", label="chance / all-positive")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves, test set (dots = chosen threshold)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_pr(preds: dict, test_rows: list[dict], path: Path) -> None:
    ops = _ops(test_rows)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for model in _models(preds, ops, "test", LEARNED + RULES):
        d = preds[(model, "test")]
        precision, recall, _ = precision_recall_curve(d["label"], d["score"])
        r = ops[model]
        (line,) = ax.plot(
            recall, precision, _line_style(model), linewidth=1.6,
            label=f"{model} (PR-AUC {r['pr_auc']:.3f})",
        )
        ax.plot(r["recall"], r["precision"], "o", color=line.get_color())
    pos_rate = next(iter(ops.values()))["positive_rate"]
    ax.axhline(
        pos_rate, linestyle=":", color="grey",
        label=f"all-positive baseline (precision {pos_rate:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves, test set (dots = chosen threshold)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_calibration(
    preds: dict, test_rows: list[dict], path: Path, n_bins: int = 8
) -> None:
    ops = _ops(test_rows)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.plot([0, 1], [0, 1], ":", color="grey", label="perfect calibration")
    for model in _models(preds, ops, "test", LEARNED + ["topic_rate_only"]):
        d = preds[(model, "test")]
        bins = pd.qcut(d["score"], q=n_bins, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(
            predicted=("score", "mean"), observed=("label", "mean")
        )
        ax.plot(g["predicted"], g["observed"], "o-", linewidth=1.4, label=model)
    pos_rate = next(iter(ops.values()))["positive_rate"]
    ax.axhline(
        pos_rate, linestyle="--", color="lightgrey",
        label=f"test positive rate ({pos_rate:.2f})",
    )
    ax.set_xlabel("Mean predicted probability (equal-count bins)")
    ax.set_ylabel("Observed fraction of topics that appeared")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram, test set")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_confusion(test_rows: list[dict], picks: dict, path: Path) -> bool:
    ops = _ops(test_rows)
    picks = {title: m for title, m in picks.items() if m in ops}
    if not picks:
        return False
    fig, axes = plt.subplots(
        1, len(picks), figsize=(4.6 * len(picks), 4.3), squeeze=False
    )
    for ax, (title, model) in zip(axes[0], picks.items()):
        r = ops[model]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
                ax.text(
                    j, i, f"{cm[i, j]}\n({pct:.0f}% of actual {i})",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["predicted 0", "predicted 1"])
        ax.set_yticklabels(["actual 0", "actual 1"])
        ax.set_title(f"{title}\n{model}, threshold {r['threshold']:.3g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def make_figures(preds: dict, rows: dict, fig_dir: Path) -> list[Path]:
    """Write the four report figures; returns the paths that were written."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    test_rows, val_rows = rows["test"], rows["val"]
    written = []

    for name, func in [
        ("eval_roc_test.png", plot_roc),
        ("eval_pr_test.png", plot_pr),
        ("eval_calibration_test.png", plot_calibration),
    ]:
        func(preds, test_rows, fig_dir / name)
        written.append(fig_dir / name)

    best_learned = pick_best(val_rows, LEARNED)
    best_rule = pick_best(val_rows, RULES)
    print(f"\nconfusion matrices: best learned = {best_learned}, "
          f"best rule = {best_rule} (both chosen by validation F1)")
    picks = {"Best learned model": best_learned, "Best rule": best_rule}
    if plot_confusion(test_rows, picks, fig_dir / "eval_confusion_test.png"):
        written.append(fig_dir / "eval_confusion_test.png")
    return written