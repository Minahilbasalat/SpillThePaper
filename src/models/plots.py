"""Report figures for the Phase 3 model section.

Reads results/ only (no training, no test data). Saves PNGs to
report/figures/.

Run:  python -m src.models.plots
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.models.preprocess import ROOT

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"
N_FEATURES = 12


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=1.5))


def plot_architecture(cfg: dict) -> None:
    h1, h2 = cfg["hidden"]
    p = cfg["dropout"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    y, h, w = 1.2, 1.3, 1.35
    boxes = [
        (0.2, f"Input\n{N_FEATURES} features", "#cfe2f3"),
        (2.0, f"Linear\n{N_FEATURES} -> {h1}", "#d9ead3"),
        (3.7, "ReLU", "#fff2cc"),
        (5.0, f"Dropout\np = {p}", "#f4cccc"),
        (6.7, f"Linear\n{h1} -> {h2}", "#d9ead3"),
        (8.4, "ReLU", "#fff2cc"),
        (9.4, f"Dropout\np = {p}", "#f4cccc"),
        (11.0, f"Linear\n{h2} -> 1", "#d9ead3"),
    ]
    widths = [w, w, 0.9, w, w, 0.7, w, w]
    for (x, text, col), bw in zip(boxes, widths):
        _box(ax, x, y, bw, h, text, col)
    for i in range(len(boxes) - 1):
        x_end = boxes[i][0] + widths[i]
        _arrow(ax, x_end + 0.03, boxes[i + 1][0] - 0.03, y + h / 2)
    ax.text(11.7, 0.6, "logit", ha="center", fontsize=9)
    _arrow(ax, 12.4, 12.9, y + h / 2)
    ax.text(12.45, 2.75, "sigmoid", ha="center", fontsize=9)
    ax.text(12.45, 0.6, "P(label = 1)", ha="center", fontsize=8)
    ax.set_title("MLP architecture (loss: BCEWithLogitsLoss; "
                 f"optimizer: {cfg['optimizer'].upper()}, lr = {cfg['lr']})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_architecture.png", dpi=200)
    plt.close(fig)


def plot_curves(hist: pd.DataFrame) -> None:
    best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    for cols, ylabel, name in (
        (("train_loss", "val_loss"), "Loss (BCE)", "model_loss_curves.png"),
        (("train_acc", "val_acc"), "Accuracy", "model_accuracy_curves.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(hist["epoch"], hist[cols[0]], label="train")
        ax.plot(hist["epoch"], hist[cols[1]], label="validation")
        ax.axvline(best_epoch, color="gray", ls="--",
                   label=f"best val-loss epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / name, dpi=200)
        plt.close(fig)


def plot_tuning(tuning: pd.DataFrame, base: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col in zip(axes, ("lr", "dropout")):
        levels = sorted(tuning[col].unique())
        for i, lv in enumerate(levels):
            vals = tuning.loc[tuning[col] == lv, "mean_val_auc"].to_numpy()
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.7)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="black")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(lv) for lv in levels])
        ax.set_xlabel(col)
        ax.axhline(base["logreg"]["val_auc"], color="green", ls="--",
                   label="logistic regression")
        ax.axhline(base["reference_topic_rate_prior_val_auc"], color="red",
                   ls=":", label="raw topic_rate_prior")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean validation AUC-ROC (3 seeds)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("MLP tuning: 40 configurations (black bar = group mean)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_tuning.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RESULTS / "best_config.json").read_text())
    hist = pd.read_csv(RESULTS / "training_history_mlp.csv")
    tuning = pd.read_csv(RESULTS / "tuning_results.csv")
    base = json.loads((RESULTS / "baseline_best.json").read_text())
    plot_architecture(cfg)
    plot_curves(hist)
    plot_tuning(tuning, base)
    print("Saved 4 figures to", FIG_DIR)


if __name__ == "__main__":
    main()