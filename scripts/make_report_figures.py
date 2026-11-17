"""Writes the three figures of the final report from the project's own data and models.

    python scripts/make_report_figures.py

Output (report/figures/): fig_predicted_topics.png (sample topic forecast of the app),
fig_ocr_extraction.png (text extraction and OCR outcome), fig_question_features.png (question
types and difficulty). Needs the runtime bundle (models) for the first figure. Run it again
after `python scripts/rebuild_all.py` so the figures follow the data."""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SUBJECT, EXAM, YEAR = "Artificial Intelligence", "final", 2026


def forecast():
    from src.inference.predictor import Predictor
    p = Predictor()
    r = p.predict(SUBJECT, EXAM, YEAR)
    colors = {"Likely": "#2e7d32", "Possible": "#f9a825", "Unlikely": "#9e9e9e"}
    d = r.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh([textwrap.shorten(x, 38, placeholder="...") for x in d.topic_label], d.probability,
            color=[colors[b] for b in d.band])
    ax.axvline(r.attrs["threshold"], ls="--", c="k", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability that the topic appears")
    ax.set_title(f"{SUBJECT}, {EXAM.title()} {YEAR}: topic forecast (logistic regression)", fontsize=10)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()]
              + [plt.Line2D([], [], ls="--", c="k", label="threshold")], loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_predicted_topics.png", dpi=200)
    plt.close()


def ocr():
    e = pd.read_csv(ROOT / "data/processed/extraction_results.csv")
    o = pd.read_csv(ROOT / "data/processed/ocr_results.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    m = e.extraction_method.value_counts()
    axs[0].bar(m.index, m.values, color="#4c78a8")
    axs[0].set_title("Text extraction method (files)", fontsize=10)
    axs[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(m.values):
        axs[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    s = o.status.value_counts()
    vals = [s.get("ok", 0), s.get("rejected_low_quality", 0), s.get("error", 0)]
    axs[1].bar(["accepted", "rejected\n(low quality)", "error"], vals, color=["#2e7d32", "#c62828", "#9e9e9e"])
    axs[1].set_title(f"OCR outcome of {len(o):,} flagged scans", fontsize=10)
    for i, v in enumerate(vals):
        axs[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_ocr_extraction.png", dpi=200)
    plt.close()


def question_features():
    q = pd.read_csv(ROOT / "data/interim/question_table.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    t = q.qtype.value_counts().iloc[::-1]
    axs[0].barh(t.index, t.values, color="#4c78a8")
    axs[0].set_title(f"Question types ({len(q):,} questions)", fontsize=10)
    for i, v in enumerate(t.values):
        axs[0].text(v, i, f" {v}", va="center", fontsize=7)
    d = q.difficulty.value_counts().reindex(["easy", "medium", "hard"])
    axs[1].bar(d.index, d.values, color=["#81c784", "#ffd54f", "#e57373"])
    axs[1].set_title("Difficulty proxy", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_question_features.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    for fn in (ocr, question_features, forecast):
        try:
            fn()
            print("ok:", fn.__name__)
        except Exception as ex:          # e.g. models missing: say so, keep going
            print("skipped", fn.__name__, "-", ex)
