"""Figure for the report: rows per target year (coloured by split) and positive rate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.data.schema import MODEL_DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures" / "dataset_by_year.png"
COLORS = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}


def main():
    frames = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(ROOT / MODEL_DATA_DIR / f"{split}.csv")
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby(["target_year", "split"])
           .agg(rows=("label", "size"), pos=("label", "mean")).reset_index())

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(g["target_year"], g["rows"], color=[COLORS[s] for s in g["split"]])
    ax.set_xlabel("Target exam year")
    ax.set_ylabel("Rows (subject, topic, exam)")
    ax2 = ax.twinx()
    ax2.plot(g["target_year"], g["pos"], "k-o", ms=3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Positive rate (black line)")
    ax.legend(handles=[Patch(color=c, label=s) for s, c in COLORS.items()],
              loc="upper left", title="Split")
    ax.set_title("Dataset size and positive rate by target year")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()