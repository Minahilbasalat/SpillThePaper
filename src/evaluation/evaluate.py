"""Evaluate every model: threshold on VAL, metrics + 95% CIs on VAL and TEST.

Rules:
- The decision threshold is chosen ONCE on the validation set (maximum F1)
  and applied unchanged to the test set. Rule baselines with a fixed rule
  (rule_count_ge5, rule_last_year, all_positive) keep their fixed threshold.
- Confidence intervals come from a cluster bootstrap over target exams.
- Run the real evaluation (without --dummy) only when all predictions exist,
  because the test set is evaluated exactly once at the end.

Run:  python -m src.evaluation.evaluate --dummy   (dummy data, results/dummy/)
      python -m src.evaluation.evaluate           (real data, results/)
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.schema import DUMMY_DATA_DIR, MODEL_DATA_DIR
from src.evaluation.baselines import BASELINES, FIXED_THRESHOLDS, UNIT_RANGE
from src.evaluation.metrics import METRIC_NAMES, best_f1_threshold, evaluate_with_ci
from src.evaluation.predictions_io import REPO_ROOT, read_split, validate_predictions

LEARNED = ["logreg", "random_forest", "mlp"]
ALL_MODELS = LEARNED + BASELINES
SPLITS = ["val", "test"]


def get_paths(dummy: bool) -> dict:
    """Where to read data/predictions and where to write outputs."""
    if dummy:
        results = REPO_ROOT / "results" / "dummy"
        return {
            "data_dir": DUMMY_DATA_DIR,
            "pred_dir": results / "predictions",
            "out_dir": results,
            "fig_dir": results / "figures",
        }
    return {
        "data_dir": MODEL_DATA_DIR,
        "pred_dir": REPO_ROOT / "results" / "predictions",
        "out_dir": REPO_ROOT / "results",
        "fig_dir": REPO_ROOT / "report" / "figures",
    }


def load_predictions(paths: dict) -> dict:
    """{(model, split): validated predictions}. Missing learned models are skipped."""
    split_dfs = {s: read_split(s, paths["data_dir"]) for s in SPLITS}
    preds = {}
    for model in ALL_MODELS:
        for split in SPLITS:
            path = paths["pred_dir"] / f"{model}_{split}.csv"
            if not path.exists():
                if model in LEARNED:
                    print(f"  skipping {model}_{split}: file not found")
                    continue
                raise FileNotFoundError(
                    f"{path} missing - run python -m src.evaluation.baselines first"
                )
            pred = pd.read_csv(path)
            unit_range = model in LEARNED or model in UNIT_RANGE
            validate_predictions(
                pred, split_dfs[split], model, split, require_unit_range=unit_range
            )
            preds[(model, split)] = pred
    return preds


def choose_threshold(model: str, val_pred: pd.DataFrame) -> tuple[float, str]:
    """Fixed rule threshold, or the best-F1 threshold on the validation set."""
    if model in FIXED_THRESHOLDS:
        return FIXED_THRESHOLDS[model], "fixed_rule"
    threshold, _ = best_f1_threshold(val_pred["label"], val_pred["score"])
    return threshold, "val_best_f1"


def evaluate_all(preds: dict, n_boot: int, seed: int) -> dict:
    """Metric rows per split, one row per model (threshold from VAL only)."""
    rows = {"val": [], "test": []}
    models = [m for m in ALL_MODELS if (m, "val") in preds and (m, "test") in preds]
    for model in models:
        threshold, source = choose_threshold(model, preds[(model, "val")])
        for split in SPLITS:
            res = evaluate_with_ci(
                preds[(model, split)], threshold, n_boot=n_boot, seed=seed
            )
            if model == "all_positive":  # constant score: Brier is meaningless
                for suffix in ["", "_lo", "_hi"]:
                    res[f"brier{suffix}"] = float("nan")
            res.update({"model": model, "split": split, "threshold_source": source})
            rows[split].append(res)
    return rows


def to_table(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "model",
        "split",
        "threshold",
        "threshold_source",
        "n_rows",
        "n_exams",
        "positive_rate",
    ]
    for name in METRIC_NAMES:
        cols += [name, f"{name}_lo", f"{name}_hi"]
    cols += ["tp", "fp", "fn", "tn"]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="use dummy predictions")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_paths(args.dummy)
    if args.dummy:
        print("*** DUMMY RUN: fake data, never report these numbers ***")

    preds = load_predictions(paths)
    rows = evaluate_all(preds, n_boot=args.n_boot, seed=args.seed)

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    show = ["model", "threshold", "precision", "recall", "f1", "auc_roc",
            "auc_roc_lo", "auc_roc_hi"]
    for split in SPLITS:
        table = to_table(rows[split])
        path = paths["out_dir"] / f"metrics_{split}.csv"
        table.to_csv(path, index=False)
        print(f"\n[{split}] wrote {path.relative_to(REPO_ROOT)}")
        print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()