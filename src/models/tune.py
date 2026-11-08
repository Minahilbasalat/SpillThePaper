"""Random search for the MLP, selected by MEAN VALIDATION AUC-ROC.

40 configurations sampled (fixed seed) from the search space, each trained
with 3 seeds. The test split is never loaded here.

Outputs:
  results/tuning_results.csv   every configuration with mean and std
  results/best_config.json     the selected configuration

Run:  python -m src.models.tune
"""
from __future__ import annotations

import itertools
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from src.models.preprocess import ROOT, fit_preprocessor, load_split, split_to_xy
from src.models.train_mlp import run_training

SAMPLE_SEED = 42
N_CONFIGS = 40
SEEDS = (42, 43, 44)

SPACE = {
    "lr": [1e-3, 3e-3, 1e-2],
    "batch_size": [32, 64, 128],
    "dropout": [0.0, 0.2, 0.4],
    "hidden": [(16, 8), (32, 16), (64, 32)],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "optimizer": ["adam", "sgd"],
}

RESULTS_PATH = ROOT / "results" / "tuning_results.csv"
BEST_PATH = ROOT / "results" / "best_config.json"


def sample_configs(n: int = N_CONFIGS, seed: int = SAMPLE_SEED) -> list[dict]:
    keys = list(SPACE)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*SPACE.values())]
    chosen = random.Random(seed).sample(grid, n)  # distinct, reproducible
    for cfg in chosen:
        cfg["hidden"] = list(cfg["hidden"])
        cfg["pos_weight"] = False
    return chosen


def main() -> None:
    torch.set_num_threads(1)
    train, val = load_split("train"), load_split("val")
    pre = fit_preprocessor(train)  # train only
    Xtr, ytr = split_to_xy(train, pre)
    Xva, yva = split_to_xy(val, pre)

    configs = sample_configs()
    rows, start = [], time.time()
    for i, cfg in enumerate(configs, 1):
        aucs, epochs, losses = [], [], []
        for seed in SEEDS:
            res = run_training(cfg, Xtr, ytr, Xva, yva, seed=seed)
            aucs.append(res["best_val_auc"])
            epochs.append(res["best_epoch"])
            losses.append(res["best_val_loss"])
        row = {
            "config_id": i,
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "dropout": cfg["dropout"],
            "hidden": str(tuple(cfg["hidden"])),
            "weight_decay": cfg["weight_decay"],
            "optimizer": cfg["optimizer"],
            "mean_val_auc": float(np.mean(aucs)),
            "std_val_auc": float(np.std(aucs, ddof=0)),
            "mean_val_loss": float(np.mean(losses)),
            "mean_best_epoch": float(np.mean(epochs)),
            "seed_aucs": ";".join(f"{a:.4f}" for a in aucs),
        }
        rows.append(row)
        print(f"[{i:2d}/{len(configs)}] mean val AUC {row['mean_val_auc']:.3f} "
              f"(std {row['std_val_auc']:.3f}) {cfg}  "
              f"[{time.time() - start:.0f}s]", flush=True)

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    best_idx = int(table["mean_val_auc"].idxmax())
    best_cfg = configs[best_idx]
    BEST_PATH.write_text(json.dumps(best_cfg, indent=2))

    print("\nTop 5 configurations by mean validation AUC:")
    print(table.sort_values("mean_val_auc", ascending=False).head(5)
          .drop(columns="seed_aucs").to_string(index=False))
    print(f"\nSelected config_id {best_idx + 1}: {best_cfg}")
    print(f"Mean val AUC {table.loc[best_idx, 'mean_val_auc']:.3f} "
          f"+/- {table.loc[best_idx, 'std_val_auc']:.3f}")
    print(f"Range over all configurations: {table['mean_val_auc'].min():.3f} "
          f"to {table['mean_val_auc'].max():.3f}")
    print("Saved results/tuning_results.csv and results/best_config.json")


if __name__ == "__main__":
    main()