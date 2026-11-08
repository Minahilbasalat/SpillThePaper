"""Train the PyTorch MLP with mini-batches and early stopping.

Early stopping monitors VALIDATION LOSS (patience 15, max 200 epochs) and
restores the best weights. The test split is never loaded here.

Outputs:
  results/training_history_mlp.csv   per-epoch loss / accuracy / val AUC
  models/mlp_best.pt                 best weights (git-ignored)

Run:  python -m src.models.train_mlp
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.models.mlp import MLP, make_loss, make_optimizer
from src.models.preprocess import (
    ROOT,
    fit_preprocessor,
    load_split,
    split_to_xy,
)

SEED = 42
MAX_EPOCHS = 200
PATIENCE = 15
HISTORY_PATH = ROOT / "results" / "training_history_mlp.csv"
WEIGHTS_PATH = ROOT / "models" / "mlp_best.pt"
BEST_CONFIG_PATH = ROOT / "results" / "best_config.json"

DEFAULT_CONFIG = {
    "lr": 3e-3,
    "batch_size": 64,
    "dropout": 0.2,
    "hidden": [32, 16],
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "pos_weight": False,
}


def _eval(model, loss_fn, X: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = float(loss_fn(logits, y))
        prob = torch.sigmoid(logits).numpy()
        acc = float(((prob >= 0.5) == y.numpy().astype(bool)).mean())
    return loss, acc, prob


def run_training(config, Xtr, ytr, Xva, yva, seed=SEED,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train one model. Returns a dict with history, best epoch, best state."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    h1, h2 = config["hidden"]
    model = MLP(Xtr.shape[1], h1, h2, config["dropout"])
    loss_fn = make_loss(ytr, config.get("pos_weight", False))
    opt = make_optimizer(model.parameters(), config["optimizer"],
                         config["lr"], config["weight_decay"])

    Xt = torch.tensor(np.array(Xtr), dtype=torch.float32)
    yt = torch.tensor(np.array(ytr), dtype=torch.float32)
    Xv = torch.tensor(np.array(Xva), dtype=torch.float32)
    yv = torch.tensor(np.array(yva), dtype=torch.float32)

    history, best = [], {"val_loss": float("inf"), "epoch": 0, "state": None}
    bad_epochs = 0
    n, bs = len(Xt), int(config["batch_size"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n)
        running, seen = 0.0, 0
        for start in range(0, n, bs):
            idx = torch.as_tensor(order[start:start + bs])
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        train_loss_batches = running / seen

        # evaluation mode (dropout off) on full train and val sets
        tr_loss, tr_acc, _ = _eval(model, loss_fn, Xt, yt)
        va_loss, va_acc, va_prob = _eval(model, loss_fn, Xv, yv)
        va_auc = float(roc_auc_score(yva, va_prob))
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_loss_running": train_loss_batches,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "val_auc": va_auc,
        })

        if va_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": va_loss, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best["state"])  # restore best weights
    _, _, best_prob = _eval(model, loss_fn, Xv, yv)
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_auc": float(roc_auc_score(yva, best_prob)),
        "epochs_run": len(history),
        "state": best["state"],
    }


def load_config() -> dict:
    if BEST_CONFIG_PATH.exists():
        print(f"Using tuned config from {BEST_CONFIG_PATH.name}")
        return json.loads(BEST_CONFIG_PATH.read_text())
    print("results/best_config.json not found: using DEFAULT_CONFIG")
    return dict(DEFAULT_CONFIG)


def main() -> None:
    train, val = load_split("train"), load_split("val")
    pre = fit_preprocessor(train)  # train only
    Xtr, ytr = split_to_xy(train, pre)
    Xva, yva = split_to_xy(val, pre)

    config = load_config()
    print("Config:", config)
    res = run_training(config, Xtr, ytr, Xva, yva, seed=SEED)

    HISTORY_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(res["history"]).to_csv(HISTORY_PATH, index=False)
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": res["state"], "config": config,
                "in_features": int(Xtr.shape[1])}, WEIGHTS_PATH)

    print(f"Epochs run: {res['epochs_run']}, best epoch: {res['best_epoch']}")
    print(f"Best val loss: {res['best_val_loss']:.4f}, "
          f"val AUC at best epoch: {res['best_val_auc']:.3f}")
    print(f"Saved {HISTORY_PATH.name} and {WEIGHTS_PATH.name}")


if __name__ == "__main__":
    main()