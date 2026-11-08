# Phase 3 models (Khadija Rao)

Task: predict whether a topic appears in an upcoming exam (label 1/0).
Data: `data/processed/model/{train,val,test}.csv`, split by time
(train 2014-2022, val 2023, test 2024-2026). Hyperparameters are chosen on
validation AUC-ROC only. The test split is used once, to export predictions.

## Preprocessing (`src/models/preprocess.py`)

Fitted on the train split only:

- `years_since_last` NaN (topic never seen) -> train max + 1 (= 10.0),
  plus a `never_seen` indicator column
- `target_type` one-hot encoded with train categories
- numeric features standardised with train mean/std
- identifier columns (subject, topic_id, topic_label) are not features

Result: 12 features (8 numeric + never_seen + 3 one-hot).
Tests: `tests/test_preprocess.py` (train-only statistics, no NaN after transform).

## Models

| Model                                           | File                                | Selected settings                          | Val AUC-ROC                              |
| ----------------------------------------------- | ----------------------------------- | ------------------------------------------ | ---------------------------------------- |
| Raw `topic_rate_prior` (reference, no learning) | -                                   | -                                          | 0.820                                    |
| Logistic regression (Scikit-Learn)              | `baselines.py`                      | C = 0.01                                   | 0.847                                    |
| Random forest (Scikit-Learn)                    | `baselines.py`                      | 300 trees, max_depth 5, min_samples_leaf 5 | 0.827                                    |
| MLP (PyTorch)                                   | `mlp.py`, `train_mlp.py`, `tune.py` | see below                                  | 0.838 (seed 42), 0.842 (mean of 3 seeds) |

Differences between models are small. The validation set has only 36 target
exams, so they should be read as noise-level.

### MLP

Linear(12, 16) -> ReLU -> Dropout(0.4) -> Linear(16, 8) -> ReLU -> Dropout(0.4)
-> Linear(8, 1) (logit). 353 parameters.
Loss: BCEWithLogitsLoss (pos_weight off: train positives are 47.3%).
Optimizer: Adam, lr 0.01, weight decay 0, batch size 64.
Early stopping on validation loss (patience 15, max 200 epochs), best weights
restored. Final run: best epoch 9, stopped at epoch 24.
Tuning: random search, 40 configurations x 3 seeds (42, 43, 44), selected by
mean validation AUC-ROC. All results: `results/tuning_results.csv`;
selected configuration: `results/best_config.json`.

## Predictions

`results/predictions/<model>_<split>.csv` for logreg, random_forest and mlp
(val 301 rows, test 568 rows). Columns: row_idx, subject, topic_id,
target_year, target_type, label, score, model, split.

## Reproduce

From the project root, with the virtual environment active:

    python -m src.models.baselines
    python -m src.models.tune
    python -m src.models.train_mlp
    python -m src.models.export_predictions
    python -m src.models.plots

Tested with Python 3.14.7, torch 2.15.0 (CPU), scikit-learn 1.9.1.
Run `python -m pytest tests/test_preprocess.py -q` for the preprocessing tests.

## Known limitations

Small data, noisy topic labels, a small validation set, and tuning noise
between seeds. The positive rate drifts down over time (47% -> 39% -> 35%),
so probabilities may be miscalibrated on newer exams.

Model weights: <Google Drive link>
