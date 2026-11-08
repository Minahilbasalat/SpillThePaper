# Phase 4: Evaluation (Iman Abid)

Code: `src/evaluation/`. Tests: `tests/test_metrics.py`. This document explains
how models are validated and scored, and how to reproduce every evaluation
output. All numbers in the report come from real runs of these scripts.

## Validation strategy

- **Task:** one row = (subject, topic, target exam). `label = 1` if the topic
  appears in at least one paper of that subject, year and assessment type.
  Features use only papers from strictly earlier years.
- **Split by time, not at random.** Train = target years 2014-2022
  (1,421 rows, 166 target exams), validation = 2023 (301 rows, 36 exams),
  test = 2024-2026 (568 rows, 67 exams). A random K-fold would mix future
  exams into training and hide the drift in the positive rate
  (47.3% train, 39.2% val, 34.7% test). The time split matches the real use:
  predict an upcoming exam from the past.
- **Validation** is the only set used to choose models, hyperparameters and
  decision thresholds. **The test set is evaluated once, at the end.**
- **Decision threshold:** chosen once on validation (maximum F1; ties go to the
  smallest threshold) and applied unchanged to test. Rule baselines
  `rule_count_ge5` (count >= 5), `rule_last_year` (appeared last year) and
  `all_positive` keep their fixed rule threshold.
- **Confidence intervals:** 95% percentile intervals from a cluster bootstrap.
  Rows of the same target exam (subject + target_year + target_type) are
  correlated, so whole exams are resampled with replacement
  (1,000 resamples, seed 42).
- Validation has only 36 target exams and test only 67, so small metric
  differences are noise. Judge differences by the confidence intervals.

## Models compared

- Learned models from Khadija: `logreg`, `random_forest`, `mlp`.
- Non-learned baselines (`src/evaluation/baselines.py`):
  `rule_count_ge5` (score = topic_count_prior), `rule_last_year`
  (score = appeared_last_year plus a tiny tie-breaker),
  `topic_rate_only` (score = topic_rate_prior, threshold chosen on validation),
  `all_positive` (predicts 1 for every row).

## Predictions format (shared contract)

`results/predictions/<model>_<split>.csv` with columns
`row_idx, subject, topic_id, target_year, target_type, label, score, model, split`.
`row_idx` is the 0-based row of `data/processed/model/<split>.csv`; `score` is
the predicted probability of label 1. Rows keep the order of the split file,
with none missing (val 301, test 568). `src/evaluation/predictions_io.py`
validates every file and stops with an error if the contract is broken.

## Metrics

With TP, FP, FN, TN counted at the chosen threshold:

- **Precision** = TP / (TP + FP): of the topics we tell a student to study, the
  share that really appear.
- **Recall** = TP / (TP + FN): of the topics that appear, the share we found.
- **F1** = harmonic mean of precision and recall.
- **AUC-ROC:** probability that a random appearing topic is scored above a
  random non-appearing topic; it does not depend on a threshold.
- **PR-AUC** (average precision): area under the precision-recall curve; more
  informative when the positive rate drifts.
- **Brier score:** mean squared error of the probability (lower is better);
  checks calibration. It is not reported for rule scores that are not
  probabilities (counts) or for the constant `all_positive` score.

Accuracy alone hides the trade-off between missing a topic that appears and
studying topics that do not, so it is not the headline metric.

## Reproduction commands

Run from the repository root with the project's virtual environment active.

```powershell
python -m src.evaluation.baselines
python -m src.evaluation.evaluate
python -m src.evaluation.error_analysis
python -m src.evaluation.bias_check
python -m src.evaluation.contribution_table
```

- `baselines` writes the four rule baselines to `results/predictions/`.
- `evaluate` writes `results/metrics_val.csv`, `results/metrics_test.csv` and the
  figures `report/figures/eval_roc_test.png`, `eval_pr_test.png`,
  `eval_confusion_test.png`, `eval_calibration_test.png`.
- `error_analysis` writes `results/error_analysis_*.csv` and
  `report/figures/eval_errors_by_group.png` (best learned model chosen by
  validation F1).
- `bias_check` writes `results/bias_check.csv`, `bias_check_gaps.csv`, two
  data-level tables and `report/figures/eval_bias_by_group.png`.
- `contribution_table` runs after all branches are merged to `main` and writes
  `results/contribution_table.csv` plus a LaTeX fragment.

Add `--dummy` to `baselines`, `evaluate`, `error_analysis` and `bias_check` to
develop on random dummy predictions. Dummy runs write only to `results/dummy/`
(never committed), and their numbers must never be reported.

Run the real evaluation only when all three learned models have predictions in
`results/predictions/`. Tests: `python -m pytest tests/test_metrics.py -q`.

## Known limitations

- Topics come from noisy clustering (median silhouette 0.097); some clusters are
  instruction text.
- The label depends on how many papers the target group contains.
- The positive rate drifts down over time, so probabilities may be miscalibrated
  on newer exams.
- Only 28 of 105 subjects are covered, and there are very few cold-start rows.
- A simple frequency feature is already a strong baseline; "no clear gain" of a
  learned model over it is an acceptable, honest finding.