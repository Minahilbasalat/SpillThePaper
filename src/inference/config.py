from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODEL_DATA = ROOT / "data" / "processed" / "model"
PRED_DIR = ROOT / "results" / "predictions"

# ADAPT: real file names from Step 3
DATASET_CSV = MODEL_DATA / "dataset.csv"
QUESTIONS_CSV = MODEL_DATA / "questions_topics.csv"   # one row per question
TOPICS_CSV = MODEL_DATA / "topics.csv"

MODEL_FILES = {
    "logreg": "logreg.joblib",
    "random_forest": "random_forest.joblib",
    "mlp": "mlp_best.pt",
}
# Validation-chosen thresholds from your Deliverable 2 report
THRESHOLDS = {"logreg": 0.470, "random_forest": 0.470, "mlp": 0.484}
MIN_EVIDENCE = 6      # topics seen fewer times are flagged (report Section 6)