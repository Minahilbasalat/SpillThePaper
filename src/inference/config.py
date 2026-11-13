"""Paths and constants for the inference layer (real file names of this repo)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "results" / "predictions"
METRICS_VAL_CSV = ROOT / "results" / "metrics_val.csv"

QUESTIONS_CSV = ROOT / "data" / "interim" / "questions_topics.csv"   # not in Git
TOPIC_CATALOG_CSV = ROOT / "data" / "processed" / "topic_catalog.csv"
ELIGIBILITY_CSV = ROOT / "data" / "processed" / "subject_eligibility.csv"

MODEL_FILES = {
    "logreg": "logreg.joblib",
    "random_forest": "random_forest.joblib",
    "mlp": "mlp_best.pt",
}
DEFAULT_THRESHOLDS = {"logreg": 0.470, "random_forest": 0.470, "mlp": 0.484}
MIN_EVIDENCE = 6        # topics seen fewer times are flagged as low-evidence
RECENT_WINDOW = 2       # must equal src/data/build_dataset.py RECENT_WINDOW