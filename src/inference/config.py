"""Paths and constants for the inference layer (real file names of this repo)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "results" / "predictions"
METRICS_VAL_CSV = ROOT / "results" / "metrics_val.csv"

# derived data written by the Phase 3 scripts
QUESTIONS_CSV = ROOT / "data" / "interim" / "questions_topics.csv"   # not in Git
QUESTION_TABLE_CSV = ROOT / "data" / "interim" / "question_table.csv"  # not in Git
PAPERS_CSV = ROOT / "data" / "processed" / "papers_clean.csv"
BLUEPRINTS_CSV = ROOT / "data" / "processed" / "paper_blueprints.csv"
TOPIC_CATALOG_CSV = ROOT / "data" / "processed" / "topic_catalog.csv"
ELIGIBILITY_CSV = ROOT / "data" / "processed" / "subject_eligibility.csv"

MODEL_FILES = {
    "logreg": "logreg.joblib",
    "random_forest": "random_forest.joblib",
    "mlp": "mlp_best.pt",
}
# fallback values (validation-chosen F1 thresholds from the Deliverable 2 report)
DEFAULT_THRESHOLDS = {"logreg": 0.470, "random_forest": 0.470, "mlp": 0.484}
MIN_EVIDENCE = 6        # topics seen fewer times are flagged as low-evidence
RECENT_WINDOW = 2       # must equal src/data/build_dataset.py RECENT_WINDOW

# where the original papers live (pinned snapshot used for the whole project)
ARCHIVE_URL = "https://github.com/saleha-muzammil/Academic-Time-Machine/blob"
RAW_URL = "https://raw.githubusercontent.com/saleha-muzammil/Academic-Time-Machine"
CACHE_DIR = ROOT / "data" / "cache"                     # solution cache (NOT in Git)
ARCHIVE_COMMIT = "e0e197577432957a5fad0f76d2ab92d78f48e1d8"
