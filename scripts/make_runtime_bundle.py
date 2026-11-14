"""Pack the files the app needs but Git does not hold (models and question bank) into ONE zip
that you share with teammates (Google Drive). They unzip it into their project folder.

    python scripts/make_runtime_bundle.py
Run it again after every stage that changes the data or the models."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "models/preprocess.joblib", "models/logreg.joblib", "models/random_forest.joblib",
    "models/mlp_best.pt",
    "data/interim/question_table.csv", "data/interim/questions_topics.csv",
]
OUT = ROOT / "runtime_bundle.zip"

missing = [f for f in FILES if not (ROOT / f).exists()]
if missing:
    sys.exit("cannot build the bundle, missing: " + ", ".join(missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, f)
        print(f"added {f}  ({(ROOT / f).stat().st_size / 1e6:.2f} MB)")
print(f"\ncreated {OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")
print("Teammates: copy it into the project folder and unzip it there (keep the folder names).")
