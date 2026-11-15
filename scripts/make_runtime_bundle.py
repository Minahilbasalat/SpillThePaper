"""Pack the files the app needs but Git does not hold (models, topic models, question bank)
into ONE zip that you share with teammates (Google Drive). They unzip it into their project
folder.

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
] + sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "models" / "topics").glob("*.joblib"))
OUT = ROOT / "runtime_bundle.zip"

missing = [f for f in FILES if not (ROOT / f).exists()]
if missing or len(FILES) < 10:
    sys.exit("cannot build the bundle, missing: " + ", ".join(missing or ["models/topics/*.joblib (run: python -m src.data.topics)"]))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, f)
print(f"added {len(FILES)} files")
print(f"created {OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")
print("Teammates: copy it into the project folder and unzip it there (keep the folder names).")
