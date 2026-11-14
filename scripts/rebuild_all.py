"""Rebuild the whole pipeline in the right order, stopping at the first failure.

    python scripts/rebuild_all.py                 # everything after the OCR step
    python scripts/rebuild_all.py --from topics   # restart from one step
    python scripts/rebuild_all.py --from papers --to qtable
    python scripts/rebuild_all.py --dry-run       # only print the commands
    python scripts/rebuild_all.py --list
Steps are listed in STEPS; add new stages (syllabus topics, model v2, ...) there.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("papers",    [PY, "data scripts/05_build_papers.py"]),
    ("clean",     [PY, "data scripts/06_clean.py"]),
    ("features",  [PY, "data scripts/07_features.py"]),
    ("eda",       [PY, "data scripts/08_eda.py"]),
    ("questions", [PY, "-m", "src.data.build_questions"]),
    ("topics",    [PY, "-m", "src.data.topics"]),
    ("qtable",    [PY, "-m", "src.data.build_question_table"]),
    ("dataset",   [PY, "-m", "src.data.build_dataset"]),
    ("preprocess", [PY, "-m", "src.models.preprocess"]),
    ("baselines", [PY, "-m", "src.models.baselines"]),
    ("tune",      [PY, "-m", "src.models.tune"]),
    ("train_mlp", [PY, "-m", "src.models.train_mlp"]),
    ("export",    [PY, "-m", "src.models.export_predictions"]),
    ("plots",     [PY, "-m", "src.models.plots"]),
    ("eval_baselines", [PY, "-m", "src.evaluation.baselines"]),
    ("evaluate",  [PY, "-m", "src.evaluation.evaluate"]),
    ("errors",    [PY, "-m", "src.evaluation.error_analysis"]),
    ("bias",      [PY, "-m", "src.evaluation.bias_check"]),
    ("tests",     [PY, "-m", "pytest", "tests", "-q"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0])
    ap.add_argument("--to", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [n for n, _ in STEPS]
    if a.list:
        print(" ".join(names))
        return
    for n in (a.start, a.only, a.to):
        if n and n not in names:
            sys.exit(f"unknown step {n!r}; choose from: {' '.join(names)}")
    lo = names.index(a.start)
    hi = names.index(a.to) if a.to else len(names) - 1
    todo = [s for s in STEPS if (s[0] == a.only if a.only else lo <= names.index(s[0]) <= hi)]
    for name, cmd in todo:
        print(f"\n=== {name}: {' '.join(Path(c).name if c == PY else c for c in cmd)}", flush=True)
        if a.dry_run:
            continue
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"\nSTOPPED: step '{name}' failed. Fix it, then rerun with --from {name}")
    print("\nDone." if not a.dry_run else "\n(dry run)")


if __name__ == "__main__":
    main()
