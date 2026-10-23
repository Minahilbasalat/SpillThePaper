"""Check config/subject_aliases.yaml before using it.

Finds: typos (names that do not exist in the data), names listed twice,
and shows how many folder names are still unlisted.
Run:  python "data scripts/check_aliases.py"
"""
import collections

import pandas as pd
import yaml

from common import CONFIG_DIR, PROCESSED_DIR

aliases = yaml.safe_load(open(CONFIG_DIR / "subject_aliases.yaml", encoding="utf-8")) or {}
papers = pd.read_csv(PROCESSED_DIR / "papers.csv", keep_default_na=False)
outlines = pd.read_csv(PROCESSED_DIR / "outlines.csv", keep_default_na=False)
raw_names = set(papers["subject_raw"]) | set(outlines["subject_raw"])
existing = {name.strip().lower() for name in raw_names}

listed = collections.Counter()
for clean_name, raws in aliases.items():
    for raw in ([raws] if isinstance(raws, str) else raws):
        listed[str(raw).strip().lower()] += 1

problems = 0
for name, times in listed.items():
    if times > 1:
        print(f"LISTED TWICE (in more than one group): {name}")
        problems += 1
    if name not in existing:
        print(f"NOT FOUND in the data (typo?): {name}")
        problems += 1

merging = {clean: raws for clean, raws in aliases.items() if not isinstance(raws, str) and len(raws) > 1}
print(f"{len(aliases)} groups, {len(merging)} of them merge 2+ folder names, {len(listed)} names listed")
print(f"{len(existing - set(listed))} folder names are not listed (they keep their own name)")
print("OK - no problems found" if problems == 0 else f"{problems} problem(s) to fix")