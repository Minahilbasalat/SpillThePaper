"""Step 3 - DOWNLOAD: fetch the selected files into data/raw/ (git-ignored).

Reads : data/processed/selected_files.csv
Saves : data/raw/<file_id>.<ext>           (the downloaded files)
        data/processed/collection_manifest.csv   (what happened to every file)

Usage:
    python "data scripts/03_download.py"            # download everything missing
    python "data scripts/03_download.py" --limit 20 # try only 20 files (for testing)
The script can be stopped and run again: files already downloaded are skipped.
"""
import argparse
import hashlib
import os
import time
from urllib.parse import quote

import pandas as pd
import requests

from common import PROCESSED_DIR, RAW_DIR, load_yaml, log

cfg = load_yaml("repo_config.yaml")
owner, repo = cfg["owner"], cfg["repo"]
MAX_DOWNLOAD_MB = load_yaml("filters.yaml").get("max_download_mb", 100)


def download_one(row):
    """Download one file straight to disk in small chunks (low memory use).
    Returns an empty string on success, or a short message if it failed / was skipped."""
    size_mb = pd.to_numeric(row.size_bytes, errors="coerce") / 1e6
    if pd.notna(size_mb) and size_mb > MAX_DOWNLOAD_MB:
        return f"skipped: too large ({size_mb:.0f} MB)"

    url = (f"https://raw.githubusercontent.com/{owner}/{repo}/"
           f"{row.commit_sha}/{quote(row.repo_path)}")
    local_path = RAW_DIR / f"{row.file_id}{row.extension}"
    temp_path = RAW_DIR / f"{row.file_id}{row.extension}.part"
    last_error = ""
    for attempt in range(3):  # try up to 3 times
        try:
            with requests.get(url, timeout=60, stream=True) as response:
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                else:
                    with open(temp_path, "wb") as out:
                        for chunk in response.iter_content(chunk_size=1024 * 256):
                            out.write(chunk)
                    os.replace(temp_path, local_path)  # only now does the file "exist"
                    return ""
        except Exception as error:  # network problems, memory problems, disk problems...
            last_error = f"{type(error).__name__}"
        if temp_path.exists():
            temp_path.unlink()
        time.sleep(2 * (attempt + 1))
    return last_error


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="download at most this many new files (testing)")
    args = parser.parse_args()

    selected = pd.read_csv(PROCESSED_DIR / "selected_files.csv", keep_default_na=False)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    missing = [r for r in selected.itertuples()
               if not (RAW_DIR / f"{r.file_id}{r.extension}").exists()]
    if args.limit is not None:
        missing = missing[:args.limit]
    log(f"{len(selected)} selected files, {len(missing)} to download in this run")

    errors = {}
    for number, row in enumerate(missing, start=1):
        error = download_one(row)
        if error:
            errors[row.file_id] = error
        if number % 25 == 0 or number == len(missing):
            log(f"  progress: {number}/{len(missing)} (failed so far: {len(errors)})")

    # Build the manifest from what is really on disk
    statuses, sizes, hashes = [], [], []
    for row in selected.itertuples():
        path = RAW_DIR / f"{row.file_id}{row.extension}"
        if path.exists() and path.stat().st_size > 0:
            statuses.append("downloaded")
            sizes.append(path.stat().st_size)
            hashes.append(sha256_of(path))
        else:
            if row.file_id not in errors:
                statuses.append("pending")
            elif errors[row.file_id].startswith("skipped"):
                statuses.append(errors[row.file_id])
            else:
                statuses.append("failed: " + errors[row.file_id])
            sizes.append(None)
            hashes.append("")
    selected["local_file"] = [f"{r.file_id}{r.extension}" for r in selected.itertuples()]
    selected["status"] = statuses
    selected["downloaded_size_bytes"] = sizes
    selected["sha256"] = hashes
    selected.to_csv(PROCESSED_DIR / "collection_manifest.csv", index=False, encoding="utf-8")

    log(f"Manifest status counts: {selected['status'].value_counts().to_dict()}")
    log(f"Saved {PROCESSED_DIR / 'collection_manifest.csv'}")


if __name__ == "__main__":
    main()