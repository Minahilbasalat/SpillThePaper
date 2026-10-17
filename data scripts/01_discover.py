"""Step 1 - DISCOVER: list every file in the Academic-Time-Machine repository.

Uses GitHub's API to get the whole file list in ONE request (no download of papers).
Saves: data/processed/repo_tree.csv   (one row per file in the repository)
"""
import os
import sys

import pandas as pd
import requests

from common import PROCESSED_DIR, load_yaml, log

cfg = load_yaml("repo_config.yaml")
owner, repo, branch = cfg["owner"], cfg["repo"], cfg["branch"]

headers = {"Accept": "application/vnd.github+json"}
token = os.environ.get("GITHUB_TOKEN")  # optional, only needed if GitHub says "rate limit"
if token:
    headers["Authorization"] = f"Bearer {token}"


def github_get(url):
    """GET a GitHub API url and stop with a helpful message if it fails."""
    response = requests.get(url, headers=headers, timeout=120)
    if response.status_code != 200:
        log(f"ERROR {response.status_code} from {url}: {response.text[:200]}")
        if "rate limit" in response.text.lower():
            log("Hint: wait an hour, or set a GITHUB_TOKEN environment variable.")
        sys.exit(1)
    return response.json()


def build_tree_table(tree_json, commit_sha):
    """Turn GitHub's tree JSON into a table with one row per file."""
    rows = []
    for entry in tree_json["tree"]:
        if entry["type"] != "blob":  # blob = file, tree = folder
            continue
        rows.append({
            "repo_path": entry["path"],
            "size_bytes": entry.get("size"),
            "git_blob_sha": entry["sha"],
        })
    table = pd.DataFrame(rows)
    table["extension"] = table["repo_path"].str.extract(r"(\.[^./]+)$")[0].str.lower().fillna("")
    table["commit_sha"] = commit_sha
    return table


def main():
    log(f"Discovering files in {owner}/{repo} (branch {branch})")
    commit = github_get(f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}")
    commit_sha = commit["sha"]
    log(f"Commit being collected (for reproducibility): {commit_sha}")

    tree_json = github_get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit_sha}?recursive=1")
    if tree_json.get("truncated"):
        log("WARNING: GitHub says the file list was truncated. Results are incomplete.")

    table = build_tree_table(tree_json, commit_sha)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "repo_tree.csv"
    table.to_csv(out_path, index=False, encoding="utf-8")

    log(f"Total files found: {len(table)}")
    log(f"Top file types: {table['extension'].value_counts().head(8).to_dict()}")
    log(f"Saved {out_path}")


if __name__ == "__main__":
    main()