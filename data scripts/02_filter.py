"""Step 2 - FILTER: keep only exam papers (mid / final / quiz) and course outlines.

Reads : data/processed/repo_tree.csv
Rules : config/filters.yaml and config/assessment_rules.yaml (no subject names in code)
Saves : data/processed/selected_files.csv   (files we will download)
        data/processed/excluded_files.csv   (files skipped + the reason, for transparency)
"""
import re

import pandas as pd

from common import PROCESSED_DIR, load_yaml, log, make_file_id

filters = load_yaml("filters.yaml")
rules = load_yaml("assessment_rules.yaml")

assessment_regex = {name: re.compile(p, re.I) for name, p in rules["assessment_patterns"].items()}
outline_regex = re.compile(rules["outline_pattern"], re.I)
fallback_regex = re.compile(rules["fallback_exam_pattern"], re.I)
exclude_folder_regex = [re.compile(p, re.I) for p in rules["exclude_folder_patterns"]]
exclude_name_regex = [re.compile(p, re.I) for p in rules["exclude_filename_patterns"]]


def find_assessment(text):
    """Return 'final', 'mid' or 'quiz' if the text matches a rule, else None."""
    for name, regex in assessment_regex.items():
        if regex.search(text):
            return name
    return None


def split_path(repo_path):
    """Split a repo path into (subject, folders_below_subject, file_name, container)."""
    parts = repo_path.split("/")
    file_name, folders = parts[-1], parts[:-1]
    if folders and folders[0] in filters["container_folders"]:
        subject = folders[1] if len(folders) > 1 else ""
        return subject, folders[2:], file_name, folders[0]
    subject = folders[0] if folders else ""
    return subject, folders[1:], file_name, ""


def classify(repo_path, extension):
    """Decide what to do with one file. Returns (assessment_type, subject, reason).
    assessment_type is None when the file is excluded; reason then says why."""
    subject, inner_folders, file_name, container = split_path(repo_path)
    top = repo_path.split("/")[0] if "/" in repo_path else ""

    if top in filters["exclude_top_level"]:
        return None, subject, "excluded_top_level_folder"
    if extension not in filters["allowed_extensions"]:
        return None, subject, "extension_not_allowed"

    whitelist = filters.get("subject_whitelist") or []
    if whitelist and subject not in whitelist:
        return None, subject, "subject_not_in_whitelist"

    # 1. Course outlines
    if container in filters["outline_containers"] or outline_regex.search(file_name):
        return "outline", subject, "ok"

    # 2. File names that are clearly not exams (assignment, slides, ...) are always skipped
    if any(r.search(file_name) for r in exclude_name_regex):
        return None, subject, "non_exam_file_name"

    # 3. File name says quiz / mid / final
    assessment = find_assessment(file_name)
    if assessment:
        return assessment, subject, "ok"

    # 4. Walk the folders from the deepest upwards. The first folder that either names an
    #    exam type (keep) or is a non-exam folder (skip) decides.
    for folder in reversed(inner_folders):
        assessment = find_assessment(folder)
        if assessment:
            return assessment, subject, "ok"
        if any(r.search(folder) for r in exclude_folder_regex):
            return None, subject, "non_exam_folder"

    # 5. Fallback: looks like an exam paper but the type is not written anywhere
    if fallback_regex.search(file_name) or any(fallback_regex.search(f) for f in inner_folders):
        return "exam_unspecified", subject, "ok"
    return None, subject, "no_assessment_keyword"


def main():
    tree = pd.read_csv(PROCESSED_DIR / "repo_tree.csv", keep_default_na=False)
    log(f"Filtering {len(tree)} files")

    results = [classify(p, e) for p, e in zip(tree["repo_path"], tree["extension"])]
    tree["assessment_type"] = [r[0] for r in results]
    tree["subject_raw"] = [r[1] for r in results]
    tree["reason"] = [r[2] for r in results]

    selected = tree[tree["assessment_type"].notna()].copy()
    excluded = tree[tree["assessment_type"].isna()].copy()
    selected["file_id"] = selected["repo_path"].apply(make_file_id)

    cols = ["file_id", "repo_path", "subject_raw", "assessment_type", "extension",
            "size_bytes", "commit_sha"]
    selected[cols].to_csv(PROCESSED_DIR / "selected_files.csv", index=False, encoding="utf-8")
    excluded[["repo_path", "extension", "reason"]].to_csv(
        PROCESSED_DIR / "excluded_files.csv", index=False, encoding="utf-8")

    log(f"Selected: {len(selected)}   Excluded: {len(excluded)}")
    log(f"Selected by type: {selected['assessment_type'].value_counts().to_dict()}")
    log(f"Selected by extension: {selected['extension'].value_counts().to_dict()}")
    log(f"Excluded by reason: {excluded['reason'].value_counts().to_dict()}")
    log(f"Subjects with selected files: {selected['subject_raw'].nunique()}")
    total_mb = pd.to_numeric(selected["size_bytes"], errors="coerce").sum() / 1e6
    log(f"Approximate download size: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()