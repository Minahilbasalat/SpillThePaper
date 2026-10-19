"""Step 5 - BUILD the papers table: structured metadata for every downloaded file.

Reads : data/processed/collection_manifest.csv, data/processed/extraction_results.csv,
        data/interim/text/<file_id>.txt, config/subject_aliases.yaml (optional)
Saves : data/processed/papers.csv     (exam papers: quiz / mid / final / exam_unspecified)
        data/processed/outlines.csv   (course outlines)
        data/processed/missing_values_report.csv

Three kinds of columns (see the report):
  RAW        - read directly from the path / file name (subject_raw, assessment_type, ...)
  EXTRACTED  - read from the file's text (total_marks, question_count_est, header term/year)
  DERIVED    - computed by us (subject after name cleaning, flags, has_text, ...)

Nothing is deleted here. Missing values stay empty and are counted in the report.
"""
import os
import re

import pandas as pd
import yaml

from common import CONFIG_DIR, PROCESSED_DIR, ROOT, load_yaml, log

TEXT_DIR = ROOT / "data" / "interim" / "text"
# The repository covers roughly 2012-2026. A "year" outside this range is a parsing
# mistake (for example a room number), so we ignore it.
MIN_YEAR, MAX_YEAR = 2005, 2026

SEASON_NAMES = {"spring": "Spring", "sp": "Spring", "s": "Spring",
                "fall": "Fall", "fa": "Fall", "f": "Fall",
                "summer": "Summer", "su": "Summer"}

# ---- term / year patterns (tried in this order: full words, abbreviations, bare year) ----
SEASON_THEN_YEAR = re.compile(r"(spring|fall|summer)[\s\-_'.]*((?:20)?\d{2})(?!\d)", re.I)
YEAR_THEN_SEASON = re.compile(r"(20\d{2})[\s\-_]*(spring|fall|summer)", re.I)
ABBREVIATION = re.compile(r"(?<![a-z0-9])(sp|su|fa|f|s)[\s\-_]?(\d{2})(?![0-9a-z])", re.I)
BARE_YEAR = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d|[\s\-_]*\d{2}(?!\d))")  # skips ranges 2018-19

# ---- other patterns ----
ROMAN = {"i": 1, "ii": 2, "iii": 3}
MID_NUMBER = re.compile(r"mid(?:term)?[\s\-_]*(?:no\.?)?[\s\-_#]*(iii|ii|i|1|2|3)(?![a-z0-9])", re.I)
QUIZ_NUMBER = re.compile(r"quiz(?:zes)?[\s\-_#]*(\d{1,2})(?!\d)", re.I)
SECTION_CODE = re.compile(r"(?<![a-z])(?:bcs|bse|bai|bds|bcy|bsd)[\s\-_]*\d[a-h](?![a-z])", re.I)
SECTION_FOLDER = re.compile(r"^\(?[a-h](?:\s*(?:and|&|,|\+)\s*[a-h])*\)?$", re.I)
SECTION_IN_NAME = re.compile(r"\(([a-h](?:\s*(?:and|&|,)\s*[a-h])*)\)\s*$", re.I)
SOLUTION = re.compile(r"solution|(?<![a-z])sols?(?![a-z])|answer[\s\-_]*key|(?<![a-z])ans(?![a-z])", re.I)
MERGED = re.compile(r"merged?|compiled|combined|all[\s\-_]*(?:past|papers)", re.I)
RETAKE = re.compile(r"retake|re[\s\-_]*take|make[\s\-_]*up|supplementary|improvement", re.I)
SYLLABUS_TAG = re.compile(r"(old|new)[\s\-_]*syllabus", re.I)
IS_LAB = re.compile(r"(?<![a-z])lab(?![a-z])", re.I)
TOTAL_MARKS = re.compile(r"(?:total|max(?:imum)?)\s*(?:marks|points)\s*(?:are|is)?\s*[:\-=]?\s*(\d{1,3})", re.I)
MARKS_LINE = re.compile(r"(?:^|\n)[ \t]*marks[ \t]*[:\-=][ \t]*(\d{1,3})[ \t]*(?:\n|$)", re.I)  # a line that is only "Marks: 40"
QUESTION_NUMBER = re.compile(r"(?:^|\n)\s*q(?:uestion)?\s*\.?\s*(?:no\.?)?\s*#?\s*(\d{1,2})\b", re.I)


# ---------------------------------------------------------------- path handling
def path_parts(repo_path, container_folders):
    """Return (subject_raw, folders_below_subject, file_stem)."""
    parts = repo_path.split("/")
    stem = os.path.splitext(parts[-1])[0]
    folders = parts[:-1]
    if folders and folders[0] in container_folders:  # e.g. "---- Course Outlines ----/<subject>/"
        return (folders[1] if len(folders) > 1 else ""), folders[2:], stem
    return (folders[0] if folders else ""), folders[1:], stem


def search_order(folders, stem):
    """File name first, then folders from the deepest upwards."""
    return [stem] + list(reversed(folders))


# ---------------------------------------------------------------- parsers
def valid_year(text):
    year = int(text)
    year = year + 2000 if year < 100 else year
    return year if MIN_YEAR <= year <= MAX_YEAR else None


def parse_term_year(texts):
    """Return (term, year, source). Tries full season+year first, then abbreviations
    like 'F22', then a bare year like '2024'."""
    for text in texts:
        for regex, season_group, year_group in ((SEASON_THEN_YEAR, 1, 2), (YEAR_THEN_SEASON, 2, 1)):
            match = regex.search(text)
            if match and valid_year(match.group(year_group)):
                return (SEASON_NAMES[match.group(season_group).lower()],
                        valid_year(match.group(year_group)), "path")
    for text in texts:
        match = ABBREVIATION.search(text)
        if match and valid_year(match.group(2)):
            return SEASON_NAMES[match.group(1).lower()], valid_year(match.group(2)), "path"
    for text in texts:
        match = BARE_YEAR.search(text)
        if match and valid_year(match.group(1)):
            return "", valid_year(match.group(1)), "path_year_only"
    return "", None, ""


def parse_assessment_number(assessment_type, texts):
    """Mid 1 / Mid 2 / Quiz 3 ... or None when the number is not written."""
    regex = {"mid": MID_NUMBER, "quiz": QUIZ_NUMBER}.get(assessment_type)
    if regex is None:
        return None
    for text in texts:
        match = regex.search(text)
        if match:
            value = match.group(1).lower()
            return ROMAN.get(value, int(value) if value.isdigit() else None)
    return None


def parse_section(folders, stem):
    """Class section such as 'BSE-5B' or 'A and B' (kept as raw text; not validated)."""
    for text in [stem] + list(reversed(folders)):
        match = SECTION_CODE.search(text)
        if match:
            return match.group(0)
    for folder in reversed(folders):
        if SECTION_FOLDER.match(folder.strip()):
            return folder.strip()
    match = SECTION_IN_NAME.search(stem)
    return match.group(1) if match else ""


def parse_header_term_year(text):
    """Look for 'Spring 2023' in the first part of the paper text."""
    return parse_term_year([text[:2500]])


def parse_total_marks(text):
    match = TOTAL_MARKS.search(text[:3000]) or MARKS_LINE.search(text[:3000])
    return int(match.group(1)) if match else None


def estimate_question_count(text):
    """Rough count of distinct question numbers (Q1, Question 2, ...). An ESTIMATE only."""
    numbers = {int(n) for n in QUESTION_NUMBER.findall(text)}
    numbers = {n for n in numbers if 1 <= n <= 30}
    return len(numbers) if numbers else None


def clean_subject_names():
    """Build raw-name -> clean-name lookup from config/subject_aliases.yaml, if it exists.
    File format:  Clean Name: [raw folder name, another raw folder name]"""
    path = CONFIG_DIR / "subject_aliases.yaml"
    lookup = {}
    if path.exists():
        for clean, raws in (yaml.safe_load(open(path, encoding="utf-8")) or {}).items():
            if isinstance(raws, str):  # allow  Clean Name: "Raw Name"  without brackets
                raws = [raws]
            for raw in raws:
                lookup[str(raw).strip().lower()] = clean
    return lookup


def tidy_name(raw):
    """Basic cleaning when no alias exists: trim, collapse spaces, and use '&' = 'and'."""
    return re.sub(r"\s+", " ", raw.replace("&", "and")).strip().strip('"')


# ---------------------------------------------------------------- main
def build_row(row, container_folders, alias_lookup, extraction):
    subject_raw, folders, stem = path_parts(row.repo_path, container_folders)
    texts = search_order(folders, stem)
    term, year, term_source = parse_term_year(texts)
    all_names = " / ".join(folders + [stem])

    record = {
        # RAW (from path / file name)
        "file_id": row.file_id, "repo_path": row.repo_path, "extension": row.extension,
        "subject_raw": subject_raw, "assessment_type": row.assessment_type,
        "assessment_number": parse_assessment_number(row.assessment_type, texts),
        "term": term, "year": year, "term_source": term_source,
        "section_raw": parse_section(folders, stem),
        "syllabus_tag": (SYLLABUS_TAG.search(" / ".join(folders)).group(1).lower()
                         if SYLLABUS_TAG.search(" / ".join(folders)) else ""),
        # DERIVED flags
        "is_solution": bool(SOLUTION.search(all_names)),
        "is_merged": bool(MERGED.search(all_names)),
        "is_retake": bool(RETAKE.search(all_names)),
        "is_lab_subject": bool(IS_LAB.search(subject_raw)),
        "subject": alias_lookup.get(subject_raw.strip().lower(), tidy_name(subject_raw)),
    }

    # EXTRACTED (from the paper's text)
    info = extraction.get(row.file_id, {})
    record["extraction_method"] = info.get("extraction_method", "")
    record["page_count"] = info.get("page_count")
    record["char_count"] = info.get("char_count", 0)
    record["word_count"] = info.get("word_count", 0)
    record["has_text"] = info.get("extraction_method") in ("pdf_text", "docx") and info.get("char_count", 0) > 0
    text_path = TEXT_DIR / f"{row.file_id}.txt"
    text = text_path.read_text(encoding="utf-8") if record["has_text"] and text_path.exists() else ""
    record["total_marks"] = parse_total_marks(text) if text else None
    record["question_count_est"] = estimate_question_count(text) if text else None

    # Fill a missing year from the paper's own header
    if record["year"] is None and text:
        header_term, header_year, _ = parse_header_term_year(text)
        if header_year:
            record["term"], record["year"], record["term_source"] = header_term, header_year, "header"
    return record


def missing_report(papers):
    """Count missing values in the exam-papers table."""
    exam_number = papers[papers["assessment_type"].isin(["mid", "quiz"])]
    checks = [
        ("subject", (papers["subject"] == "").sum(), len(papers), "no subject folder (root-level file)"),
        ("assessment_type = exam_unspecified", (papers["assessment_type"] == "exam_unspecified").sum(),
         len(papers), "type not written in path"),
        ("year", papers["year"].isna().sum(), len(papers), "no year in path or header"),
        ("term (season)", ((papers["term"] == "") & papers["year"].notna()).sum(), papers["year"].notna().sum(),
         "year known but season unknown"),
        ("assessment_number (mid/quiz only)", exam_number["assessment_number"].isna().sum(),
         len(exam_number), "e.g. folder just says 'Mids'"),
        ("section_raw", (papers["section_raw"] == "").sum(), len(papers), "section not written"),
        ("text (has_text = False)", (~papers["has_text"]).sum(), len(papers), "scanned PDF / photo / error"),
        ("total_marks (among papers with text)", papers[papers["has_text"]]["total_marks"].isna().sum(),
         papers["has_text"].sum(), "no 'Total Marks' in header"),
        ("question_count_est (among papers with text)",
         papers[papers["has_text"]]["question_count_est"].isna().sum(), papers["has_text"].sum(),
         "no Q1/Question 1 pattern"),
    ]
    report = pd.DataFrame(checks, columns=["field", "missing", "out_of", "note"])
    report["percent"] = (100 * report["missing"] / report["out_of"].clip(lower=1)).round(1)
    return report


def main():
    filters = load_yaml("filters.yaml")
    manifest = pd.read_csv(PROCESSED_DIR / "collection_manifest.csv", keep_default_na=False)
    manifest = manifest[manifest["status"] == "downloaded"]
    extraction = pd.read_csv(PROCESSED_DIR / "extraction_results.csv", keep_default_na=False)
    extraction = extraction.set_index("file_id").to_dict("index")
    alias_lookup = clean_subject_names()
    log(f"Building table for {len(manifest)} files "
        f"({'with' if alias_lookup else 'without'} subject_aliases.yaml)", "cleaning_log.txt")

    records = [build_row(r, filters["container_folders"], alias_lookup, extraction)
               for r in manifest.itertuples()]
    table = pd.DataFrame(records)
    table["page_count"] = pd.to_numeric(table["page_count"], errors="coerce")

    papers = table[table["assessment_type"] != "outline"].copy()
    outlines = table[table["assessment_type"] == "outline"].copy()
    papers.to_csv(PROCESSED_DIR / "papers.csv", index=False, encoding="utf-8")
    outlines.to_csv(PROCESSED_DIR / "outlines.csv", index=False, encoding="utf-8")

    report = missing_report(papers)
    report.to_csv(PROCESSED_DIR / "missing_values_report.csv", index=False, encoding="utf-8")

    log(f"papers.csv: {len(papers)} rows   outlines.csv: {len(outlines)} rows", "cleaning_log.txt")
    log(f"Subjects (clean names): {papers['subject'].nunique()}", "cleaning_log.txt")
    log(f"Term source counts: {papers['term_source'].replace('', 'none').value_counts().to_dict()}",
        "cleaning_log.txt")
    log(f"Flags: solution={int(papers['is_solution'].sum())} merged={int(papers['is_merged'].sum())} "
        f"retake={int(papers['is_retake'].sum())}", "cleaning_log.txt")
    log("Missing values:", "cleaning_log.txt")
    for r in report.itertuples():
        log(f"  {r.field}: {r.missing}/{r.out_of} ({r.percent}%)", "cleaning_log.txt")


if __name__ == "__main__":
    main()