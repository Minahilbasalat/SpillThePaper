# SpillThePaper

**AI-powered analysis of historical FAST-NUCES exam papers, aiming to produce a tentative prediction of an upcoming paper and personalised practice.**

AI course project, **Track A (Application Development)**.

> **Current status: Deliverable 1 complete (Phase 1: problem definition, Phase 2: data acquisition and preprocessing).**
> No prediction model, paper generator, user interface or deployment exists yet. These are future phases (see below).

---

## 1. What is SpillThePaper?

Students preparing for exams have access to many past mid, final and quiz papers, but the archive is large, unstructured and inconsistent. SpillThePaper's long-term goal is to turn it into evidence-based preparation:

1. analyse past papers (topics, question types, marks, recency, assessment type),
2. compare them with the **current** syllabus,
3. estimate how likely syllabus areas are to appear (keeping some lower-probability areas, to stay honest about uncertainty),
4. generate a tentative paper with a realistic structure,
5. let a student attempt it, find weak areas, and generate a personalised follow-up paper.

It is a study aid. It does **not** guarantee what will appear in an actual exam.

**Target audience:** FAST-NUCES undergraduate students preparing for courses with enough historical paper data.

## 2. Project status

| Phase | Description | Status |
|---|---|---|
| 1 | Problem definition (problem statement, audience, AI justification) | Done (see the report) |
| 2 | Data acquisition and preprocessing (collection, cleaning, feature strategy, EDA) | **Done** |
| 3 | Implementation (question-level extraction, NLP, prediction model) | Not started |
| 4 | Evaluation (backtesting, metrics, bias check) | Not started |
| 5 | Output (hosted web application) | Not started |
| 6 | Final documentation and report | Not started |

## 3. Data source

Papers and course outlines come from the public GitHub archive
[Academic-Time-Machine](https://github.com/saleha-muzammil/Academic-Time-Machine) (student-contributed, not curated).

* The collection in the report was pinned to commit `e0e197577432957a5fad0f76d2ab92d78f48e1d8`. The commit used is also recorded in `logs/collection_log.txt` and `data/processed/repo_tree.csv`.
* **Raw papers are not stored in this repository.** The scripts download them into `data/raw/` (excluded by `.gitignore`), so the dataset is reproducible without redistributing files.
* The archive contains **no reliable professor information** for papers or quizzes, so professor-specific analysis is not part of this project phase.

## 4. Data pipeline (Phase 2)

All code is in [`data scripts/`](data%20scripts). Rules are in [`config/`](config), so **adding a new subject, semester or paper to the archive needs no code change**.

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_discover.py` | Lists every file in the archive through the GitHub API (no papers downloaded) | `data/processed/repo_tree.csv` |
| 2 | `02_filter.py` | Keeps quizzes, mids, finals and course outlines using configurable rules; lists every excluded file with its reason | `selected_files.csv`, `excluded_files.csv` |
| 3 | `03_download.py` | Downloads the selected files (resumable, streamed to disk) and records size and SHA-256 | `collection_manifest.csv` |
| 4 | `04_extract.py` | Extracts text (PyMuPDF for PDFs, python-docx for Word); flags scans and photos as *needs OCR* | `extraction_results.csv` |
| 5 | `05_build_papers.py` | Builds the papers table: subject, assessment type and number, term, year, flags; reports missing values | `papers.csv`, `outlines.csv`, `missing_values_report.csv` |
| 6 | `06_clean.py` | Removes duplicates, treats invalid values as missing, flags numeric outliers | `papers_clean.csv`, `papers_flagged.csv`, `cleaning_summary.csv`, `outlier_bounds.csv` |
| 7 | `07_features.py` | Encoding, scaling and normalisation strategy, applied only where meaningful | `features.csv`, `feature_strategy.csv` |
| 8 | `08_eda.py` | Nine exploratory charts and a log of the numbers behind them | `eda/*.png`, `logs/eda_log.txt` |

Helper tools: `check_aliases.py` (validates the subject-name mapping) and `peek_subject.py` (shows the first lines of papers from given subject folders, to check whether two folder names are the same course).

## 5. Repository structure

```text
SpillThePaper/
├── README.md
├── .gitignore
├── requirements.txt
├── config/              # filter rules, assessment rules, subject name mapping
├── data scripts/        # all Phase 2 code
├── data/
│   ├── raw/             # downloaded papers        (NOT in Git)
│   ├── interim/         # extracted text           (NOT in Git)
│   └── processed/       # small result tables (CSV)
├── eda/                 # exploratory charts (PNG)
├── logs/                # execution logs (collection, extraction, cleaning, EDA)
└── report/              # Deliverable 1 report (LaTeX source and PDF)
```

## 6. Setup

Requirements: Python 3.10 or newer and Git.

```powershell
git clone https://github.com/Minahilbasalat/SpillThePaper.git
cd SpillThePaper
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

## 7. Running the Phase 2 scripts

Run from the project root, in this order:

```powershell
python "data scripts/01_discover.py"
python "data scripts/02_filter.py"
python "data scripts/03_download.py" --limit 20   # quick test first
python "data scripts/03_download.py"              # full download
python "data scripts/04_extract.py"
python "data scripts/05_build_papers.py"
python "data scripts/06_clean.py"
python "data scripts/07_features.py"
python "data scripts/08_eda.py"
```

Notes:

* The full download was about 2.4 GB (3,567 files) and took roughly one and a half to two hours in our run. It can be stopped and re-run; finished files are skipped.
* If step 1 reports a GitHub *rate limit*, set a personal token first: `$env:GITHUB_TOKEN="<token>"` (never commit a token).
* The archive keeps growing, so a new run on the latest commit can give slightly different numbers than the report. The commit used is recorded in the logs.
* Step 5 reads `config/subject_aliases.yaml` if it exists. Check it with `python "data scripts/check_aliases.py"`.

## 8. Key numbers from our run

All values below come from the logs in `logs/` and the tables in `data/processed/`.

| Item | Value |
|---|---|
| Files in the archive (at the collected commit) | 8,562 |
| Files selected (quiz 1,400, mid 1,215, final 819, outline 124, unspecified exam 9) | 3,567 |
| Files downloaded without failure | 3,567 |
| Text extracted (PDF 2,132, Word 311) | 2,443 |
| Flagged for OCR (scanned PDF or photo) | 1,123 |
| Exam papers in the papers table | 3,443 |
| Papers after removing duplicates, merged compilations and solution companions | 2,874 |
| Clean papers with readable text | 1,823 |
| Subject labels in the clean table | 131 |
| Year known (clean papers) | 2,349 of 2,874 (years 2008 to 2026) |

## 9. Limitations

* About one third of exam papers (32.6%) have no extractable text (scans and photographs), most often finals. 24 subjects have no readable paper.
* Question-level data (topic, difficulty, question type) is **not** available yet; it needs NLP and belongs to Phase 3.
* Subject coverage is uneven (median 11 papers per subject; 31 subjects have fewer than five).
* Year, term, assessment type and number are parsed with rules from paths and headers; accuracy on a manually labelled sample has not been measured.
* Total marks and question counts are estimates from text patterns and are missing for about a third of readable papers.
* Median-filled numeric features are placeholders, not observed data.
* Solution files without a plain companion are kept (flagged), so some question text may repeat.
* No professor-specific analysis (no reliable instructor information for papers or quizzes).
* The source archive is community-maintained and may contain mislabelled or incomplete files. Course outlines come from different terms, so none can be assumed to be the current syllabus.

## 10. Team

| Name | Roll number |
|---|---|
| Minahil Basalat | 24L-3082 |
| Iman Abid | 24L-0707 |
| Khadija Rao | 24L-3042 |

## 11. Acknowledgements and data notes

* Past papers and outlines come from the community archive *Academic-Time-Machine*; credit to its maintainers and contributors.
* Raw files are not redistributed here. Course outlines may name instructors; instructor names are not used or published in this project.
