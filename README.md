# SpillThePaper

**AI-powered analysis of historical FAST-NUCES exam papers, aiming to produce a tentative prediction of an upcoming paper and personalised practice.**

AI course project, **Track A (Application Development)**.

> **Current status: Deliverables 1 and 2 complete (Phases 1 to 4).**
> The data foundation, a supervised dataset, three trained models and a full evaluation exist.
> The tentative-paper generator, personalisation, web application and deployment are future phases (see below).

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
| 1 | Problem definition (problem statement, audience, AI justification) | Done (Deliverable 1 report) |
| 2 | Data acquisition and preprocessing (collection, cleaning, feature strategy, EDA) | Done (Deliverable 1) |
| 3 | Model architecture and implementation (question segmentation, topic discovery, supervised dataset, three models, hyperparameter tuning) | **Done (Deliverable 2)** |
| 4 | Training and evaluation (time-based validation, metrics, error analysis, bias check) | **Done (Deliverable 2)** |
| 5 | Output (hosted web application) | Not started |
| 6 | Final documentation and report | Not started |

## 3. Data source

Papers and course outlines come from the public GitHub archive
[Academic-Time-Machine](https://github.com/saleha-muzammil/Academic-Time-Machine) (student-contributed, not curated).

* The collection in the report was pinned to commit `e0e197577432957a5fad0f76d2ab92d78f48e1d8`. The commit used is also recorded in `logs/collection_log.txt` and `data/processed/repo_tree.csv`.
* **Raw papers are not stored in this repository.** The scripts download them into `data/raw/` (excluded by `.gitignore`), so the dataset is reproducible without redistributing files.
* The archive contains **no reliable professor information** for papers or quizzes, so professor-specific analysis is not part of this project.

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

## 5. Phase 3: from papers to a supervised task and models

Deliverable 1 produced no labels, so Phase 3 first defines a prediction target.

**Task.** One row is a (subject, topic, target exam) triple. The label is 1 if the topic appears in at least one paper of that subject, year and assessment type. All features use only papers from strictly earlier years. The model estimates the probability that a topic appears.

| Stage | Code | What it does |
|---|---|---|
| Text loader | `src/data/load_text.py` | Joins `papers_clean.csv` with the extracted text (1,638 papers with readable text and a known year) |
| Question segmentation | `src/data/segment_questions.py`, `build_questions.py` | Splits papers into 7,780 questions; removes solutions, grading tables and header text |
| Topic discovery | `src/data/topics.py` | Per subject: TF-IDF and KMeans (K chosen by silhouette), fitted on training years only; 28 subjects and 220 topics qualify |
| Dataset builder | `src/data/build_dataset.py` | Builds the leakage-safe dataset and the time-based split |
| Preprocessing | `src/models/preprocess.py` | Imputation, one-hot encoding and scaling, fitted on the training split only |
| Baselines | `src/models/baselines.py` | Logistic regression and random forest (Scikit-Learn) |
| MLP | `src/models/mlp.py`, `train_mlp.py`, `tune.py` | PyTorch multi-layer perceptron (12-16-8-1, dropout 0.4, Adam), random search over 40 configurations with 3 seeds each |

**Dataset and split (by time, not random):**

| Split | Target years | Rows | Positive rate | Target exams |
|---|---|---|---|---|
| Train | 2014 to 2022 | 1,421 | 0.473 | 166 |
| Validation | 2023 | 301 | 0.392 | 36 |
| Test | 2024 to 2026 | 568 | 0.347 | 67 |

The committed files are `data/processed/model/train.csv`, `val.csv` and `test.csv`; the schema is in `docs/dataset_schema.md`. Models, hyperparameters and decision thresholds are chosen on validation only; the test split is evaluated once.

## 6. Phase 4: evaluation

Code: `src/evaluation/`. Documentation: `docs/evaluation.md`.

* **Metrics:** precision, recall, F1, AUC-ROC, PR-AUC and Brier score, with 95% intervals from a cluster bootstrap (whole target exams are resampled, 1,000 resamples).
* **Baselines:** four non-learned rules (count of earlier appearances at least 5, appeared last year, raw topic rate, all positive).
* **Error analysis and bias check:** breakdowns by assessment type, year, history, topic familiarity, subject, coverage tier and subject kind (`results/error_analysis_*.csv`, `results/bias_check*.csv`).

## 7. Results (test split)

Thresholds were fixed on validation. Intervals are 95% cluster-bootstrap intervals.

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic regression | 0.839 [0.796, 0.880] | 0.686 |
| Random forest | 0.843 [0.804, 0.881] | 0.703 |
| MLP (PyTorch) | 0.849 [0.811, 0.886] | 0.704 |
| Rule: appeared last year | 0.827 [0.787, 0.865] | 0.672 |
| Rule: count of at least 5 | 0.822 [0.783, 0.859] | 0.585 |
| Raw topic rate | 0.787 [0.742, 0.830] | 0.608 |

**Honest summary.** The learned models are competitive with, but not demonstrably better than, simple history rules: all AUC-ROC intervals overlap. They are clearly more precise than the "appeared at least 5 times" rule, which flags 81% of the test rows. Quizzes and Computer Networks are the weakest cases, probabilities are overestimated, and the test split contains no topic that never appeared before. See the Deliverable 2 report for the full analysis.

## 8. Model weights

Weight files are not stored in Git. They are hosted on Google Drive:

**https://drive.google.com/drive/folders/1IQb9xincr7PjG4kQvqEmj8jBBezRqQRZ?usp=drive_link**

The folder contains `mlp_best.pt` (final MLP, 353 parameters), `preprocess.joblib` (preprocessing fitted on the training split), `logreg.joblib` and `random_forest.joblib`. Download them into the `models/` folder (excluded by `.gitignore`). Re-running the training scripts reproduces the committed predictions in `results/predictions/` exactly.

## 9. Repository structure

```text
SpillThePaper/
├── README.md
├── .gitignore
├── requirements.txt
├── config/              # filter rules, assessment rules, subject name mapping
├── data scripts/        # Phase 2 code (steps 01-08)
├── src/
│   ├── data/            # Phase 3: questions, topics, supervised dataset
│   ├── models/          # Phase 3: preprocessing, baselines, MLP, tuning, predictions
│   └── evaluation/      # Phase 4: metrics, baselines, error analysis, bias check
├── tests/               # unit tests (including the dataset leakage test)
├── docs/                # dataset schema, data settings, models, evaluation
├── data/
│   ├── raw/             # downloaded papers        (NOT in Git)
│   ├── interim/         # extracted text, questions (NOT in Git)
│   └── processed/       # small result tables (CSV); model/ holds train, val, test
├── models/              # model weights            (NOT in Git, see section 8)
├── results/             # tuning, predictions, metrics, error analysis, bias check
├── eda/                 # exploratory charts (PNG)
├── logs/                # execution logs (collection, extraction, cleaning, EDA)
└── report/              # report sources and PDFs; report/figures holds the figures
```

## 10. Setup

Requirements: Python 3.10 or newer and Git. The results in the reports were produced with Python 3.14, scikit-learn 1.9.1 and PyTorch 2.15.0 (CPU).

```powershell
git clone https://github.com/Minahilbasalat/SpillThePaper.git
cd SpillThePaper
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

If a package such as `torch` cannot be installed on your Python version, create the environment with an older Python (for example `py -3.12 -m venv venv`).

Check the installation:

```powershell
python -m pytest tests -q
```

## 11. Running the code

Run everything from the project root with the virtual environment active.

### 11.1 Models and evaluation (no raw data needed)

The committed tables in `data/processed/model/` are the inputs, so the Phase 2 download is not required.

```powershell
python -m src.models.baselines          # logistic regression and random forest, tuned on validation
python -m src.models.tune               # random search for the MLP (40 configurations x 3 seeds)
python -m src.models.train_mlp          # final MLP with the selected configuration
python -m src.models.export_predictions # results/predictions/<model>_<split>.csv
python -m src.models.plots              # architecture, curves and tuning figures

python -m src.evaluation.baselines      # rule baselines
python -m src.evaluation.evaluate       # metrics with intervals, ROC, PR, calibration, confusion figures
python -m src.evaluation.error_analysis
python -m src.evaluation.bias_check
python -m src.evaluation.contribution_table   # run on main after all branches are merged
```

Details: `docs/models.md` and `docs/evaluation.md`.

### 11.2 Rebuilding the dataset (needs the Phase 2 outputs)

These scripts read the extracted text in `data/interim/text/` produced by the Phase 2 pipeline (section 11.3).

```powershell
python -m src.data.build_questions      # segment papers into questions
python -m src.data.topics               # per-subject topics (training years only)
python -m src.data.build_dataset        # train, validation and test in data/processed/model/
python -m src.data.make_dummy_dataset   # random dummy data with the same schema, for development only
```

### 11.3 Phase 2 scripts

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
* The archive keeps growing, so a new run on the latest commit can give slightly different numbers than the reports. The commit used is recorded in the logs.
* Step 5 reads `config/subject_aliases.yaml` if it exists. Check it with `python "data scripts/check_aliases.py"`.

## 12. Key numbers

**Phase 2 (Deliverable 1)**, from the logs in `logs/` and the tables in `data/processed/`:

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

**Phases 3 and 4 (Deliverable 2):**

| Item | Value |
|---|---|
| Papers with readable text and a known year | 1,638 |
| Questions segmented | 7,780 (from 1,575 papers) |
| Subjects with enough history / topics | 28 of 105 / 220 |
| Supervised dataset | 2,290 rows (1,421 train, 301 validation, 568 test) |
| MLP parameters | 353 |

## 13. Limitations

* About one third of exam papers (32.6%) have no extractable text (scans and photographs), most often finals. 24 subjects have no readable paper.
* Topics come from unsupervised clustering with low silhouette scores (median 0.097); some clusters capture generic wording instead of course content, and the labels inherit this noise.
* Only 28 of 105 subjects have enough history for the models; the test split contains almost only theory subjects and no topic that never appeared before.
* The validation split has 36 target exams and the test split 67, so small metric differences are noise.
* The positive rate falls over time (0.473, 0.392, 0.347) and predicted probabilities are too high.
* A simple history rule ("appeared last year") ranks topics about as well as the learned models.
* Subject coverage is uneven (median 11 papers per subject; 31 subjects have fewer than five).
* Year, term, assessment type and number are parsed with rules from paths and headers; accuracy on a manually labelled sample has not been measured. The same holds for question segmentation.
* Total marks and question counts are estimates from text patterns and are missing for about a third of readable papers.
* Solution files without a plain companion are kept (flagged), so some question text may repeat.
* No professor-specific analysis (no reliable instructor information for papers or quizzes).
* The source archive is community-maintained and may contain mislabelled or incomplete files. Course outlines come from different terms, so none can be assumed to be the current syllabus.

## 14. Team

| Name | Roll number | Main modules |
|---|---|---|
| Minahil Basalat | 24L-3082 | `src/data`, dataset and settings documentation, integration |
| Iman Abid | 24L-0707 | `src/evaluation`, evaluation documentation |
| Khadija Rao | 24L-3042 | `src/models`, model documentation |

The full commit-to-module mapping is in `results/contribution_table.csv` and `results/contribution_commits.csv`, and in the Deliverable 2 report.

## 15. Acknowledgements and data notes

* Past papers and outlines come from the community archive *Academic-Time-Machine*; credit to its maintainers and contributors.
* Raw files are not redistributed here. Course outlines may name instructors; instructor names are not used or published in this project.