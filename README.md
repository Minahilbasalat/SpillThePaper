# SpillThePaper

**From past papers to a predicted, solved and personalised exam.** An AI study tool for FAST-NUCES students: it learns from the archive of past quizzes, mids and finals, forecasts which topics are likely to come, generates a tentative paper with a realistic structure and timer, shows solutions, grades your own answers from photos, finds your weak areas and builds a personalised follow-up paper.

AI course project, **Track A (Application Development)**. It is a study aid and does **not** guarantee what will appear in a real exam.

> **Status: all phases complete (Deliverables 1, 2 and 3).** Hosted app: _add your link here after deployment_.

---

## 1. What it does

| Page | What you can do |
|---|---|
| **Home** | Overview and entry to the three tools |
| **Past papers** | Browse all 130 subjects, grouped into Final, Mid 1, Mid 2 and Quiz n; open the original file on GitHub or read readable papers as questions |
| **Upcoming paper** | Pick a subject and exam, optionally upload recent papers and a syllabus, set the number of questions and marks; get a topic forecast and a predicted paper (likelihood per question, "How this paper was made", timer, exam mode, solved paper, PDF download) |
| **My results** | Upload photos of your answers, review the marks, see weak and strong areas and your history, and generate a **personalised paper** |

Inference is real time: the models are loaded once and a paper is generated in well under a second. Optional language-model steps (AI-written questions, AI solutions, photo grading) need an API key; without one the app still works with real, tidied past questions and archived solutions.

## 2. How it works

```text
Archive (GitHub) -> collect, clean, OCR -> question bank (10,538 questions: marks, type, difficulty)
   -> per-subject topics (TF-IDF + KMeans) -> topic-appearance dataset (time-split, leakage-safe)
   -> models (logistic regression, random forest, PyTorch MLP) -> Predictor + blueprints
   -> Paper generator -> solutions / PDF -> grading, weak areas, personalised paper -> Streamlit UI
```

* **Prediction task.** One row = (subject, topic, target exam); label = does the topic appear? Features use only earlier years.
* **Generator.** Structure (questions, marks, type and difficulty mix) comes from real past papers; topics come from the model, with about 20% reserved for less-likely areas. Questions are real past questions chosen for the predicted topics; an optional AI step rewrites them as new similar questions, with the real tidied question as the fallback.
* **Solutions.** Archived solutions first; missing ones are written by the AI in the background and cached.
* **Personalisation.** Weakness from your latest attempt (70%) and history (30%) tilts about half of the main questions toward weak topics.

## 3. Results

Time-based split: train 2014-2022 (1,940 rows), validation 2023 (383), test 2024-2026 (1,142 rows, 138 exams). Thresholds are chosen on validation; intervals are 95% cluster-bootstrap.

| Model | AUC-ROC (test) | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.850 [0.823, 0.876] | 0.630 | 0.800 | 0.705 |
| Random forest | **0.871** [0.848, 0.892] | 0.673 | 0.822 | 0.740 |
| MLP (PyTorch, 12-16-8-1, 353 parameters) | 0.857 [0.831, 0.881] | 0.577 | 0.901 | 0.703 |
| Rule: appeared last year | 0.836 [0.813, 0.860] | 0.584 | 0.770 | 0.665 |
| Raw topic rate | 0.834 [0.808, 0.860] | 0.545 | 0.901 | 0.679 |
| Rule: count of at least 5 | 0.824 [0.797, 0.852] | 0.458 | 0.980 | 0.625 |

**Honest summary.** The learned models rank slightly better than simple history rules and are clearly more precise than the "appeared at least five times" rule, but the intervals still overlap with the best rule, so we do not claim a statistically separated gain. Quizzes are the weakest exam type, probabilities are over-estimated (read them as rankings), and the test split has no topic that never appeared before. Full analysis, error analysis and bias check: see the final report.

## 4. Quick start

Requirements: Python 3.10+ and Git (results were produced with Python 3.14, scikit-learn 1.9.1, PyTorch 2.15.0 CPU). OCR needs the Tesseract program installed (only for rebuilding the data).

```powershell
git clone https://github.com/Minahilbasalat/SpillThePaper.git
cd SpillThePaper
python -m venv venv
venv\Scripts\Activate.ps1          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Get the trained models and the question bank** (not stored in Git; see section 6):

```powershell
python scripts/get_runtime.py
```

**Run the tests and the app:**

```powershell
python -m pytest tests -q
streamlit run app/Home.py
```

**Optional AI features.** Create a file named `.env` in the project folder (it is git-ignored; never commit keys):

```text
GEMINI_API_KEY=your-key
GROQ_API_KEY=your-groq-key        # optional second provider used as a fallback
GEMINI_MODEL=gemini-3.8-flash     # optional, any provider has GROQ_MODEL / MISTRAL_MODEL / ANTHROPIC_MODEL
```

Check it: `python -m src.inference.llm`. If the main model hits a quota error, the client tries the next model of the same provider and then every other provider with a key.

## 5. Reproducing the data and models

All commands run from the project root. The committed tables in `data/processed/model/` are enough to retrain models; the raw download is only needed to rebuild the data.

```powershell
# Phase 2: data (code in "data scripts/")
python "data scripts/01_discover.py"
python "data scripts/02_filter.py"
python "data scripts/03_download.py"        # about 2.4 GB, resumable
python "data scripts/04_extract.py"
python "data scripts/04b_ocr.py"            # OCR of scanned papers (needs Tesseract)
python "data scripts/05_build_papers.py"
python "data scripts/06_clean.py"
python "data scripts/07_features.py"
python "data scripts/08_eda.py"

# Everything after extraction in one go (questions, topics, dataset, models, evaluation)
python scripts/rebuild_all.py               # supports --from and --to

# Phases 3-4 step by step
python -m src.models.baselines
python -m src.models.tune                   # random search, 40 configurations x 3 seeds
python -m src.models.train_mlp
python -m src.models.export_predictions
python -m src.models.plots
python -m src.evaluation.evaluate
python -m src.evaluation.error_analysis
python -m src.evaluation.bias_check

# Figures of the final report
python scripts/make_report_figures.py
```

## 6. Model weights and runtime bundle

Large binary files are not in Git. Weights (`mlp_best.pt`, `preprocess.joblib`, `logreg.joblib`, `random_forest.joblib`) are hosted here:

**https://drive.google.com/drive/folders/1IQb9xincr7PjG4kQvqEmj8jBBezRqQRZ?usp=drive_link**

The application also needs the **runtime bundle** (`runtime_bundle.zip`, 43 files, about 5 MB: trained models, 37 per-subject topic models and the question bank). The bundle is `runtime_bundle.zip`, hosted in the same Drive folder. Its link is stored in `config/runtime_url.txt`, so `python scripts/get_runtime.py` downloads and unpacks it automatically. The question bank is derived from exam papers and is therefore not redistributed inside the repository.

## 7. Repository structure

```text
SpillThePaper/
├── README.md, requirements.txt, .gitignore
├── app/                 # Streamlit app: Home.py, pages/ (Past papers, Upcoming paper, My results), components/
├── config/              # filter rules, assessment rules, subject aliases
├── data scripts/        # Phase 2: collection, extraction, OCR, cleaning, features, EDA
├── src/
│   ├── data/            # OCR, question segmentation, cleaning, question features, topics, dataset
│   ├── models/          # preprocessing, baselines, MLP, tuning, plots
│   ├── evaluation/      # metrics, baselines, error analysis, bias check, contribution table
│   └── inference/       # backend: library, predictor, syllabus, uploads, generator, solutions,
│                        #   LLM client, question rewriting, grading, results, personalised paper, PDF export
├── scripts/             # rebuild_all, get_runtime, make_runtime_bundle, diagnose, make_report_figures
├── tests/               # 200+ automated tests (no network or API key needed)
├── docs/                # dataset schema, data settings, models, evaluation, backend API
├── data/                # raw/ and interim/ are NOT in Git; processed/ holds small tables
├── models/              # weights (NOT in Git, see section 6)
├── results/             # tuning, predictions, metrics, error analysis, bias check
├── eda/, logs/          # exploratory charts and run logs
└── report/              # LaTeX reports (Deliverables 1, 2 and the final report) and figures
```

## 8. Testing

`python -m pytest tests -q` runs 200+ tests: no leakage (target-year changes never alter features), serving features equal training features, saved topic models reproduce stored topic ids, metrics and bootstrap, question segmentation and cleaning, syllabus parser, generator (structure, determinism), solution service and background jobs, the LLM client (mocked HTTP, fallback), grading and personalisation, and every Streamlit page.

## 9. Limitations

* Predicted questions are mostly real past questions; probabilities are rankings and over-estimated; difficulty is a proxy.
* Learned models are not statistically separated from the best simple rule; no cold-start (never-seen topic) evidence.
* Topics are keyword clusters (median silhouette 0.093), not syllabus topics.
* Only 37 of 127 subjects have enough history; 349 clean papers (mostly handwritten) stay unreadable.
* Marks parse for about half of the questions; question-type accuracy has not been measured against hand labels.
* Archived solutions can be partial; AI solutions and photo grading must be checked by the student; free API plans have rate limits.
* The generator has not yet been scored paper by paper against real exams.
* No instructor-specific analysis (no reliable instructor data).

## 10. Team

| Name | Roll number |
|---|---|
| Minahil Basalat | 24L-3082 |
| Iman Abid | 24L-0707 |
| Khadija Rao | 24L-3042 |

The commit-to-module mapping is in `results/contribution_table.csv` (run `python -m src.evaluation.contribution_table`).

## 11. Acknowledgements and data notes

Past papers and outlines come from the community archive [Academic-Time-Machine](https://github.com/saleha-muzammil/Academic-Time-Machine) (pinned commit `e0e197577432957a5fad0f76d2ab92d78f48e1d8`); credit to its maintainers and contributors. Raw papers are not redistributed; scripts download them into the git-ignored `data/raw/`.