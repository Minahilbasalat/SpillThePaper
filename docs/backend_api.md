# Backend API (all REAL data and models - no demo data anywhere)

Setup on a teammate's computer: clone the repo, `git checkout feature/ocr-and-v2`, create a venv,
`pip install -r requirements.txt`, then `python scripts/get_runtime.py <URL>` (URL of
runtime_bundle.zip given by Minahil; it unpacks `models/` and `data/interim/` into the project),
then `python -m pytest tests -q` must pass.
Why a bundle: the trained models and the question bank are not stored in Git (weights are hosted
externally; the question bank is derived from exam papers we do not redistribute in the repo).
Scanned PDFs / photos need Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki; set the
environment variable TESSERACT_CMD if it is not on PATH). AI solutions need ANTHROPIC_API_KEY
(optional; without it only archived solutions are shown).

Pages import only these cached helpers (app/common.py) and begin with `import bootstrap`:
    from common import get_predictor, get_library, get_generator, get_solutions

Exam values used everywhere: "quiz", "mid1", "mid2", "final"  ("mid" = unnumbered mid).

## Library  (get_library())  - browsing real data
all_subjects() -> DataFrame[subject, papers, readable, first_year, last_year, supported, kinds]
    `kinds` is a ready string like "Final · Mid 1 · Mid 2 · Quiz"; `supported` = prediction available
exam_kinds(subject) -> [{"kind": "final|mid1|mid2|mid|quiz1|quiz2|...|other", "label": "Mid 1", "papers": n}]
papers(subject, kind=None) -> DataFrame[file_id, year, term, exam_kind, label, filename,
    includes_solution, has_text, total_marks, questions, original_url, download_url]
    original_url = view the REAL paper on GitHub, download_url = raw file; questions>0 means the
    paper can also be read as text with paper(file_id)
paper(file_id, duration_minutes=None) -> paper dict (below); KeyError if its text was not readable
overview(subject) -> dict(supported, reason_if_unsupported, papers, readable_papers, questions,
    first_year, last_year, papers_by_kind, topics)
blueprint(subject, exam) -> dict(questions, total_marks, difficulty_mix{easy,medium,hard},
    question_type_shares, mean_bloom_level, papers_used, scope, duration_minutes)
questions(subject, topic_id=None, assessment_type=None, qtype=None, n=10) -> DataFrame
topics(subject) -> DataFrame[topic_id, topic_label, questions]
from src.inference.library import default_duration   # quiz 30, mid 60, final 180 minutes

## Predictor  (get_predictor())
subjects() -> supported subject names
predict(subject, exam, year, model="logreg", extra_papers=None, extra_incidence=None) -> DataFrame
    [topic_id, topic_label, probability, model_probability, rank, band(Likely/Possible/Unlikely),
     topic_count_prior, topic_rate_prior, years_since_last, appeared_last_year, few_history,
     examined_this_year, examined_in]   df.attrs: prior_papers, prior_years, threshold,
     phase_adjusted, uploaded_papers
    raises src.inference.errors.LowConfidenceError (message = reason) for unsupported subjects
history(subject, topic_id) -> DataFrame[year, assessment_type, appeared]
model_names() -> ["logreg", "random_forest", "mlp"]

## Syllabus, uploads
from src.inference.syllabus import parse_syllabus, default_scope
parse_syllabus(text) -> {"items": [str], "mid1_boundary": int|None, "mid2_boundary": int|None}
default_scope(parsed, exam) -> list of item indexes the exam probably covers
from src.inference.documents import extract_text
extract_text(filename, bytes) -> {"text", "method", "warning"}   # pdf/docx/txt/jpg/png;
    raises ValueError (unsupported/too large) or RuntimeError (OCR not installed)
from src.inference.uploads import ingest_papers
ingest_papers(subject, [{"name", "data": bytes, "year": int, "assessment_type": "quiz|mid|final",
    "number": int|None}]) -> UploadResult(papers, incidence, questions, warnings, n_papers)
    pass the result as `extra=` to generate(); uploads are used for that prediction only

## Generator  (get_generator())
generate(subject, exam, year, model="logreg", syllabus_items=None, extra=None, n_questions=None,
         total_marks=None, explore_share=0.2, strict_syllabus=False, seed=None) -> paper dict
    reproducible: same inputs + seed = same paper; change `seed` for "another version"

## Solutions and PDF  (get_solutions())
solve_paper(paper, allow_ai=True) -> list aligned with paper["questions"]:
    {"number", "text" (markdown, may contain ``` code blocks), "marking_scheme" (str|None),
     "source": "archive"|"ai"|"none", "note" (ALWAYS show it)}
    call it with allow_ai=False first (instant, archive only), and with allow_ai=True when the
    student presses "Generate missing solutions with AI" (slow: a few seconds per question).
ai_available() -> bool       (False when ANTHROPIC_API_KEY is not set)
solve(subject, question_dict, allow_ai=True) -> one entry (same shape)
from src.inference.export import paper_to_pdf
paper_to_pdf(paper, solutions=None) -> bytes   (for st.download_button; pass solve_paper output
    to include solutions)

## The paper dictionary (past papers AND predicted papers)
{"title", "subject", "assessment_type", "exam_kind", "year", "source": "past_paper"|"predicted",
 "duration_minutes", "total_marks",
 "questions": [{"number", "text", "marks", "qtype", "difficulty": "easy|medium|hard",
                "topic_id", "topic_label", "solution", "solution_source",
                # past papers only:   "low_quality" (True = garbled scan: warn the student)
                # predicted only:     "marks_note", "reason", "explore", "topic_probability",
                #                     "likelihood" (1-5), "leetcode_style" (bool),
                #                     "syllabus_match", "source"}]}
predicted papers also carry: "notes" (list[str]; ALWAYS show them), "blueprint", "topics"
(the prediction DataFrame), "realised": {"difficulty_mix", "coding_questions",
"problem_solving_questions", "explore_questions"}

## What must be said honestly in the UI
- Predicted questions are REAL PAST questions chosen for predicted topics, not new questions.
- Probabilities and the "likely" squares are rankings, not exact chances. Difficulty is a proxy.
  Syllabus matching compares words and is approximate. Mid 1 / Mid 2 weighting needs 3+ earlier
  papers of that kind.
- Solutions: "archive" ones can be partial; "ai" ones must carry "AI-generated: verify".
- Unsupported subjects: show `reason_if_unsupported`; past papers can still be browsed.
