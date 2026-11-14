# UI handoff - build pages on REAL backend functions only

Everything below runs on the real question bank and the real trained models. There is no
demo data. A page is built only when the backend function it needs exists.

## Setup (each teammate)
    git clone https://github.com/Minahilbasalat/SpillThePaper
    cd SpillThePaper
    git checkout feature/ocr-and-v2
    git checkout -b ui/<your-name>
    python -m venv venv ; venv\Scripts\Activate.ps1
    pip install -r requirements.txt
Copy `runtime_bundle.zip` (shared on Drive by Minahil) into the project folder and unzip it
THERE (keep the folder names: it creates models/ and data/interim/).
    python -m pytest tests -q        # must pass before you start
    streamlit run app/Home.py

## Rules
1. Pages import only `from common import get_predictor, get_library` (see app/common.py)
   and begin with `import bootstrap`. Never read CSV files in a page.
2. Never hard-code subject names; ask the backend (`lib.subjects()`).
3. Honesty in the UI: probabilities are RANKINGS, not exact chances (they are
   over-estimated); difficulty is a PROXY; subjects not supported must show the reason
   (`lib.overview(s)["reason_if_unsupported"]`); topics with `few_history=True` are
   labelled low-evidence.
4. Put each page in app/pages/ as `N_Name.py`. Do not edit src/ (ask Minahil for backend
   changes).

## Backend you can call today
Predictor (models):
    pred.subjects() -> list[str]
    pred.predict(subject, assessment_type, year, model="logreg") -> DataFrame
        columns: topic_id, topic_label, probability, rank, band (Likely/Possible/Unlikely),
                 topic_count_prior, topic_rate_prior, years_since_last,
                 appeared_last_year, few_history        (df.attrs: prior_papers, threshold)
        raises LowConfidenceError (src.inference.errors) for unsupported subjects
    pred.history(subject, topic_id) -> DataFrame(year, assessment_type, appeared)
    pred.model_names() -> ["logreg", "random_forest", "mlp"]
Library (data):
    lib.subjects() ; lib.overview(subject) -> dict ; lib.topics(subject) -> DataFrame
    lib.questions(subject, topic_id=None, assessment_type=None, qtype=None, n=10) -> DataFrame
    lib.blueprint(subject, "quiz|mid|final") -> dict   (typical questions, marks, type shares)
    lib.past_papers(subject) -> DataFrame(file_id, year, assessment_type, questions, marks)
    lib.paper(file_id, duration_minutes=None) -> paper dict (structure in library.py)

## Pages to build NOW
1. Predict topics: subject + exam type + year + model -> ranked topics with probability bars,
   band colours, evidence flag, topic history, and 3 real example questions per topic.
2. Subject explorer: overview, typical paper (blueprint), topic list, question browser.
3. Practice a real past paper: choose subject and paper -> questions in order with a
   countdown timer (default 30 / 60 / 180 min for quiz / mid / final, editable, start/pause/
   reset), a notes/answer box per question, "show solution" only where `solution` is not None.

## Pages that unlock later (design the layout, wire it when the function exists)
| Page | Function | Unlocked by |
|---|---|---|
| Upload syllabus | parse syllabus -> syllabus topics | Stage 3 |
| Predicted paper + solutions | generate_paper(...) -> paper dict (same structure) | Stage 5 |
| Submit photo of answers | grade_attempt(...) -> marks + feedback | Stage 6 |
| Weak areas + new personalised paper | weak_areas(...), personalised_paper(...) | Stage 7 |
| Add my own subject | add_subject(...) | Stage 8 |
