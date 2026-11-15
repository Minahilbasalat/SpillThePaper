# Prompt for Iman's Claude  (copy everything below into Claude, and attach the mockup PDF)

You are building the Streamlit UI of "SpillThePaper", a FAST-NUCES exam-preparation app. The backend
is finished and runs on real data and trained models. I attach `SpillThePaper - Streamlit UI Mockup.pdf`:
the app must look EXACTLY like it (colours, fonts, borders, hard shadows, tilted sticky notes, stamps,
layouts, spacing, wording of headings and buttons). Compare your running app with the PDF page by page
and keep fixing until they match. Do not change anything in `src/`. Never invent data: the numbers in
the mockup ("24 papers", "Fall 2025", "sample data" badges, the Data Structures questions...) are
placeholders; every number, paper, question and solution on screen must come from the backend. Remove
the "sample data" badges.

Build these screens: Home, Past papers (subject grid + subject page), Upcoming paper (form, predicted
paper, exam mode, solved paper) and a "My results" page that is only a styled "Coming soon" card
(upload your solution, weak areas and personalised paper are NOT built yet; any button that leads there
is disabled with the caption "Coming soon").

## 0. Get the project running
1. `git clone https://github.com/Minahilbasalat/SpillThePaper`, `cd SpillThePaper`
2. `git checkout feature/ocr-and-v2`, `git checkout -b ui/iman`
3. `python -m venv venv`, activate, `pip install -r requirements.txt` (Streamlit >= 1.39 is required:
   `st.container(key=...)`, `st.fragment(run_every=...)`, `st.link_button`, `st.pills`)
4. Git does not hold the trained models and the question bank (weights are hosted externally and the
   question bank is derived from exam papers). Minahil gives you a URL: `python scripts/get_runtime.py <URL>`
   unpacks `models/` and `data/interim/` into the project.
5. Optional: Tesseract OCR (UB-Mannheim build, set `TESSERACT_CMD`) for scanned uploads; set
   `ANTHROPIC_API_KEY` in your terminal to switch on AI-written solutions.
6. `python -m pytest tests -q` must pass before you change anything.
7. READ `docs/backend_api.md` completely. It is the contract: function names, columns, the paper
   dictionary, exam values quiz/mid1/mid2/final. Look at `app/Home.py`, `app/common.py`, `app/bootstrap.py`:
   every page starts with `import bootstrap` and uses
   `from common import get_predictor, get_library, get_generator, get_solutions`.

## 1. Design system (read from the mockup)
- Colours: sage green page background #BED3B4, cream cards #FCF6E4, ink #111111, lime accent #D9FF3A
  (active nav text, black-button text, big numbers), red #D4281A (hard shadow of black buttons, the
  LEAKED stamp), yellow #FFD542, pink #FF9CC7, blue #8FD6FF, orange #FFB347. Sample the exact colours
  from the PDF if they differ.
- Fonts (Google Fonts via CSS @import): headings = heavy grotesk "Archivo Black"; body/UI = "Space
  Grotesk"; typewriter text inside the paper cards = "Special Elite" (or "Courier Prime", whichever is
  closest); handwritten sticky notes = "Caveat" bold. Section labels are small uppercase typewriter text
  with letter-spacing ("PREDICTION DESK", "THE ARCHIVE", "DECLASSIFIED").
- Neo-brutalist boxes: 2px solid #111 border, hard offset shadow `6px 6px 0 #111` (cards) or
  `6px 6px 0 #D4281A` (black buttons and the black panels), radius about 4px (nav pills 10px).
- Tilted elements: hero card -1deg, sticky notes +-1.5deg, stamps -6deg. Yellow translucent "tape"
  strip on top-left of big cards. Dashed 2px separators. A key word in each page title has a yellow
  highlight behind it ("Spill the paper **before** it spills you", "Past **papers**", ...).
- Chips: rectangular, 2px border, pink for syllabus topics (with x), yellow for selected options.
- Top navigation on every page: logo "SPILL THE PAPER" left, pills on the right: Home, Past papers,
  Upcoming paper, My results (active pill = black with lime text). Hide Streamlit's own sidebar/menu/footer;
  build the nav with `st.page_link` styled by CSS. Exam mode replaces it with "EXAM MODE · subject exam"
  and an "Exit exam" button.
- Implement with `app/theme.py` (CSS injected on every page via `st.markdown(unsafe_allow_html=True)`),
  `.streamlit/config.toml` (base colours) and `st.container(key="name")` + CSS on `.st-key-name` to style
  real Streamlit widgets. Use HTML only for static visuals; keep buttons/inputs as real Streamlit widgets.
  Desktop first, but columns must stack sensibly on a phone.

## 2. Files
- `app/Home.py`, `app/pages/1_Past_Papers.py`, `app/pages/2_Upcoming_Paper.py`, `app/pages/3_My_Results.py`
- `app/theme.py` (CSS + helpers: nav, sticky note, stamp, tape card, chip, badge, highlighted title)
- `app/components/timer.py` (feature: timer), `app/components/paper_view.py` (question rows, badges)
- `tests/test_app.py`: smoke tests with `streamlit.testing.v1.AppTest` that run every page and assert no
  exception (use real data; skip if the runtime bundle is missing)
Each page with several screens uses a small state machine in `st.session_state` (no extra pages).

## 3. Home (mockup page 1)
Hero card (tilted, tape, LEAKED stamp, "CLASSIFIED - FAST EXAM INTEL", the headline, the paragraph, the
two black redaction bars as decoration, buttons "Predict my paper ->" (go to Upcoming paper) and "Browse
past papers"). Right: three sticky notes (Past papers, Upcoming paper, Upload your solution - this one leads
to My results = Coming soon). Bottom black strip with the five numbered steps 01-05 as in the mockup.

## 4. Past papers (mockup pages 2-3), data = Library
Grid screen: title "Past papers" + search box "Search a subject" (filters `lib.all_subjects()`, case
insensitive), 4-column grid of folder-tab cards (tab colour cycles yellow/pink/blue/orange): subject name,
`kinds` string, highlighted "{papers} papers". Sort by paper count (descending) and show 24 with a
"Show more" button. Pink banner at the bottom: "Can't find your subject?" + button "Predict the next paper ->".
Subject screen: breadcrumb "Past papers / Subject / Final", title with the last word highlighted, left
column of big exam-kind buttons from `lib.exam_kinds(subject)` (label + "N papers", active = black with
lime text and red shadow), right card "Final papers" (taped label) listing `lib.papers(subject, kind)`:
file-type badge from the extension (PDF/DOC/JPG), title "Final - Fall 2025" (label, term, year), small text
"{questions} questions - {total_marks} marks" (omit what is unknown; if `questions == 0` write "scan not
readable here"), buttons View (`st.link_button` to `original_url`), Download (`download_url`) and, only when
`questions > 0`, "Read and practise" which opens the paper (via `lib.paper(file_id)`) in the exam-mode
screen below (timer optional). Show "includes solution" badge when `includes_solution`; show a small
"scan may be garbled" note for questions with `low_quality`. Right: yellow sticky note "Studied these? Now
see what the NEXT final could look like." and the black button "Predict the next one ->" which opens the
Upcoming paper page with this subject and exam preselected (`st.session_state`).

## 5. Upcoming paper (mockup pages 4-7)
Form screen, title "Predict the next paper" (label "PREDICTION DESK"). Left column, four numbered cards
(black square number): 
1. Subject and exam: subject quick-chips (first few of `pred.subjects()` by paper count) plus "More
   subjects" (searchable `st.selectbox` over `lib.all_subjects()`; subjects that are not `supported` show
   `lib.overview(subject)["reason_if_unsupported"]` in a styled note with a link to Past papers); exam as four
   big buttons Final / Mid 1 / Mid 2 / Quiz (values final, mid1, mid2, quiz; selected = yellow with shadow);
   a year selector (current year and next) kept small.
2. "Upload recent papers" (blue OPTIONAL badge): dashed drop zone ("Drop PDFs, photos or docs here / More
   recent papers means a sharper prediction.") = `st.file_uploader` (pdf, docx, txt, jpg, jpeg, png,
   multiple). For each file ask year, type (quiz/mid/final), number (optional), then call
   `ingest_papers(subject, files)`; show yellow chips "name.pdf check" per accepted file, the number of
   questions found, and every message in `result.warnings`. Keep the UploadResult in session state.
3. "Syllabus for this paper": pink removable chips (x) for the topics, a text input "Add a topic, e.g.
   Hashing" with an "Add" button, and "Or upload syllabus PDF" (`extract_text` then `parse_syllabus`; the
   items become chips, pre-selected with `default_scope(parsed, exam)`; the student removes what is not
   included). Chips selected = `syllabus_items`. Add a small "strict syllabus" checkbox and the sentence
   "Syllabus matching compares words, so it is approximate."
4. "Timer" (blue OPTIONAL badge): chips Timer ON/OFF, 30 min, 1 hr, 90 min, 3 hrs, Custom (number input).
   The default chip follows the exam (quiz 30, mid 60, final 180; `default_duration(exam)`).
Right column: sticky note "Difficulty, the FAST way" with a 3-segment bar (easy blue, medium cream, hard
pink) and percentages from `lib.blueprint(subject, exam)["difficulty_mix"]`, shown as soon as subject and exam
are chosen; a pink note "Also included: a few less-predicted syllabus areas, so surprise questions do not
catch you out."; the big black button "Generate predicted paper ->" (red shadow). It calls
`get_generator().generate(subject, exam, year, syllabus_items=..., extra=...)` inside `st.spinner`, catches
`LowConfidenceError` and shows its message. An "Advanced" expander holds the model (`pred.model_names()`),
number of questions, total marks and the share of less-predicted questions.
Predicted paper screen: tilted hero card with tape + LEAKED stamp: label "TENTATIVE PAPER - SUBJECT - EXAM",
title "What your paper probably looks like", "Time allowed ... - Total marks ...", dashed rule, then each
question: big "Q1", the text (typewriter), chips: "{marks} marks", difficulty badge (Easy / Medium / Hard),
"Likely" with five squares filled according to `likelihood` (1-5), blue "LeetCode-style" chip when
`leetcode_style`. A question with `explore == True` is redacted (two black bars) with the red outlined tag
"Less-predicted area" and a black "Declassify this question" button that reveals it (session state). Right: a
yellow sticky note "Difficulty" written from `paper["realised"]` (for example "{n} coding question(s),
{m} problem-solving question(s), {k} surprise") and a white card "YOUR NEXT MOVE": black "Start with timer
hh:mm:ss" (opens exam mode), pink "See solution" (opens the solved paper), blue "Upload my solution"
(disabled, Coming soon), link "Download as PDF" (`st.download_button` with `paper_to_pdf(paper)`). Below, in a
styled expander "How this paper was made", ALWAYS show every line of `paper["notes"]`, and a plotly bar chart
of `paper["topics"]` (probability per topic coloured by `band`, a marker for `examined_this_year`).
Exam mode screen: left cream card "SOLVE ON PAPER, THEN PHOTOGRAPH YOUR ANSWERS" with the paper title and the
questions with "{marks} marks" tags only (no hints). Right: black panel "TIME LEFT" with huge lime hh:mm:ss, a
progress bar "of 03:00:00 - 10% used", buttons Pause/Resume and Hide clock; a yellow sticky tip; red button
"Finish and upload my solution ->" (disabled, Coming soon) plus a normal "Finish exam" that returns to the
paper. The timer: `@st.fragment(run_every="1s")`, state in `st.session_state` as timestamps (remaining
seconds when paused), amber at <= 10 min and red at <= 5 min, a clear "Time is up" banner, minutes default
from `paper["duration_minutes"]` and editable.
Solved paper screen ("DECLASSIFIED", title "The **solved** paper", buttons "<- Back to paper" and "Upload my
solution" (disabled)): one card per question: "Q1 - {topic_label}", marks badge, LeetCode-style badge if
applicable; the solution as markdown from `get_solutions().solve_paper(paper, allow_ai=False)` first
(instant, archive only); fenced code blocks must render as the dark code box of the mockup; show a small
badge from `source` ("From the archive" / "AI-generated") and always the `note`; if `source == "none"` show
the note in a friendly box. Provide the button "Generate missing solutions with AI" (calls `solve_paper(...,
allow_ai=True)` in a spinner) only when `get_solutions().ai_available()`; otherwise a caption "AI solutions
are switched off (ANTHROPIC_API_KEY is not set)". Right column: pink/yellow sticky notes; show the
`marking_scheme` of a question when present, otherwise no marking note. Keep results in session state.

## 6. My results
Same nav and title style, one cream card: "Coming soon: upload photos of your answers, see your weak areas
and get a personalised paper." No fake scores.

## 7. Rules
- Only call functions documented in `docs/backend_api.md`. No CSV reading in pages; no hard-coded subjects.
- Be honest on screen: predicted questions are REAL PAST questions chosen for predicted topics; the likely
  squares and probabilities are rankings; difficulty is a proxy; AI solutions need checking.
- Cache with the helpers in `app/common.py`; never reload models on every rerun; every long action in
  `st.spinner`; friendly messages for empty states and errors (never a stack trace).
- Tidy code: small functions, type hints, no duplicated markup, no business logic in the UI.

## 8. Done when
`streamlit run app/Home.py` works on a clean clone; with a real subject (for example Computer Networks)
you can browse Mid 1, view/download an original, read one with the timer, generate a predicted Mid 1 with and
without an uploaded paper and a syllabus, declassify the less-predicted question, start exam mode, open the
solved paper, and download the PDF; all screens match the mockup; unsupported subjects and unreadable papers
show friendly messages; `python -m pytest tests -q` passes (including `tests/test_app.py`); commit in small
steps (`feat(ui): home page` ...) and push the branch `ui/iman`.
