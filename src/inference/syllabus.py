"""Turn a syllabus / course outline (any text a student pastes or uploads) into topic items,
and measure which past questions belong to it.

parse_syllabus(text)  -> {"items": [str], "mid1_boundary": int|None, "mid2_boundary": int|None}
default_scope(parsed, exam_kind) -> list of item indexes the exam probably covers
SyllabusMatcher(question_texts, items).scores() -> similarity (0..1) of each question to the
                                                   best matching selected item

The matcher is TF-IDF cosine similarity (no external model). It is evaluated in
tests/test_syllabus.py: questions of a subject must score clearly higher against that
subject's own outline than questions of other subjects do.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_F = re.I
START = re.compile(r"(?im)^\s*(course\s*outline|(?:tentative\s*)?weekly\s*schedule|tentative\s*schedule|"
                   r"lecture\s*(?:plan|schedule)|course\s*contents?|topics?\s*covered|"
                   r"list\s*of\s*topics|course\s*schedule|week\s*topics?|syllabus)[^\n]{0,40}$")
END = re.compile(r"(evaluation|grading|assessment\s*(?:scheme|plan)|course\s*policies|"
                 r"marks\s*distribution|grade\s*distribution|academic\s*integrity|"
                 r"plagiarism|attendance\s*policy)", _F)
ADMIN = re.compile(r"(e-?mail|credit\s*hours?|class\s*venue|class\s*meeting|office\s*hours?|"
                   r"instructor|pre-?requisites?|text\s*books?|reference\s*books?|"
                   r"course\s*website|google\s*classroom|program:|type:|sec\s*\w+:|"
                   r"edition|isbn|plo\s*\d|clo\s*\d|ta\s*name|office\s*location|\bname\s*:|location\s*:|"
                   r"objective\s*of\s*the\s*course|course\s*information|learning\s*outcomes?)", _F)
JUNK_LINE = re.compile(r"^(?:chapters?\s*[\d,\s&-]*|module|no\.?\s*of\s*lectures?|reference\s*text|"
                       r"supplement(?:ary)?\s*text.*|from\s+\w+.*|week|topics?|lectures?|lecture\s*details?|book|hours?|"
                       r"page\s*\d+\s*(?:of\s*\d+)?|[\d\s,.-]+|[-–•·*]+|total:?|"
                       r"\[?subject to.*|availability.*)$", _F)
# Course description forms (e.g. the NCEAC form) have no "Course Outline" heading: the topics sit in a
# table headed "Topics to be Covered ... Week | Lecture Details | Book Chapters" and end where the
# "Laboratory Projects" / "Class Time Spent" rows begin.
TABLE_STARTS = (re.compile(r"lecture\s*details(?:\s*book\s*chapters)?", _F),     # the column header: most precise
                re.compile(r"topics?\s+to\s+be\s+covered", _F))
TABLE_END = re.compile(r"(laboratory\s*projects|programming\s*assignments\s*done|class\s*time\s*spent|"
                       r"oral\s*and\s*written|instructor\s*name|reference\s*material|text\s*books?)", _F)
# Lines that are never a topic, wherever they appear: form headers, page headers/footers, grading
# weights, timings, names and titles.
NOISE = re.compile(r"(\bnceac\b|national\s+computing|accreditation|course\s+description\s+form|"
                   r"\binstitution\b|\bprogram\s*\(s\)|\bfast-nuces\b|\bbsc?s\b|\bbsse\b|"
                   r"course\s+(?:code|title|goals?|description|learning|instructor)\b|"
                   r"assessment\s+instruments|with\s+weights|\bhomework\b|catalog\s+description|mapping\s+of|\bclo\b|\bplo\b|"
                   r"\d+\s*%|\b\d{1,2}:\d{2}\b|"
                   r"\b(?:mon|tues?|wednes|thurs?|fri|satur|sun)days?\b|"
                   r"\b(?:dr|prof|mr|mrs|ms|engr)\.\s|\bdate\s*:|"
                   r"grading\s+scheme|minimum\s+requirement|class\s+participation|"
                   r"\(assume\b|half-hour|instruction\s+and|"
                   r"programming\s+assignments\s+done|laboratory\s+projects)", _F)
MID_MARK = re.compile(r"\bmid[\s-]*(?:term)?\b[\s-]*(?:exam(?:ination)?s?)?[\s:#-]*"
                      r"(?:no\.?\s*)?(1|2|i{1,2}|one|two)\b", _F)
FINAL_MARK = re.compile(r"\bfinal\s*exam", _F)
WEEK_PREFIX = re.compile(r"^(?:week|lecture|lec|module|unit|chapter|topic)\s*\d+\s*[:.\-–)]*\s*", _F)
BULLET = re.compile(r"^[\s•·*\-–\u2212\u2022\u25cf\u25aa)]+|^\(?\d{1,2}[.)]\s+")
STOP = sorted(set(ENGLISH_STOP_WORDS) | {"question", "questions", "marks", "mark", "write",
                                         "explain", "following", "given", "consider", "show",
                                         "state", "define", "describe", "discuss", "list",
                                         "find", "draw", "using", "use", "chapter", "week",
                                         "introduction", "basic", "basics", "concepts",
                                         "overview", "advanced", "topics", "lecture"})
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z+#]{1,}\b"


def _topic_like(line: str) -> bool:
    w = len(line.split())
    return 2 <= w <= 14 and not line.rstrip().endswith(".") and not ADMIN.search(line)


def _section(text: str) -> str:
    """The part of the document that lists the topics: the LAST heading (course outline,
    weekly schedule, ...) followed by at least 8 short topic-like lines before the grading
    section. Earlier headings are usually the title or the objectives."""
    best = None
    for m in START.finditer(text):
        e = END.search(text, m.end() + 40)
        chunk = text[m.start(): e.start() if e else len(text)]
        if sum(_topic_like(ln.strip()) for ln in chunk.split("\n")) >= 8:
            best = chunk
    if best is not None:
        return best
    # No "course outline" style heading: use the topics table of a course description form.
    for start in TABLE_STARTS:
        m = start.search(text)
        if m:
            e = TABLE_END.search(text, m.end())
            chunk = text[m.end(): e.start() if e else len(text)]
            if sum(_topic_like(ln.strip()) for ln in chunk.split("\n")) >= 5:
                return chunk
    return text


def parse_syllabus(text: str) -> dict:
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # a short comma/semicolon separated list pasted on one line
    list_mode = t.count("\n") < 3 and t.count(",") + t.count(";") >= 3
    if list_mode:
        t = re.sub(r"[;,]\s*", "\n", t)
    lines = [ln.strip() for ln in (t if list_mode else _section(t)).split("\n")]
    items, m1, m2 = [], None, None
    for ln in lines:
        ln = BULLET.sub("", ln).strip()
        if not ln:
            continue
        mid = MID_MARK.search(ln)
        if mid and len(ln.split()) <= 8:
            which = mid.group(1).lower()
            if which in ("1", "i", "one"):
                m1 = len(items)
            else:
                m2 = len(items)
            continue
        if FINAL_MARK.search(ln) and len(ln.split()) <= 6:
            continue
        ln = WEEK_PREFIX.sub("", ln).strip()
        ln = re.sub(r"^no\.?\s+of\s+", "", ln, flags=_F)
        if JUNK_LINE.match(ln) or ADMIN.search(ln) or NOISE.search(ln) or START.fullmatch(ln.strip(" :")) \
                or END.fullmatch(ln.strip(" :")) or re.match(r"^topics?\s+(?:to be covered|detail)", ln, _F):
            continue
        if ln.startswith("(") and len(ln.split()) <= 3:        # leftovers like "(Tentative)"
            continue
        words = re.findall(r"[A-Za-z]{2,}", ln)
        if len(words) < (1 if list_mode else 2) and not (len(words) == 1 and len(words[0]) >= 8):
            continue                                  # one-word topics need 8+ letters (Decidability)
        if not list_mode and (len(ln.split()) > 16 or ln.endswith(".")):
            continue
        items.append(ln)
    # glue wrapped lines: a line ending with a connector continues on the next one
    glued = []
    for it in items:
        continues = bool(glued) and (
            re.search(r"(?:\b(?:of|and|to|the|in|for|with|on|a|an)|[,&(-])$", glued[-1], _F)
            or (not list_mode and it[:1].islower()))          # "... best-case" / "and worst-case ..."
        if continues:
            glued[-1] = glued[-1] + " " + it
        else:
            glued.append(it)
    if m1 is not None:
        m1 = min(m1, len(glued))
    if m2 is not None:
        m2 = min(m2, len(glued))
    return {"items": glued, "mid1_boundary": m1, "mid2_boundary": m2}


def default_scope(parsed: dict, exam_kind: str) -> list[int]:
    """Indexes of syllabus items an exam probably covers. Uses the MIDTERM markers of the
    outline when present, otherwise the first 45% (mid 1) and 45-80% (mid 2)."""
    n = len(parsed["items"])
    b1 = parsed["mid1_boundary"] if parsed["mid1_boundary"] is not None else int(round(0.45 * n))
    b2 = parsed["mid2_boundary"] if parsed["mid2_boundary"] is not None else int(round(0.80 * n))
    b2 = max(b2, b1)
    if exam_kind == "mid1":
        return list(range(0, max(b1, 1)))
    if exam_kind == "mid2":
        return list(range(b1, max(b2, b1 + 1)))
    return list(range(n))


class SyllabusMatcher:
    """Similarity of each question to the best matching syllabus item."""

    def __init__(self, question_texts, items):
        self.items = list(items)
        self.vec = TfidfVectorizer(lowercase=True, stop_words=STOP, token_pattern=TOKEN,
                                   ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.vec.fit(list(question_texts) + self.items)
        self.item_matrix = self.vec.transform(self.items)

    def scores(self, texts) -> np.ndarray:
        if not self.items:
            return np.zeros(len(texts))
        sim = (self.vec.transform(list(texts)) @ self.item_matrix.T).toarray()
        return sim.max(axis=1)

    def best_item(self, texts) -> list[int]:
        sim = (self.vec.transform(list(texts)) @ self.item_matrix.T).toarray()
        return sim.argmax(axis=1).tolist()