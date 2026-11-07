"""Split the text of one exam paper into question chunks.

Strategy (first one that finds at least two markers wins):
  1. 'Question 1', 'Q1', 'Q.2', 'Ques 3' markers at the start of a line
  2. numbered items '1.', '2)', '(3)' at the start of a line
  3. fallback: paragraphs, merged until they reach a minimum length
"""
import re

MIN_WORDS = 4            # chunks shorter than this are dropped
FALLBACK_MIN_WORDS = 25  # paragraphs are merged until they have this many words

QUESTION_MARKER = re.compile(
    r"(?im)^[ \t]*(?:question|ques|q)\s*[.:#\-]?\s*\(?\d{1,2}\)?\s*[.:)\-]?"
)
NUMBERED_ITEM = re.compile(r"(?m)^[ \t]*\(?\d{1,2}\s*[.)]\s+(?=\S)")

# lines that belong to the paper header or footer, not to a question
BOILERPLATE_LINE = re.compile(
    r"(?i)^\s*(name\b|roll\s*(no|number)|section\b|student\s*(id|name)|"
    r"signature|time\s*allowed|total\s*time|max(imum)?\s*(marks|time)|"
    r"national university|department of|good luck|best of luck|page\s*\d+\s*(of\s*\d+)?\s*$)"
)

# chunks that are not questions: worked solutions and grading-table headers
SOLUTION_HEAD = re.compile(
    r"(?i)^\W*(?:(?:question|ques|q)\s*[.:#\-]?\s*\d{1,2}\s*[.:)\-]?\s*)?"
    r"(?:solution\b|(?:answer|ans)\s*[:.\-)])"
)
GRADING_TABLE = re.compile(r"(?i)total marks.*obtained marks|obtained marks.*total marks")


def _is_question(chunk: str) -> bool:
    if SOLUTION_HEAD.match(chunk):
        return False
    if GRADING_TABLE.search(chunk[:300]):
        return False
    return True

def _clean_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not BOILERPLATE_LINE.match(ln)]
    return "\n".join(lines)


def _normalise(chunk: str) -> str:
    return re.sub(r"\s+", " ", chunk).strip()


def _split_on(pattern: re.Pattern, text: str):
    starts = [m.start() for m in pattern.finditer(text)]
    if len(starts) < 2:
        return None
    starts.append(len(text))
    return [text[a:b] for a, b in zip(starts[:-1], starts[1:])]


def _split_paragraphs(text: str):
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        current = f"{current}\n{p}" if current else p
        if len(current.split()) >= FALLBACK_MIN_WORDS:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def segment_questions(text: str):
    """Return (list_of_question_texts, method_name)."""
    text = _clean_lines(text)
    method, chunks = "paragraph", None
    for name, pattern in (("marker", QUESTION_MARKER), ("numbered", NUMBERED_ITEM)):
        chunks = _split_on(pattern, text)
        if chunks is not None:
            method = name
            break
    if chunks is None:
        chunks = _split_paragraphs(text)
    questions = [_normalise(c) for c in chunks]
    questions = [q for q in questions if len(q.split()) >= MIN_WORDS and _is_question(q)]
    return questions, method