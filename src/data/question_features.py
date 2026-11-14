"""Question-level features: marks, answer text, question type, Bloom level, difficulty.

All rules are plain regular expressions and keyword lists so that every number in the
report can be traced. Nothing here is a trained model, and the difficulty is a PROXY
(Bloom level of the task verb + length + marks share), not a measured difficulty.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- marks
_FLAGS = re.I
_N = r"(\d{1,3}(?:\.\d)?)"
RE_MARKS_WORD = re.compile(rf"(?<![\w.]){_N}\s*(?:marks?|pts?|points?)\b", _FLAGS)
RE_MARKS_LABEL = re.compile(rf"\b(?:marks?|points?)\s*[:=]\s*{_N}", _FLAGS)
RE_MARKS_BRACKETED = re.compile(rf"[\(\[]\s*(?:marks?|points?)\s*[:=]?\s*{_N}\s*[\)\]]", _FLAGS)
RE_PLUS_SUM = re.compile(r"[\(\[]\s*(\d{1,3}(?:\s*\+\s*\d{1,3})+)\s*(?:marks?|pts?|points?)?\s*[\)\]]", _FLAGS)
RE_BRACKET = re.compile(rf"\[\s*{_N}\s*(?:marks?|pts?)?\s*\]", _FLAGS)
RE_PAREN_END = re.compile(rf"\(\s*{_N}\s*\)\s*(?:[.?:;]|\d{{0,2}})\s*$")
MARKS_MAX = 100.0


def parse_marks(text: str) -> tuple[float, str]:
    """Return (marks, source). marks is NaN when the text states none.

    Priority: '(3+4+3)' sums > explicit '5 marks' / 'Marks: 5' (several mentions for parts
    a, b, c are added up) > '(Marks 5)' > '[5]' brackets > a trailing '(5)'."""
    t = text or ""
    plus = RE_PLUS_SUM.findall(t)
    if plus:
        return _check(sum(float(x) for part in plus for x in part.split("+")), "plus_sum")
    found = [float(x) for x in RE_MARKS_WORD.findall(t)] + \
            [float(x) for x in RE_MARKS_LABEL.findall(t)]
    if found:
        return _check(sum(found), "marks_word")
    bracketed = [float(x) for x in RE_MARKS_BRACKETED.findall(t)]
    if bracketed:
        return _check(sum(bracketed), "marks_word")
    b = [float(x) for x in RE_BRACKET.findall(t)]
    if b:
        return _check(sum(b), "bracket")
    e = RE_PAREN_END.search(t.strip())
    if e:
        return _check(float(e.group(1)), "paren_end")
    return np.nan, ""


def _check(value: float, source: str) -> tuple[float, str]:
    return (value, source) if 0 < value <= MARKS_MAX else (np.nan, "")


# ------------------------------------------------------------------------ solutions
# 'Solution' with a capital S, or Sol/Answer/Ans followed by ':' or '.', inside the chunk.
RE_SOLUTION = re.compile(r"(?:(?<=\s)|^)(?:Solutions?\b|(?:Sol|Answer|Ans)\s*[:.\-–])")


def split_solution(text: str) -> tuple[str, str]:
    """Split a chunk into (question, solution) at the first explicit solution marker.
    The marker must come after at least 25 characters of question text."""
    t = text or ""
    m = RE_SOLUTION.search(t, 25)
    if not m:
        return t.strip(), ""
    return t[: m.start()].strip(), t[m.start():].strip()


# ------------------------------------------------------------------ question type
RE_MCQ_CUE = re.compile(r"\b(which of the following|choose (?:all|the|one)|select the correct|"
                        r"multiple choice|mcqs?)\b", _FLAGS)
RE_OPT_A = re.compile(r"(?:^|\s)\(?[aA][\.\)\-:]\s")
RE_OPT_B = re.compile(r"(?:^|\s)\(?[bB][\.\)\-:]\s")
RE_OPT_C = re.compile(r"(?:^|\s)\(?[cC][\.\)\-:]\s")
RE_TF = re.compile(r"\b(true\s*(?:or|/)\s*false|true/false|state whether)\b", _FLAGS)
# strong code evidence only: braces/semicolon statements, includes, function definitions
RE_CODE_STRONG = re.compile(r"\{[^}]*;[^}]*\}|#include|\bdef\s+\w+\(|\bfor\s*\([^)]*;|"
                            r"\bwhile\s*\(.+\)\s*\{|\bcout\s*<<|\bSystem\.out|"
                            r"\bpublic\s+(?:static|class)|\bconsole\.log|\bfunction\s+\w*\(")
RE_CODE_TASK = re.compile(r"\b(write|implement|code|develop)\b[^.?]{0,60}\b(program|function|"
                          r"method|class|code|script|query|algorithm|pseudo\s*code|snippet|"
                          r"procedure|macro|assembly)\b|\b(?:sql|c\+\+|java|python|javascript) "
                          r"(?:code|program|query)\b", _FLAGS)
RE_DIAGRAM = re.compile(r"\b(draw|sketch|diagram|design an? (?:er|dfd|uml)|construct|flow\s*chart|"
                        r"dfa|nfa|state machine|parse tree|er diagram|uml|turing machine|pda|cfg|"
                        r"pushdown|regular expression|grammar|circuit|karnaugh|k-map|schema)\b", _FLAGS)
RE_PROOF = re.compile(r"\b(prove|proof|show that|derive|derivation|verify that)\b", _FLAGS)
RE_NUMERIC = re.compile(r"\b(calculate|compute|find the (?:value|number|total|size|time|"
                        r"probability|throughput|output|result)|how many|how much|evaluate the|"
                        r"what is the (?:value|output|result|size|time|total))\b", _FLAGS)
RE_THEORY = re.compile(r"\b(explain|describe|define|what is|what are|differentiate|compare|"
                       r"discuss|list|state|why|how does|how do|briefly|justify|elaborate|"
                       r"advantages|disadvantages|mention|give reasons?|enlist|outline)\b", _FLAGS)
RE_SCENARIO = re.compile(r"\b(company|organi[sz]ation|scenario|case study|consider the following|"
                         r"suppose|assume|startup|firm|bank|hospital|university|a client|"
                         r"a customer|business)\b", _FLAGS)


def question_type(text: str) -> str:
    t = text or ""
    head = t[:600]
    words = t.split()
    alpha = sum(ch.isalpha() for ch in t) / max(len(t), 1)
    if len(words) < 6 or alpha < 0.45:
        return "fragment"
    if RE_TF.search(head):
        return "true_false"
    n_opts = sum(bool(r.search(t)) for r in (RE_OPT_A, RE_OPT_B, RE_OPT_C))
    if n_opts >= 3 or (n_opts >= 2 and RE_MCQ_CUE.search(t)):
        return "mcq"
    if RE_CODE_STRONG.search(t) or RE_CODE_TASK.search(head):
        return "code"
    if RE_PROOF.search(head):
        return "proof_derivation"
    if RE_DIAGRAM.search(head):
        return "diagram_design"
    if RE_NUMERIC.search(head) or (re.search(r"\d", head) and
                                   re.search(r"\b(solve|find|determine|given)\b", head, _FLAGS)):
        return "numerical"
    if RE_THEORY.search(head):
        return "theory"
    if len(words) >= 60 and RE_SCENARIO.search(head):
        return "scenario"
    if re.search(r"_{3,}", t):
        return "fill_blank"
    if "?" in t:
        return "theory"
    return "other"


# ---------------------------------------------------------------- Bloom / difficulty
BLOOM_VERBS = {
    1: r"define|list|state|name|identify|what is|what are|mention|recall|label|write down",
    2: r"explain|describe|discuss|summari[sz]e|differentiate|distinguish|compare|"
       r"interpret|illustrate|give an example|why|briefly",
    3: r"calculate|compute|solve|find|apply|use|trace|show|determine|implement|write|execute|convert",
    4: r"analy[sz]e|examine|derive|prove|break down|infer|differentiate between|debug|optimi[sz]e",
    5: r"justify|evaluate|critique|assess|argue|judge|recommend|defend",
    6: r"design|develop|construct|propose|create|formulate|build|plan|devise",
}
_BLOOM_RE = {lvl: re.compile(rf"\b(?:{v})\b", _FLAGS) for lvl, v in BLOOM_VERBS.items()}


def bloom_level(text: str) -> int:
    """Highest Bloom level whose verb appears in the first 300 characters (default 2)."""
    head = (text or "")[:300]
    hits = [lvl for lvl, rx in _BLOOM_RE.items() if rx.search(head)]
    return max(hits) if hits else 2


def add_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy score = z(Bloom) + z(log words) + z(marks share), bucketed into terciles
    inside each assessment type (easy / medium / hard)."""
    d = df.copy()
    words = d["question_text"].str.split().str.len().clip(lower=1)
    share = (d["marks"] / d.groupby("file_id")["marks"].transform("sum")).fillna(0.0)

    def z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0

    d["difficulty_score"] = z(d["bloom"].astype(float)) + z(np.log1p(words)) + 0.5 * z(share)
    d["difficulty"] = "medium"
    for _, idx in d.groupby("assessment_type").groups.items():
        s = d.loc[idx, "difficulty_score"]
        lo, hi = s.quantile([1 / 3, 2 / 3])
        d.loc[idx, "difficulty"] = np.where(s <= lo, "easy", np.where(s >= hi, "hard", "medium"))
    return d


# ----------------------------------------------------------------------- the table
def build_question_table(q: pd.DataFrame) -> pd.DataFrame:
    """q: questions_topics.csv (file_id, subject, year, assessment_type, is_solution,
    q_index, method, q_text, topic_id). Returns one row per question with the new columns."""
    d = q.copy()
    parts = d["q_text"].map(split_solution)
    d["question_text"] = [p[0] for p in parts]
    d["solution_text"] = [p[1] for p in parts]
    d["has_solution"] = d["solution_text"].str.len() > 0
    marks = d["q_text"].map(parse_marks)
    d["marks"] = [m[0] for m in marks]
    d["marks_source"] = [m[1] for m in marks]
    d["qtype"] = d["question_text"].map(question_type)
    d["bloom"] = d["question_text"].map(bloom_level)
    d["from_solution_paper"] = d["is_solution"].astype(bool)
    # a chunk from a solution paper whose answer we could not separate may still contain
    # the answer text: keep it for statistics but never show it as an example question
    d["contaminated"] = d["from_solution_paper"] & ~d["has_solution"]
    return add_difficulty(d)