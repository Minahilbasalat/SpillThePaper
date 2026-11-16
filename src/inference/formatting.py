"""Make a question that was read from a PDF/scan readable again.

The archive text arrives as ONE long paragraph: line breaks and bullets are lost ("?" instead of a
bullet), sub-parts a. b. c. run into each other, and tags like "(CLO: 3) (Marks: 10)" are mixed in.
format_question() restores the obvious structure with plain rules (no AI, no guessing of content):

    - removes course-outcome and marks tags (the paper shows its own marks, they could disagree)
    - puts bullets, sub-parts (a. / (a) / i.), "Example Input:", "Output:", "Note:" and // comments
      on their own lines
It never changes a word of the question. It is idempotent: format_question(format_question(t)) is
format_question(t). needs_figure() flags questions that refer to a diagram which is not in the text.
"""
from __future__ import annotations

import re

_TAGS = re.compile(r"\s*(?:\(\s*CLO[^)]{0,12}\)|\[\s*CLO[^\]]{0,12}\]"
                   r"|\(\s*(?:Max\.?\s*)?Marks?\s*[-:=]?\s*\d+(?:\.\d+)?\s*\)"
                   r"|\[\s*(?:Max\.?\s*)?Marks?\s*[-:=]?\s*\d+(?:\.\d+)?\s*\])\s*", re.I)
_LABEL = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d+\s*[a-z]?\s*[:.)\-\u2013]+\s*|\d+\s*[.)]\s+)", re.I)
_END_TAGS = re.compile(r"\s*(?:\(\s*\d+(?:\.\d+)?\s*marks?\s*\)|\[\s*\d+(?:\.\d+)?\s*marks?\s*\])\s*$", re.I)
_BULLET_Q = re.compile(r"(?:(?<=[:.;)])|(?<=\n))\s+\?\s+(?=[A-Z0-9(])|^\?\s+(?=[A-Z0-9(])")
_BULLET_CH = re.compile(r"\s*[•●▪]\s*")
_SUBPART = re.compile(r"(?<=[.?:;)\]])\s+(?=(?:\(?[a-hA-H][.)]|\(?(?:iv|v|i{1,3})\))\s+[A-Z(])")
_NUMPART = re.compile(r"(?<=[.?;])\s+(?=\d{1,2}[.)]\s+[A-Z])")
_KEYWORD = re.compile(r"(?<=[.;)\]])\s+(?=(?:Examples?(?:\s+(?:Input|Output))?|Sample\s+(?:Input|Output)"
                      r"|Input|Output|Constraints|Note|Hint|Test\s+Cases?|Explanation)\s*:)")
_COMMENT = re.compile(r"(?<![:/])\s+(?=//\s)")
_CODE_START = re.compile(r"\s+(?=(?:#include\b|using\s+namespace\b|int\s+main\s*\(|void\s+main\s*\())")


def format_question(text: str) -> str:
    t = str(text or "").replace("\r", "\n").replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = t.strip()
    for _ in range(4):                                    # "Question 3: (CLO: 3) (Marks: 10) ..." in any order
        t2 = _LABEL.sub("", _TAGS.sub(" ", t).strip(), count=1).strip()
        if t2 == t:
            break
        t = t2
    t = _END_TAGS.sub("", t)
    t = _BULLET_CH.sub("\n• ", t)
    t = _BULLET_Q.sub("\n• ", t)
    t = _SUBPART.sub("\n", t)
    t = _NUMPART.sub("\n", t)
    t = _KEYWORD.sub("\n", t)
    t = _COMMENT.sub("\n", t)
    t = _CODE_START.sub("\n", t)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    while _END_TAGS.search(t):                           # "... [3 marks]" left at the very end
        t = _END_TAGS.sub("", t).strip()
    return t


_FIG_WORD = re.compile(r"\b(?:tree|graph|node|nodes|circuit|gate|network|diagram|figure|automaton|dfa|nfa)\b", re.I)
_FIG_REF = [
    re.compile(r"\bfig(?:ure)?s?\.?\s*(?:\d+|[A-Z]\b)", re.I),
    re.compile(r"\b(?:shown|given|following|attached|depicted|displayed)\b[^.?\n]{0,40}"
               r"\b(?:diagram|figure|tree|graph|circuit|image|picture|chart|automaton|network)\b", re.I),
    re.compile(r"\b(?:diagram|figure|tree|graph|circuit|image|picture|chart|automaton)\b[^.?\n]{0,25}"
               r"\b(?:below|above|shown|given|attached)\b", re.I),
    re.compile(r"\b(?:in|from|of)\s+the\s+(?:figure|diagram|image|picture|circuit)\b", re.I),
]
_LABEL_LETTER = re.compile(r"\b([B-H])\b(?![.)])|\b(A)\b(?!\s+[a-z])(?![.)])")


def needs_figure(text: str) -> bool:
    """True when the question seems to depend on a figure that is not part of the text."""
    t = str(text or "")
    if any(p.search(t) for p in _FIG_REF):
        return True
    letters = {a or b for a, b in ((m.group(1), m.group(2)) for m in _LABEL_LETTER.finditer(t))}
    return len(letters) >= 3 and bool(_FIG_WORD.search(t))
