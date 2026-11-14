"""Remove exam rules and page furniture that the segmenter mistook for questions.

Two functions, applied in src/data/build_questions.py before anything else is computed:

  strip_furniture(chunk)  removes page headers/footers glued into a question
                          ("... (2 marks) | National University of Computer and Emerging
                          Sciences, Lahore Campus | Page 3 of 7")
  is_instruction(chunk, repeats)  True for exam-conduct text such as "Cell phones must be
                          turned off" or "Any cheating results in disqualification".

A chunk counts as an instruction only if it has a conduct cue AND no task signal (no
question mark, no marks, no task verb). That keeps real questions that merely mention
"question paper" or "marks will be deducted". Dropped chunks are saved for inspection.
"""
from __future__ import annotations

import re

_F = re.I

# ----------------------------------------------------------------- page furniture
FURNITURE = [
    re.compile(r"\|?\s*National University of Computer\s*(?:and|&)\s*Emerging Sciences[^|\n]*", _F),
    re.compile(r"\|?\s*FAST[\s-]*(?:NU|NUCES)?[^|\n]{0,40}(?:Campus|School of Computing)[^|\n]*", _F),
    re.compile(r"\bP\s?a\s?g\s?e\s*\d+\s*(?:\||of|/)\s*\d+\b", _F),
    re.compile(r"\bPage\s*\d+\s*(?:of|/)\s*\d+\b", _F),
    re.compile(r"Student\s*(?:Registration\s*No\.?|Name|ID|Roll\s*No\.?)\s*[:_.\s-]*_{2,}[_\s.:]*", _F),
    re.compile(r"(?:Registration|Roll)\s*(?:No|Number)\.?\s*[:_\s]*_{3,}", _F),
    re.compile(r"\b(?:Section|Name|Signature)\s*[:\s]*_{3,}", _F),
]


def strip_furniture(chunk: str) -> str:
    t = chunk
    for rx in FURNITURE:
        t = rx.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.rstrip(" |")                                 # dangling separators at the end
    return re.sub(r"(?: \|){2,}", " |", t)


# --------------------------------------------------------------- exam conduct text
CONDUCT = re.compile(
    r"\b(cell\s*phones?|mobile\s*phones?|smart\s*watch(?:es)?|turned off|switched off|"
    r"cheat(?:ing)?|cheat sheet|plagiari[sz]\w*|disqualif\w*|unfair means|academic dishonesty|"
    r"borrowing|whispering|invigilator|seating|exam(?:ination)? hall|usb\b|"
    r"protect your code|matching codes?|do not ask any clarification|"
    r"understanding the question paper|final submissions? should|section folder|"
    r"time allowed|maximum marks|attempt all|read (?:the )?instructions|"
    r"instructions?\s*(?:/\s*notes?)?\s*:)", _F)

TASK = re.compile(
    r"\?|\b(explain|calculate|compute|write|implement|design|draw|define|derive|prove|find|"
    r"what|why|how|describe|show that|determine|list|compare|differentiate|identify|"
    r"construct|create|given|consider|suppose|assume that|convert|trace|evaluate|"
    r"discuss|state|justify)\b", _F)
MARKS = re.compile(r"\(\s*\d{1,2}(?:\s*\+\s*\d{1,2})*\s*(?:marks?|pts?|points?)?\s*\)|\[\s*\d{1,2}[^\]]*\]|"
                   r"\b\d{1,2}\s*(?:marks?|points?)\b", _F)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", t.lower())).strip()


def is_instruction(chunk: str, repeats: int = 1) -> bool:
    """repeats: in how many different papers of the same subject this text (normalised)
    appears. Short text repeated in 3+ papers with no task signal is boilerplate too."""
    words = len(chunk.split())
    if words < 4:                                   # empty after removing page furniture
        return True
    has_task = bool(TASK.search(chunk)) or bool(MARKS.search(chunk))
    cues = len(CONDUCT.findall(chunk))
    if cues and not has_task:
        return True
    if cues >= 2 and "?" not in chunk and not MARKS.search(chunk):
        return True                                 # a list of rules, whatever stray verbs it has
    # conduct text that mentions a task verb only inside a rule ("do not write ...")
    if cues and words <= 45 and not MARKS.search(chunk) and "?" not in chunk:
        return True
    if repeats >= 3 and words <= 40 and not has_task:
        return True
    return False


def normalised(chunk: str) -> str:
    return _norm(chunk)
