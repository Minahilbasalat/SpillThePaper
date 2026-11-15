"""One place that decides what kind of exam a paper is (shared by data build and app)."""
import pandas as pd

LABELS = {"final": "Final", "mid1": "Mid 1", "mid2": "Mid 2", "mid": "Mid (number not stated)",
          "quiz": "Quiz (number not stated)", "other": "Other"}


def exam_kind(assessment_type, number) -> str:
    n = None if pd.isna(number) else int(number)
    if assessment_type == "final":
        return "final"
    if assessment_type == "mid":
        return {1: "mid1", 2: "mid2"}.get(n, "mid")
    if assessment_type == "quiz":
        return f"quiz{n}" if n else "quiz"
    return "other"


def label(kind: str) -> str:
    if kind.startswith("quiz") and kind != "quiz":
        return f"Quiz {kind[4:]}"
    return LABELS.get(kind, kind)


def sort_key(kind: str):
    order = {"final": 0, "mid1": 1, "mid2": 2, "mid": 3, "quiz": 99, "other": 100}
    if kind.startswith("quiz") and kind != "quiz":
        return (10 + int(kind[4:]), kind)
    return (order.get(kind, 101), kind)
