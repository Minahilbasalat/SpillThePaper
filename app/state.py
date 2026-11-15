"""Session-state keys and helpers shared by the UI pages (UI of features 1-4)."""

from __future__ import annotations

import streamlit as st

PREDICT_SUBJECT = "predict_subject"
PREDICT_EXAM = "predict_exam"


def exam_for_kind(kind: str) -> str | None:
    """Prediction exam value (quiz/mid1/mid2/final) for a past-paper kind, or None."""
    if kind in ("final", "mid1", "mid2"):
        return kind
    if kind.startswith("quiz"):
        return "quiz"
    return None


def preselect_prediction(subject: str, kind: str | None = None) -> None:
    """Remember a subject (and exam) so the Upcoming paper page opens preselected."""
    st.session_state[PREDICT_SUBJECT] = subject
    exam = exam_for_kind(kind) if kind else None
    if exam:
        st.session_state[PREDICT_EXAM] = exam
    else:
        st.session_state.pop(PREDICT_EXAM, None)


def take_preselection() -> tuple[str | None, str | None]:
    """Read and clear the subject/exam chosen on the Past papers page (used once)."""
    subject = st.session_state.pop(PREDICT_SUBJECT, None)
    exam = st.session_state.pop(PREDICT_EXAM, None)
    return subject, exam