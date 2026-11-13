import bootstrap  # noqa: F401
import streamlit as st
from common import get_predictor

st.set_page_config(page_title="SpillThePaper", page_icon="📄", layout="wide")
pred = get_predictor()

st.title("📄 SpillThePaper")
st.subheader("Evidence-based exam preparation from FAST-NUCES past papers")
st.write(
    "Pick a subject and exam type to see which topics are most likely to appear, "
    "upload a past paper to see its topic coverage, or build a tentative paper outline."
)
c1, c2, c3 = st.columns(3)
c1.metric("Supported subjects", len(pred.subjects()))
c2.metric("Models available", len(pred.model_names()))
c3.metric("Low-confidence subjects", len(pred.low_confidence_subjects()))
st.info(
    "SpillThePaper is a study aid and does not guarantee what appears in a real exam. "
    "Probabilities are best read as relative rankings, not exact chances."
)