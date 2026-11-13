import os

import streamlit as st


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    """One shared predictor for every page. Set SPILL_USE_STUB=1 to use fake data."""
    if os.environ.get("SPILL_USE_STUB") == "1":
        from src.inference.stub import DummyPredictor
        return DummyPredictor()
    from src.inference.predictor import Predictor
    return Predictor()