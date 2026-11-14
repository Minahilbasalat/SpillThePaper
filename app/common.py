import streamlit as st


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    """One shared Predictor (topic probabilities) for every page."""
    from src.inference.predictor import Predictor
    return Predictor()


@st.cache_resource(show_spinner="Loading question bank...")
def get_library():
    """One shared Library (subjects, past questions, typical papers, past papers)."""
    from src.inference.library import Library
    return Library()
