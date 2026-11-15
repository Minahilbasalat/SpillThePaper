import streamlit as st


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    """One shared Predictor (topic probabilities) for every page."""
    from src.inference.predictor import Predictor
    return Predictor()


@st.cache_resource(show_spinner="Loading question bank...")
def get_library():
    """One shared Library (browse subjects, past papers, typical papers, past questions)."""
    from src.inference.library import Library
    return Library()


@st.cache_resource(show_spinner="Preparing the paper generator...")
def get_generator():
    """One shared PaperGenerator (predicted papers)."""
    from src.inference.generator import PaperGenerator
    return PaperGenerator(get_predictor(), get_library())


@st.cache_resource(show_spinner="Preparing solutions...")
def get_solutions():
    """One shared SolutionService (archive solutions + optional AI solutions)."""
    from src.inference.solutions import SolutionService
    return SolutionService()
