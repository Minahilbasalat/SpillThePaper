import streamlit as st


def open_results(paper: dict) -> None:
    """Send a paper to the 'My results' page to upload and mark the student's answers."""
    st.session_state["res_paper"] = paper
    st.session_state["res_screen"] = "upload"
    for k in ("res_sheet", "res_grades", "res_analysis", "res_saved_id", "res_personalised"):
        st.session_state.pop(k, None)
    st.switch_page("pages/3_My_Results.py")
