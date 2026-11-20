import bootstrap  # noqa: F401
import state
import streamlit as st
import theme

theme.setup("Home", active="home")
state.reset_predict_form()      # a new prediction always starts with an empty form

HERO_HTML = (
    '<div class="hero-top">'
    f'<div class="kicker">Classified {theme.DOT} FAST exam intel</div>'
    '<div class="stamp">LEAKED</div></div>'
    '<div class="hero-h">Spill the paper <span class="hl">before</span> it spills you.</div>'
    '<div class="hero-p">Pick your subject, add the syllabus, and get a predicted paper '
    "built from every past FAST paper. Solve it, upload your answers, and find out "
    "exactly where you are weak.</div>"
    '<div class="redact" style="width:78%"></div>'
    '<div class="redact" style="width:56%"></div>'
)

left, right = st.columns([1.6, 1], gap="large")

with left:
    with st.container(key="hero"):
        st.markdown(HERO_HTML, unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.15, 1.25, 1.3])
        with c1:
            with st.container(key="cta-predict"):
                st.page_link("pages/2_Upcoming_Paper.py", label="Predict my paper " + theme.ARROW)
        with c2:
            with st.container(key="cta-browse"):
                st.page_link("pages/1_Past_Papers.py", label="Browse past papers")

with right:
    theme.note(
        "past", "Past papers",
        "Every subject. Final, Mid 1, Mid 2 and quizzes, real papers inside.",
        "yellow", "R", "pages/1_Past_Papers.py",
    )
    theme.note(
        "upcoming", "Upcoming paper",
        "Add the syllabus, upload recent papers if you have them, get the prediction.",
        "pink", "L", "pages/2_Upcoming_Paper.py",
    )
    theme.note(
        "results", "Upload your solution",
        "Snap a photo of your answers. We find your weak spots.",
        "blue", "R", "pages/3_My_Results.py",
    )

theme.steps_strip(
    [
        "Pick subject and exam type",
        "Add the syllabus (and recent papers, optional)",
        "Get the predicted paper, with an optional timer",
        "Upload your solution, see the weak areas",
        "Get a personalized paper that targets them",
    ]
)