import bootstrap  # noqa: F401
import streamlit as st
import theme

theme.setup("My results", active="results")

CSS = """
.st-key-soon-card{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:3px;
box-shadow:8px 8px 0 var(--ink);transform:rotate(-.6deg);padding:2rem 2.4rem 2rem;margin-top:1rem;
max-width:46rem;gap:.4rem;}
.st-key-soon-card::before{content:"";position:absolute;top:-16px;left:34px;width:96px;height:28px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-3deg);}
.soon-note{font-family:'Special Elite',monospace;font-size:.78rem;color:var(--ink);}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

theme.title("Evidence drop", "My", "results")

with st.container(key="soon-card"):
    st.markdown(
        '<div class="hero-top"><div class="kicker">Not built yet</div><div class="stamp">SOON</div></div>'
        '<div class="hero-h">Coming soon</div>'
        '<div class="hero-p">Coming soon: upload photos of your answers, see your weak areas '
        "and get a personalised paper.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="redact" style="width:90%"></div><div class="redact" style="width:55%"></div>',
                unsafe_allow_html=True)
    c_predict, c_browse, _ = st.columns([1.5, 1.5, 2], vertical_alignment="center")
    with c_predict:
        with st.container(key="cta-predict"):
            st.page_link("pages/2_Upcoming_Paper.py", label="Predict a paper " + theme.ARROW)
    with c_browse:
        with st.container(key="cta-browse"):
            st.page_link("pages/1_Past_Papers.py", label="Browse past papers")
    st.button("Upload my solution", key="mr_upload", disabled=True)
    st.markdown('<div class="soon-note">Coming soon</div>', unsafe_allow_html=True)