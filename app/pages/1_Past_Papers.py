import bootstrap  # noqa: F401
import streamlit as st
import theme
from common import get_library

theme.setup("Past papers", active="past")
lib = get_library()

PAGE_SIZE = 24
GRID_COLUMNS = 4
st.session_state.setdefault("past_subject", None)
st.session_state.setdefault("past_limit", PAGE_SIZE)


def open_subject(name: str | None) -> None:
    st.session_state["past_subject"] = name


def subjects_screen() -> None:
    left, right = st.columns([2.2, 1], vertical_alignment="bottom")
    with left:
        theme.title("The archive", "Past", "papers")
    with right:
        query = st.text_input(
            "Search a subject",
            placeholder="Search a subject",
            label_visibility="collapsed",
            key="past_search",
        ).strip()

    subjects = lib.all_subjects().sort_values(["papers", "subject"], ascending=[False, True])
    if query:
        subjects = subjects[subjects["subject"].str.contains(query, case=False, regex=False)]

    if subjects.empty:
        theme.notice("No subject matches your search. Try fewer letters, or predict the next paper below.")
    shown = subjects.head(st.session_state["past_limit"])

    index = 0
    for start in range(0, len(shown), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS, gap="large")
        for col, row in zip(cols, shown.iloc[start:start + GRID_COLUMNS].itertuples()):
            color = theme.FOLDER_COLORS[index % len(theme.FOLDER_COLORS)]
            with col:
                with st.container(key=f"folder-{color}-{index}"):
                    st.markdown(theme.folder_markup(row.subject, row.kinds, row.papers), unsafe_allow_html=True)
                    st.button(f"Open {row.subject}", key=f"open_{index}", on_click=open_subject, args=(row.subject,))
            index += 1

    remaining = len(subjects) - len(shown)
    if remaining > 0:
        if st.button(f"Show more ({remaining} more subjects)", key="past_more"):
            st.session_state["past_limit"] += PAGE_SIZE
            st.rerun()

    with st.container(key="banner"):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown('<div class="banner-t">Can\u2019t find your subject?</div>', unsafe_allow_html=True)
        with c2:
            with st.container(key="cta-banner"):
                st.page_link("pages/2_Upcoming_Paper.py", label="Predict the next paper " + theme.ARROW)


def subject_screen(name: str) -> None:
    st.button("\u2190 All subjects", key="past_back", on_click=open_subject, args=(None,))
    theme.title_last("The archive", name)
    st.write("The subject page is built in the next step.")


current = st.session_state["past_subject"]
if current:
    subject_screen(current)
else:
    subjects_screen()