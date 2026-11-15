import bootstrap  # noqa: F401
import state
import streamlit as st
import theme
from common import get_library
from components import paper_view

theme.setup("Past papers", active="past")
paper_view.inject_css()
lib = get_library()

PAGE_SIZE = 24
GRID_COLUMNS = 4
st.session_state.setdefault("past_subject", None)
st.session_state.setdefault("past_kind", None)
st.session_state.setdefault("past_read", None)
st.session_state.setdefault("past_limit", PAGE_SIZE)


def open_subject(name: str | None) -> None:
    st.session_state["past_subject"] = name
    st.session_state["past_kind"] = None
    st.session_state["past_read"] = None


def choose_kind(kind: str) -> None:
    st.session_state["past_kind"] = kind


def read_paper(file_id: str) -> None:
    st.session_state["past_read"] = file_id


def close_reader() -> None:
    st.session_state["past_read"] = None


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


def reader_screen(file_id: str) -> None:
    st.button("\u2190 Back to papers", key="past_read_back", on_click=close_reader)
    try:
        paper = lib.paper(file_id)
    except KeyError:
        theme.notice("The text of this paper could not be read, so it can only be viewed or downloaded.")
        return
    theme.title_last("Reading mode", paper["title"])
    st.write(f"{len(paper['questions'])} questions loaded. The reading view with the timer is built in the next step.")


def subject_screen(name: str) -> None:
    kinds = lib.exam_kinds(name)
    if not kinds:
        st.button("\u2190 All subjects", key="past_back", on_click=open_subject, args=(None,))
        theme.title_last("The archive", name)
        theme.notice("No papers are listed for this subject yet.")
        return

    ids = [k["kind"] for k in kinds]
    if st.session_state["past_kind"] not in ids:
        st.session_state["past_kind"] = ids[0]
    kind = st.session_state["past_kind"]
    label = next(k["label"] for k in kinds if k["kind"] == kind)

    st.markdown(paper_view.crumb_markup(name, label), unsafe_allow_html=True)
    head, back = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        theme.title_last("", name)
    with back:
        st.button("\u2190 All subjects", key="past_back", on_click=open_subject, args=(None,))

    side, main, extra = st.columns([1.05, 3.1, 1.2], gap="large")

    with side:
        for i, k in enumerate(kinds):
            mode = "active" if k["kind"] == kind else "idle"
            with st.container(key=f"kind-{mode}-{i}"):
                st.markdown(paper_view.kind_markup(k["label"], k["papers"]), unsafe_allow_html=True)
                st.button(f"Show {k['label']} papers", key=f"kind_btn_{i}", on_click=choose_kind, args=(k["kind"],))

    with main:
        papers = lib.papers(name, kind)
        with st.container(key="paperlist"):
            st.markdown(f'<div class="plist-h">{theme.esc(label)} papers</div>', unsafe_allow_html=True)
            if papers.empty:
                theme.notice("No papers of this kind yet.")
            for i, row in enumerate(papers.itertuples()):
                with st.container(key=f"prow-{i}"):
                    c_info, c_view, c_dl, c_read = st.columns([3.3, 0.85, 1.15, 1.5], vertical_alignment="center")
                    heading = paper_view.paper_heading(row.label, row.term, row.year)
                    meta = paper_view.paper_meta(row.questions, row.total_marks)
                    c_info.markdown(
                        paper_view.row_markup(row.filename, heading, meta, row.includes_solution),
                        unsafe_allow_html=True,
                    )
                    if isinstance(row.original_url, str) and row.original_url:
                        with c_view:
                            with st.container(key=f"lnk-view-{i}"):
                                st.link_button("View", row.original_url)
                    if isinstance(row.download_url, str) and row.download_url:
                        with c_dl:
                            with st.container(key=f"lnk-dl-{i}"):
                                st.link_button("Download", row.download_url)
                    if paper_view.question_count(row.questions) > 0:
                        with c_read:
                            st.button("Read and practise", key=f"read_{i}", on_click=read_paper, args=(row.file_id,))

    with extra:
        theme.note(
            "next", "",
            f"Studied these? Now see what the NEXT {label.lower()} could look like.",
            "yellow", "R",
        )
        with st.container(key="cta-next"):
            if st.button("Predict the next one " + theme.ARROW, key="past_predict_next"):
                state.preselect_prediction(name, kind)
                st.switch_page("pages/2_Upcoming_Paper.py")


current = st.session_state["past_subject"]
if current is None:
    subjects_screen()
elif st.session_state["past_read"]:
    reader_screen(st.session_state["past_read"])
else:
    subject_screen(current)