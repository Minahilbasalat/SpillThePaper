"""Read-only access to the REAL data behind the app. No models are loaded here.

Browsing (feature "past papers"):
    all_subjects() -> DataFrame           every subject in the archive
    exam_kinds(subject) -> list[dict]     Final, Mid 1, Mid 2, Quiz 1, Quiz 2 ...
    papers(subject, kind=None) -> DataFrame   real papers with a link to the original file
    paper(file_id) -> dict                the paper as readable questions (if its text was read)

Paper structure (the paper generator returns the same dictionary, so one page can show both):
    {"title", "subject", "assessment_type", "exam_kind", "year", "source": "past_paper"|"predicted",
     "duration_minutes", "total_marks", "questions": [{"number", "text", "marks", "qtype",
     "difficulty", "topic_id", "topic_label", "solution", ...}]}
"""
from __future__ import annotations

from urllib.parse import quote

import pandas as pd

from src.data.exam_kinds import exam_kind, label, sort_key
from src.inference import config as C
from src.inference.formatting import format_question, needs_figure

DEFAULT_DURATION_MIN = {"quiz": 30, "mid": 60, "final": 180}     # students may change these
_SKIP_TYPES = {"fragment"}


def default_duration(exam: str) -> int:
    t = "mid" if exam.startswith("mid") else "quiz" if exam.startswith("quiz") else exam
    return DEFAULT_DURATION_MIN.get(t, 60)


def original_url(repo_path: str) -> str:
    """Opens the real paper in GitHub's viewer."""
    return f"{C.ARCHIVE_URL}/{C.ARCHIVE_COMMIT}/{quote(repo_path)}"


def download_url(repo_path: str) -> str:
    """Direct link to the raw file (browsers download it)."""
    return f"{C.RAW_URL}/{C.ARCHIVE_COMMIT}/{quote(repo_path)}"


class Library:
    def __init__(self):
        q = pd.read_csv(C.QUESTION_TABLE_CSV, dtype={"file_id": str})
        q["year"] = q["year"].astype(int)
        q["needs_figure"] = q["question_text"].astype(str).map(needs_figure)       # decided on the raw text
        q["question_text"] = q["question_text"].astype(str).map(format_question)   # readable layout
        self.q = q
        q["low_quality"] = q["low_quality"].astype(bool) if "low_quality" in q else False
        # readable: could be shown to a student (garbled OCR is flagged, not hidden);
        # usable: good enough to be used as an example or to build a predicted paper
        self.readable = q[(~q["contaminated"].astype(bool)) & (~q["qtype"].isin(_SKIP_TYPES))]
        self.usable = self.readable[~self.readable["low_quality"]]
        self.catalog = pd.read_csv(C.TOPIC_CATALOG_CSV)
        self.elig = pd.read_csv(C.ELIGIBILITY_CSV)
        self.blueprints = pd.read_csv(C.BLUEPRINTS_CSV, dtype={"file_id": str})
        p = pd.read_csv(C.PAPERS_CSV, dtype={"file_id": str})
        p["exam_kind"] = [exam_kind(a, n) for a, n in zip(p["assessment_type"], p["assessment_number"])]
        self.all_papers = p
        self._label = {(r.subject, int(r.topic_id)): r.keywords
                       for r in self.catalog.itertuples(index=False)}
        self._nq = self.q.groupby("file_id").size().to_dict()

    # --------------------------------------------------------------- browsing
    def supported(self) -> list[str]:
        """Subjects the predictor can handle."""
        return sorted(self.elig.loc[self.elig["eligible"], "subject"])

    subjects = supported

    def all_subjects(self) -> pd.DataFrame:
        p = self.all_papers
        g = p.groupby("subject").agg(papers=("file_id", "count"),
                                     readable=("has_text", "sum"),
                                     first_year=("year", "min"), last_year=("year", "max"))
        def kinds(sub):
            ks = {("quiz" if k.startswith("quiz") else "mid" if k == "mid" else k) for k in sub}
            names = {"final": "Final", "mid1": "Mid 1", "mid2": "Mid 2", "mid": "Mid", "quiz": "Quiz"}
            return " · ".join(names[k] for k in ("final", "mid1", "mid2", "mid", "quiz") if k in ks)
        g["kinds"] = p.groupby("subject")["exam_kind"].apply(kinds)
        g["supported"] = g.index.isin(self.supported())
        g["readable"] = g["readable"].astype(int)
        return g.reset_index().sort_values("subject").reset_index(drop=True)

    def exam_kinds(self, subject: str) -> list[dict]:
        p = self.all_papers[self.all_papers["subject"] == subject]
        out = [{"kind": k, "label": label(k), "papers": int(n)}
               for k, n in p["exam_kind"].value_counts().items()]
        return sorted(out, key=lambda d: sort_key(d["kind"]))

    def papers(self, subject: str, kind: str | None = None) -> pd.DataFrame:
        p = self.all_papers[self.all_papers["subject"] == subject]
        if kind:
            p = p[p["exam_kind"] == kind]
        out = pd.DataFrame({
            "file_id": p["file_id"], "year": p["year"], "term": p["term"],
            "exam_kind": p["exam_kind"], "label": p["exam_kind"].map(label),
            "filename": p["repo_path"].str.rsplit("/", n=1).str[-1],
            "includes_solution": p["is_solution"].astype(bool),
            "has_text": p["has_text"].astype(bool),
            "total_marks": p["total_marks"],
            "questions": p["file_id"].map(lambda f: self._nq.get(f, 0)),
            "original_url": p["repo_path"].map(original_url),
            "download_url": p["repo_path"].map(download_url)})
        return out.sort_values(["year", "filename"], ascending=[False, True]).reset_index(drop=True)

    # ---------------------------------------------------------------- subject
    def overview(self, subject: str) -> dict:
        p = self.all_papers[self.all_papers["subject"] == subject]
        if p.empty:
            raise KeyError(f"unknown subject {subject!r}")
        e = self.elig[self.elig["subject"] == subject]
        ok = bool(len(e) and e.iloc[0]["eligible"])
        reason = "" if ok else (str(e.iloc[0]["reason"]) if len(e) else "no readable papers")
        return {"subject": subject, "supported": ok, "reason_if_unsupported": reason,
                "papers": int(len(p)), "readable_papers": int(p["has_text"].sum()),
                "questions": int((self.q["subject"] == subject).sum()),
                "first_year": int(p["year"].min()) if p["year"].notna().any() else None,
                "last_year": int(p["year"].max()) if p["year"].notna().any() else None,
                "papers_by_kind": {d["label"]: d["papers"] for d in self.exam_kinds(subject)},
                "topics": int((self.catalog["subject"] == subject).sum())}

    def topics(self, subject: str) -> pd.DataFrame:
        t = self.catalog[self.catalog["subject"] == subject]
        return (t[["topic_id", "keywords", "all_questions"]]
                .rename(columns={"keywords": "topic_label", "all_questions": "questions"})
                .sort_values("topic_id").reset_index(drop=True))

    def topic_label(self, subject: str, topic_id) -> str:
        return self._label.get((subject, int(topic_id)), "") if pd.notna(topic_id) else ""

    # -------------------------------------------------------------- questions
    def questions(self, subject: str, topic_id: int | None = None,
                  assessment_type: str | None = None, qtype: str | None = None,
                  n: int = 10) -> pd.DataFrame:
        """Real past questions, newest first (never ones that may contain their own answer)."""
        g = self.usable[self.usable["subject"] == subject]
        if topic_id is not None:
            g = g[g["topic_id"] == topic_id]
        if assessment_type:
            g = g[g["assessment_type"] == assessment_type]
        if qtype:
            g = g[g["qtype"] == qtype]
        cols = ["year", "exam_kind", "question_text", "marks", "qtype", "difficulty", "topic_id"]
        out = g.sort_values(["year", "file_id", "q_index"], ascending=[False, True, True])[cols]
        return out.head(n).reset_index(drop=True)

    # -------------------------------------------------------------- blueprint
    def blueprint(self, subject: str, exam: str) -> dict:
        """Typical structure of this exam from real past papers (medians). exam: quiz, mid1,
        mid2, mid, final. Falls back to all subjects when the subject has < 3 such papers."""
        b = self.blueprints
        atype = "mid" if exam.startswith("mid") else "quiz" if exam.startswith("quiz") else exam
        mine = b[(b["subject"] == subject) & (b["exam_kind"] == exam)]
        if len(mine) < 3:
            mine = b[(b["subject"] == subject) & (b["assessment_type"] == atype)]
        scope, d = ("subject", mine) if len(mine) >= 3 else (
            "all subjects", b[b["assessment_type"] == atype])
        if d.empty:
            raise KeyError(f"no papers of type {atype!r}")
        stated = self.all_papers.set_index("file_id")["total_marks"].reindex(d["file_id"]).dropna()
        stated_all = self.all_papers[self.all_papers["assessment_type"] == atype]["total_marks"].dropna()
        if len(stated) >= 3:
            total = float(stated.median())
        else:
            known = d[d["marks_known"] >= 0.8]
            total = float(known["marks_sum"].median()) if len(known) >= 3 else float(stated_all.median())
        shares = {c[len("share_"):]: round(float(d[c].mean()), 3)
                  for c in d.columns if c.startswith("share_")}
        diff = {k: round(float(d[f"diff_{k}"].mean()), 3) for k in ("easy", "medium", "hard")}
        return {"subject": subject, "exam": exam, "scope": scope, "papers_used": int(len(d)),
                "questions": float(d["n_questions"].median()),
                "total_marks": total,
                "mean_bloom_level": round(float(d["mean_bloom"].mean()), 2),
                "difficulty_mix": diff,
                "question_type_shares": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
                "duration_minutes": default_duration(exam)}

    # ------------------------------------------------------------- past papers
    def paper(self, file_id: str, duration_minutes: int | None = None) -> dict:
        """A past paper as readable questions. Raises KeyError if its text could not be read."""
        g = self.readable[self.readable["file_id"] == file_id].sort_values("q_index")
        if g.empty:
            raise KeyError(f"no readable questions for file {file_id!r} (open the original instead)")
        first = g.iloc[0]
        qs, total = [], 0.0
        for number, r in enumerate(g.itertuples(index=False), start=1):
            marks = None if pd.isna(r.marks) else float(r.marks)
            total += marks or 0.0
            tid = None if pd.isna(r.topic_id) else int(r.topic_id)
            sol = r.solution_text if isinstance(r.solution_text, str) and r.solution_text else None
            qs.append({"number": number, "text": r.question_text, "marks": marks, "qtype": r.qtype,
                       "difficulty": r.difficulty, "topic_id": tid,
                       "topic_label": self.topic_label(first["subject"], tid), "solution": sol,
                       "solution_source": (r.solution_source if sol else None),
                       "low_quality": bool(r.low_quality), "needs_figure": bool(r.needs_figure)})
        kind = first["exam_kind"]
        return {"title": f"{first['subject']} - {label(kind)} {int(first['year'])}",
                "subject": first["subject"], "assessment_type": first["assessment_type"],
                "exam_kind": kind, "year": int(first["year"]), "source": "past_paper",
                "duration_minutes": int(duration_minutes or default_duration(kind)),
                "total_marks": total or None, "questions": qs}
