"""Read-only access to the REAL data behind the app: subjects, topics, past questions,
typical paper structure and complete past papers.  No models are loaded here, so it is
fast and works on any machine that has the runtime bundle.

Paper structure (the same dictionary will be returned by the paper generator later, so a
page built on `Library.paper()` keeps working when generated papers arrive):

    {"title": str, "subject": str, "assessment_type": "quiz|mid|final", "year": int,
     "source": "past_paper" | "generated", "duration_minutes": int, "total_marks": float|None,
     "questions": [{"number": int, "text": str, "marks": float|None, "qtype": str,
                    "difficulty": "easy|medium|hard", "topic_id": int|None,
                    "topic_label": str, "solution": str|None}]}
"""
from __future__ import annotations

import pandas as pd

from src.inference import config as C

DEFAULT_DURATION_MIN = {"quiz": 30, "mid": 60, "final": 180}     # students may change these
TYPES = ("quiz", "mid", "final")
_SKIP_TYPES = {"fragment"}


class Library:
    def __init__(self):
        q = pd.read_csv(C.QUESTION_TABLE_CSV, dtype={"file_id": str})
        q["year"] = q["year"].astype(int)
        self.q = q
        self.catalog = pd.read_csv(C.TOPIC_CATALOG_CSV)
        self.elig = pd.read_csv(C.ELIGIBILITY_CSV)
        self.blueprints = pd.read_csv(C.BLUEPRINTS_CSV, dtype={"file_id": str})
        self._label = {(r.subject, int(r.topic_id)): r.keywords
                       for r in self.catalog.itertuples(index=False)}

    # ---------------------------------------------------------------- subjects
    def subjects(self) -> list[str]:
        return sorted(self.elig.loc[self.elig["eligible"], "subject"])

    def overview(self, subject: str) -> dict:
        e = self.elig[self.elig["subject"] == subject]
        if e.empty:
            raise KeyError(f"unknown subject {subject!r}")
        e = e.iloc[0]
        g = self.q[self.q["subject"] == subject]
        papers = g.drop_duplicates("file_id")
        return {
            "subject": subject,
            "supported": bool(e["eligible"]),
            "reason_if_unsupported": "" if e["eligible"] else str(e["reason"]),
            "papers": int(papers.shape[0]),
            "questions": int(g.shape[0]),
            "first_year": int(papers["year"].min()) if len(papers) else None,
            "last_year": int(papers["year"].max()) if len(papers) else None,
            "papers_by_type": papers["assessment_type"].value_counts().to_dict(),
            "topics": int((self.catalog["subject"] == subject).sum()),
        }

    def topics(self, subject: str) -> pd.DataFrame:
        t = self.catalog[self.catalog["subject"] == subject]
        return (t[["topic_id", "keywords", "all_questions"]]
                .rename(columns={"keywords": "topic_label", "all_questions": "questions"})
                .sort_values("topic_id").reset_index(drop=True))

    # --------------------------------------------------------------- questions
    def questions(self, subject: str, topic_id: int | None = None,
                  assessment_type: str | None = None, qtype: str | None = None,
                  n: int = 10) -> pd.DataFrame:
        """Real past questions, newest first. Questions that may contain their own answer
        text (solution papers whose answer could not be separated) are never returned."""
        g = self.q[(self.q["subject"] == subject) & (~self.q["contaminated"].astype(bool))
                   & (~self.q["qtype"].isin(_SKIP_TYPES))]
        if topic_id is not None:
            g = g[g["topic_id"] == topic_id]
        if assessment_type:
            g = g[g["assessment_type"] == assessment_type]
        if qtype:
            g = g[g["qtype"] == qtype]
        cols = ["year", "assessment_type", "question_text", "marks", "qtype", "difficulty",
                "topic_id"]
        out = g.sort_values(["year", "file_id", "q_index"], ascending=[False, True, True])[cols]
        return out.head(n).reset_index(drop=True)

    # --------------------------------------------------------------- blueprint
    def blueprint(self, subject: str, assessment_type: str) -> dict:
        """What a typical paper of this kind looks like (medians over real past papers).
        Falls back to all subjects when this subject has fewer than 3 such papers."""
        b = self.blueprints
        mine = b[(b["subject"] == subject) & (b["assessment_type"] == assessment_type)]
        scope, d = ("subject", mine) if len(mine) >= 3 else (
            "all subjects", b[b["assessment_type"] == assessment_type])
        if d.empty:
            raise KeyError(f"no papers of type {assessment_type!r}")
        known = d[d["marks_known"] >= 0.8]
        shares = {c[len("share_"):]: round(float(d[c].mean()), 3)
                  for c in d.columns if c.startswith("share_")}
        return {
            "subject": subject, "assessment_type": assessment_type, "scope": scope,
            "papers_used": int(len(d)),
            "questions": float(d["n_questions"].median()),
            "total_marks": float(known["marks_sum"].median()) if len(known) else None,
            "mean_bloom_level": round(float(d["mean_bloom"].mean()), 2),
            "question_type_shares": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
            "duration_minutes": DEFAULT_DURATION_MIN[assessment_type],
        }

    # -------------------------------------------------------------- past papers
    def past_papers(self, subject: str) -> pd.DataFrame:
        """Complete real papers of a subject that can be shown to a student."""
        g = self.q[(self.q["subject"] == subject) & (~self.q["contaminated"].astype(bool))
                   & (~self.q["qtype"].isin(_SKIP_TYPES))]
        agg = g.groupby("file_id").agg(
            year=("year", "first"), assessment_type=("assessment_type", "first"),
            questions=("q_index", "count"), marks=("marks", "sum"),
            marks_known=("marks", lambda s: s.notna().mean())).reset_index()
        agg["marks"] = agg["marks"].where(agg["marks_known"] >= 0.8)
        return agg.sort_values(["year", "assessment_type"], ascending=[False, True]) \
                  .reset_index(drop=True)

    def paper(self, file_id: str, duration_minutes: int | None = None) -> dict:
        g = self.q[(self.q["file_id"] == file_id) & (~self.q["contaminated"].astype(bool))
                   & (~self.q["qtype"].isin(_SKIP_TYPES))].sort_values("q_index")
        if g.empty:
            raise KeyError(f"no usable questions for file {file_id!r}")
        first = g.iloc[0]
        qs, total = [], 0.0
        for number, r in enumerate(g.itertuples(index=False), start=1):
            marks = None if pd.isna(r.marks) else float(r.marks)
            total += marks or 0.0
            tid = None if pd.isna(r.topic_id) else int(r.topic_id)
            qs.append({"number": number, "text": r.question_text, "marks": marks,
                       "qtype": r.qtype, "difficulty": r.difficulty, "topic_id": tid,
                       "topic_label": self._label.get((first["subject"], tid), ""),
                       "solution": (r.solution_text or None) if isinstance(r.solution_text, str)
                       and r.solution_text else None})
        atype = first["assessment_type"]
        return {"title": f"{first['subject']} - {atype} {int(first['year'])} (past paper)",
                "subject": first["subject"], "assessment_type": atype,
                "year": int(first["year"]), "source": "past_paper",
                "duration_minutes": int(duration_minutes or DEFAULT_DURATION_MIN.get(atype, 60)),
                "total_marks": total or None, "questions": qs}
