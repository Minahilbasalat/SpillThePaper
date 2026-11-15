"""Add papers uploaded by a student to the history used for one prediction.

ingest_papers(subject, files) -> UploadResult
  files: list of dicts {name, data (bytes), year, assessment_type ('quiz|mid|final'),
                        number (int or None)}
The uploads are used for that prediction only (they are not written to the shared data)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from src.data.clean_questions import is_instruction, normalised, strip_furniture
from src.data.exam_kinds import exam_kind
from src.data.question_features import build_question_table
from src.data.segment_questions import segment_questions
from src.data.text_quality import load_known_words
from src.inference.documents import extract_text
from src.inference.topic_assign import assign_topics, has_topic_model


@dataclass
class UploadResult:
    papers: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["file_id", "subject", "year", "assessment_type", "exam_kind"]))
    incidence: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["file_id", "topic_id"]))
    questions: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list = field(default_factory=list)

    @property
    def n_papers(self):
        return len(self.papers)


def ingest_papers(subject: str, files: list[dict]) -> UploadResult:
    res, papers, qs = UploadResult(), [], []
    if not has_topic_model(subject):
        res.warnings.append(f"'{subject}' has no topic model, so uploaded papers cannot be used.")
        return res
    seen = set()
    for f in files:
        name = f["name"]
        try:
            ex = extract_text(name, f["data"])
        except (ValueError, RuntimeError) as e:        # unsupported type, too large, OCR missing
            res.warnings.append(f"{name}: {e}")
            continue
        except Exception as e:                          # damaged or renamed file
            res.warnings.append(f"{name}: this file could not be read ({type(e).__name__}); "
                                "is it damaged or renamed?")
            continue
        if ex["warning"]:
            res.warnings.append(f"{name}: {ex['warning']}")
        chunks, method = segment_questions(ex["text"])
        chunks = [strip_furniture(c) for c in chunks]
        chunks = [c for c in chunks if not is_instruction(c, 1)]
        if not chunks:
            res.warnings.append(f"{name}: no questions could be found in this file")
            continue
        fid = "up-" + hashlib.sha1(f["data"]).hexdigest()[:10]
        if fid in seen:
            res.warnings.append(f"{name}: the same file was uploaded twice, it is counted once")
            continue
        seen.add(fid)
        kind = exam_kind(f["assessment_type"], f.get("number"))
        topics = assign_topics(subject, chunks)
        papers.append({"file_id": fid, "subject": subject, "year": int(f["year"]),
                       "assessment_type": f["assessment_type"], "exam_kind": kind})
        for i, (text, tid) in enumerate(zip(chunks, topics), start=1):
            qs.append({"file_id": fid, "subject": subject, "year": int(f["year"]),
                       "assessment_type": f["assessment_type"], "exam_kind": kind,
                       "is_solution": False, "q_index": i, "method": method,
                       "q_text": text, "topic_id": tid})
    if papers:
        res.papers = pd.DataFrame(papers)
        q = pd.DataFrame(qs)
        res.incidence = q[["file_id", "topic_id"]].drop_duplicates()
        res.questions = build_question_table(q, load_known_words())
    return res
