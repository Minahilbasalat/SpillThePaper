"""Loads the trained models once and predicts topic probabilities for an upcoming exam.

predict(subject, exam, year, model="logreg", extra_papers=None, extra_incidence=None)
  exam: "quiz" | "mid1" | "mid2" | "mid" | "final"
  extra_*: papers a student uploaded (see src/inference/uploads.py)

Honest notes (also shown in the UI):
  * The models were trained on exam TYPE (quiz / mid / final). "mid1" and "mid2" use the mid
    model, then each topic is re-weighted by how often it appeared in earlier Mid 1 versus
    Mid 2 papers of the subject (shrunk towards the overall mid rate). It is applied only when
    the subject has at least 3 earlier papers of that kind (attrs["phase_adjusted"]).
  * Probabilities are rankings: they are over-estimated (see the Deliverable 2 reliability
    diagram), so compare topics with each other rather than reading them as exact chances.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.data.exam_kinds import exam_kind as make_kind
from src.inference import config as C
from src.inference.errors import LowConfidenceError
from src.inference.features import build_rows

MIN_PHASE_PAPERS = 3
PHASE_SHRINK = 3.0


class Predictor:
    def __init__(self, models_dir=C.MODELS_DIR):
        from src.models.preprocess import load_preprocessor

        self.models_dir = models_dir
        self.pre = load_preprocessor(models_dir / "preprocess.joblib")
        self._models: dict = {}                                 # loaded lazily

        q = pd.read_csv(C.QUESTION_TABLE_CSV, dtype={"file_id": str})
        q = q[q["topic_id"].notna()].copy()                     # supported subjects only
        q["year"] = q["year"].astype(int)
        q["topic_id"] = q["topic_id"].astype(int)
        self.catalog = pd.read_csv(C.TOPIC_CATALOG_CSV)
        self.elig = pd.read_csv(C.ELIGIBILITY_CSV)
        self.papers = (q[["file_id", "subject", "year", "assessment_type", "exam_kind"]]
                       .drop_duplicates("file_id").reset_index(drop=True))
        self.incidence = q[["file_id", "topic_id"]].drop_duplicates()
        self.thresholds = self._load_thresholds()

    # ------------------------------------------------------------ loading
    def _load_thresholds(self):
        thr = dict(C.DEFAULT_THRESHOLDS)
        try:
            m = pd.read_csv(C.METRICS_VAL_CSV)
            for name, t in zip(m["model"], m["threshold"]):
                if name in thr:
                    thr[name] = float(t)
        except FileNotFoundError:
            pass
        return thr

    def _model(self, name: str):
        if name not in self._models:
            if name == "mlp":
                self._models[name] = self._load_mlp(self.models_dir / C.MODEL_FILES["mlp"])
            elif name in ("logreg", "random_forest"):
                self._models[name] = joblib.load(self.models_dir / C.MODEL_FILES[name])
            else:
                raise ValueError(f"unknown model {name!r}")
        return self._models[name]

    @staticmethod
    def _load_mlp(path):
        import torch
        from src.models.mlp import MLP
        ckpt = torch.load(path, map_location="cpu")
        h1, h2 = ckpt["config"]["hidden"]
        net = MLP(ckpt["in_features"], h1, h2, ckpt["config"]["dropout"])
        net.load_state_dict(ckpt["state_dict"])
        return net.eval()

    def _score(self, name: str, X: np.ndarray) -> np.ndarray:
        if name == "mlp":
            from src.models.mlp import predict_proba
            return predict_proba(self._model("mlp"), X)
        return self._model(name).predict_proba(X)[:, 1]

    # ----------------------------------------------------------- catalogue
    def model_names(self):
        return list(C.MODEL_FILES)

    def subjects(self):
        return sorted(self.elig.loc[self.elig["eligible"], "subject"])

    def low_confidence_subjects(self):
        return sorted(self.elig.loc[~self.elig["eligible"], "subject"])

    def low_confidence_reason(self, subject):
        r = self.elig.loc[self.elig["subject"] == subject, "reason"]
        return str(r.iloc[0]) if len(r) else "subject not found"

    def topics(self, subject):
        t = self.catalog[self.catalog["subject"] == subject]
        return (t[["topic_id", "keywords"]].rename(columns={"keywords": "topic_label"})
                .sort_values("topic_id").reset_index(drop=True))

    # ------------------------------------------------------------- predict
    @staticmethod
    def _band(p, thr):
        return "Likely" if p >= thr else "Possible" if p >= 0.6 * thr else "Unlikely"

    @staticmethod
    def _split_exam(exam: str):
        exam = exam.lower().strip()
        if exam in ("mid1", "mid2"):
            return "mid", exam
        if exam in ("quiz", "mid", "final"):
            return exam, exam
        raise ValueError("exam must be one of: quiz, mid1, mid2, mid, final")

    def _history(self, extra_papers, extra_inc):
        papers, inc = self.papers, self.incidence
        if extra_papers is not None and len(extra_papers):
            papers = pd.concat([papers, extra_papers[papers.columns]], ignore_index=True)
            inc = pd.concat([inc, extra_inc], ignore_index=True)
        return papers, inc

    def _phase_factor(self, papers, inc, subject, year, kind, topic_ids):
        """Per-topic multiplier for mid1 / mid2 from earlier numbered mids; None if too little data."""
        p = papers[(papers["subject"] == subject) & (papers["year"] < year)]
        mids = p[p["exam_kind"].isin(["mid1", "mid2", "mid"])]
        mine = p[p["exam_kind"] == kind]
        if len(mine) < MIN_PHASE_PAPERS or len(mids) == 0:
            return None
        out = {}
        for t in topic_ids:
            has = set(inc.loc[inc["topic_id"] == t, "file_id"])
            overall = (mids["file_id"].isin(has).sum() + 0.5) / (len(mids) + 1)
            k_rate = (mine["file_id"].isin(has).sum() + PHASE_SHRINK * overall) / (len(mine) + PHASE_SHRINK)
            out[t] = k_rate / overall
        return out

    def predict(self, subject, exam, year, model="logreg",
                extra_papers=None, extra_incidence=None) -> pd.DataFrame:
        if subject not in self.subjects():
            raise LowConfidenceError(
                f"'{subject}': {self.low_confidence_reason(subject)}. No prediction is made.")
        target_type, kind = self._split_exam(exam)
        year = int(year)
        papers, inc = self._history(extra_papers, extra_incidence)
        topics = self.topics(subject)

        raw = build_rows(papers, inc, topics["topic_id"], subject, year, target_type)
        X = self.pre.transform(raw).to_numpy(dtype="float64")
        thr = self.thresholds[model]
        out = raw.merge(topics, on="topic_id")
        out["model_probability"] = self._score(model, X)

        factor = self._phase_factor(papers, inc, subject, year, kind, out["topic_id"]) \
            if kind in ("mid1", "mid2") else None
        out["probability"] = out["model_probability"]
        if factor:
            out["probability"] = (out["model_probability"] * out["topic_id"].map(factor)).clip(0.01, 0.99)

        # topics already examined earlier in the target year (data + uploads)
        same_year = papers[(papers["subject"] == subject) & (papers["year"] == year)]
        seen = {}
        for r in same_year.itertuples(index=False):
            for t in inc.loc[inc["file_id"] == r.file_id, "topic_id"]:
                seen.setdefault(int(t), set()).add(r.exam_kind)
        out["examined_this_year"] = out["topic_id"].map(lambda t: int(t) in seen)
        out["examined_in"] = out["topic_id"].map(lambda t: ", ".join(sorted(seen.get(int(t), []))))

        out["band"] = [self._band(p, thr) for p in out["probability"]]
        out["few_history"] = out["topic_count_prior"] < C.MIN_EVIDENCE
        out = out.sort_values("probability", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", out.index + 1)
        cols = ["topic_id", "topic_label", "probability", "model_probability", "rank", "band",
                "topic_count_prior", "topic_rate_prior", "years_since_last",
                "appeared_last_year", "few_history", "examined_this_year", "examined_in"]
        res = out[cols].copy()
        res.attrs.update(prior_papers=int(raw["prior_papers"].iloc[0]),
                         prior_years=int(raw["prior_years"].iloc[0]), threshold=thr, model=model,
                         exam=exam, year=year, phase_adjusted=bool(factor),
                         uploaded_papers=0 if extra_papers is None else int(len(extra_papers)))
        return res

    # --------------------------------------------------- details for the UI
    def history(self, subject, topic_id) -> pd.DataFrame:
        p = self.papers[self.papers["subject"] == subject]
        has = set(self.incidence.loc[self.incidence["topic_id"] == topic_id, "file_id"])
        g = (p.assign(appeared=p["file_id"].isin(has).astype(int))
             .groupby(["year", "assessment_type"], as_index=False)["appeared"].max())
        return g.sort_values("year").reset_index(drop=True)
