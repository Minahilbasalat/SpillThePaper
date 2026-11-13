"""Loads the trained models once and predicts topic probabilities for an exam."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.inference import config as C
from src.inference.errors import LowConfidenceError
from src.inference.features import build_rows


class Predictor:
    def __init__(self, models_dir=C.MODELS_DIR):
        from src.models.preprocess import load_preprocessor

        self.models_dir = models_dir
        self.pre = load_preprocessor(models_dir / "preprocess.joblib")
        self._models: dict = {}                                 # loaded lazily

        q = pd.read_csv(C.QUESTIONS_CSV, dtype={"file_id": str})
        q["year"] = q["year"].astype(int)
        self.q = q
        self.catalog = pd.read_csv(C.TOPIC_CATALOG_CSV)
        self.elig = pd.read_csv(C.ELIGIBILITY_CSV)
        self.papers = (q[["file_id", "subject", "year", "assessment_type"]]
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

    def predict(self, subject, target_type, target_year, model="logreg") -> pd.DataFrame:
        if subject not in self.subjects():
            raise LowConfidenceError(
                f"'{subject}': {self.low_confidence_reason(subject)}. No prediction is made.")
        topics = self.topics(subject)
        raw = build_rows(self.papers, self.incidence, topics["topic_id"],
                         subject, int(target_year), target_type)
        X = self.pre.transform(raw).to_numpy(dtype="float64")
        thr = self.thresholds[model]
        out = raw.merge(topics, on="topic_id")
        out["probability"] = self._score(model, X)
        out["band"] = [self._band(p, thr) for p in out["probability"]]
        out["few_history"] = out["topic_count_prior"] < C.MIN_EVIDENCE
        out = out.sort_values("probability", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", out.index + 1)
        cols = ["topic_id", "topic_label", "probability", "rank", "band",
                "topic_count_prior", "topic_rate_prior", "years_since_last",
                "appeared_last_year", "few_history"]
        res = out[cols].copy()
        res.attrs.update(prior_papers=int(raw["prior_papers"].iloc[0]),
                         prior_years=int(raw["prior_years"].iloc[0]),
                         threshold=thr, model=model)
        return res

    # --------------------------------------------------- details for the UI
    def history(self, subject, topic_id) -> pd.DataFrame:
        p = self.papers[self.papers["subject"] == subject]
        has = set(self.incidence.loc[self.incidence["topic_id"] == topic_id, "file_id"])
        g = (p.assign(appeared=p["file_id"].isin(has).astype(int))
             .groupby(["year", "assessment_type"], as_index=False)["appeared"].max())
        return g.sort_values("year").reset_index(drop=True)

    def questions(self, subject, topic_id, n=3):
        s = self.q[(self.q["subject"] == subject) & (self.q["topic_id"] == topic_id)
                   & (~self.q["is_solution"].astype(bool))]
        return s.sort_values("year", ascending=False)["q_text"].head(n).tolist()