import joblib
import numpy as np
import pandas as pd

from src.inference import config as C
from src.inference.features import build_rows
from src.inference.stub import LowConfidenceError


class Predictor:
    def __init__(self, models_dir=C.MODELS_DIR):
        self.pre = joblib.load(models_dir / "preprocess.joblib")
        self.models = {
            "logreg": joblib.load(models_dir / C.MODEL_FILES["logreg"]),
            "random_forest": joblib.load(models_dir / C.MODEL_FILES["random_forest"]),
            "mlp": self._load_mlp(models_dir / C.MODEL_FILES["mlp"]),
        }
        q = pd.read_csv(C.QUESTIONS_CSV)
        self.q = q
        self.topics_df = pd.read_csv(C.TOPICS_CSV)
        # ADAPT column names to your tables
        self.papers = (q[["paper_id", "subject", "year", "assessment_type"]]
                       .drop_duplicates("paper_id"))
        self.incidence = (q.dropna(subset=["topic_id"])[["paper_id", "topic_id"]]
                          .drop_duplicates())
        self._eligible = sorted(self.topics_df.subject.unique())

    # ---------- loading ----------
    @staticmethod
    def _load_mlp(path):
        import torch
        from src.models.mlp import MLP            # ADAPT class name and import
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            net = obj
        else:
            net = MLP(...)                         # ADAPT: 12 -> 16 -> 8 -> 1, dropout 0.4
            net.load_state_dict(obj.get("state_dict", obj))
        return net.eval()

    # ---------- catalogue ----------
    def model_names(self): return list(self.models)
    def subjects(self): return self._eligible
    def low_confidence_subjects(self):
        return sorted(set(self.papers.subject.unique()) - set(self._eligible))

    def topics(self, subject):
        t = self.topics_df[self.topics_df.subject == subject]
        return t[["topic_id", "topic_label"]].reset_index(drop=True)   # ADAPT label column

    # ---------- scoring ----------
    def _score(self, name, X):
        if name == "mlp":
            import torch
            with torch.no_grad():
                return torch.sigmoid(self.models["mlp"](
                    torch.tensor(np.asarray(X), dtype=torch.float32))).squeeze(1).numpy()
        return self.models[name].predict_proba(X)[:, 1]

    @staticmethod
    def _band(p, thr):
        return "Likely" if p >= thr else "Possible" if p >= 0.6 * thr else "Unlikely"

    def predict(self, subject, target_type, target_year, model="logreg"):
        if subject not in self._eligible:
            raise LowConfidenceError(
                f"'{subject}' has too little history (needs 6+ papers from 3+ years "
                "before 2023). No prediction is made.")
        topics = self.topics(subject)
        raw = build_rows(self.papers, self.incidence, topics.topic_id,
                         subject, target_year, target_type)
        X = self.pre.transform(raw)                # ADAPT: how your preprocessor is called
        thr = C.THRESHOLDS[model]
        out = raw.merge(topics, on="topic_id")
        out["probability"] = self._score(model, X)
        out["band"] = out.probability.map(lambda p: self._band(p, thr))
        out["few_history"] = out.topic_count_prior < C.MIN_EVIDENCE
        out = out.sort_values("probability", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", out.index + 1)
        out.attrs.update(prior_papers=int(raw.prior_papers.iloc[0]),
                         prior_years=int(raw.prior_years.iloc[0]), threshold=thr)
        return out[["topic_id", "topic_label", "probability", "rank", "band",
                    "topic_count_prior", "topic_rate_prior", "years_since_last",
                    "appeared_last_year", "few_history"]].assign()

    # ---------- details for the UI ----------
    def history(self, subject, topic_id):
        p = self.papers[self.papers.subject == subject]
        has = set(self.incidence.loc[self.incidence.topic_id == topic_id, "paper_id"])
        g = p.assign(appeared=p.paper_id.isin(has).astype(int)) \
             .groupby(["year", "assessment_type"], as_index=False).appeared.max()
        return g.sort_values("year")

    def questions(self, subject, topic_id, n=3):
        s = self.q[(self.q.subject == subject) & (self.q.topic_id == topic_id)]
        col = "question_text"                      # ADAPT
        return s.sort_values("year", ascending=False)[col].head(n).tolist()

    def assign_topics(self, subject, question_texts):
        raise NotImplementedError   # Step 9