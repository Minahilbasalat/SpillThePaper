import numpy as np
import pandas as pd


from src.inference.errors import LowConfidenceError  # noqa: F401

class DummyPredictor:
    """Same interface as Predictor, with fake data, so the UI can be built early."""
    _subjects = ["Computer Networks", "Artificial Intelligence", "Database Systems"]

    def model_names(self): return ["logreg", "random_forest", "mlp"]
    def subjects(self): return self._subjects
    def low_confidence_subjects(self): return ["Some Rare Subject"]

    def topics(self, subject):
        return pd.DataFrame({"topic_id": range(6),
                             "topic_label": [f"keyword{i}, keyword{i+1}" for i in range(6)]})

    def predict(self, subject, target_type, target_year, model="logreg"):
        if subject not in self._subjects:
            raise LowConfidenceError("Too little history for this subject.")
        rng = np.random.default_rng(abs(hash((subject, target_type, target_year))) % 2**32)
        p = np.sort(rng.uniform(0.05, 0.9, 6))[::-1]
        df = pd.DataFrame({
            "topic_id": range(6),
            "topic_label": self.topics(subject).topic_label,
            "probability": p,
            "rank": range(1, 7),
            "band": ["Likely" if x >= 0.47 else "Possible" if x >= 0.3 else "Unlikely" for x in p],
            "topic_count_prior": rng.integers(1, 15, 6),
            "topic_rate_prior": rng.uniform(0, 1, 6),
            "years_since_last": rng.integers(1, 5, 6),
            "appeared_last_year": rng.integers(0, 2, 6),
            "few_history": rng.integers(0, 2, 6).astype(bool),
        })
        df.attrs.update(prior_papers=30, prior_years=8, threshold=0.47)
        return df

    def history(self, subject, topic_id):
        return pd.DataFrame({"year": range(2018, 2026),
                             "assessment_type": ["mid"] * 8,
                             "appeared": [1, 0, 1, 1, 0, 1, 0, 1]})

    def questions(self, subject, topic_id, n=3):
        return [f"Example question {i+1} about topic {topic_id}" for i in range(n)]

    def assign_topics(self, subject, question_texts):
        return [i % 6 for i, _ in enumerate(question_texts)]