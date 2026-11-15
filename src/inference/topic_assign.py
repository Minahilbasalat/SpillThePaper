"""Map new question texts to the topics of a subject, using the SAME fitted TF-IDF + KMeans
that built the topic catalog (saved by src/data/topics.py in models/topics/)."""
from __future__ import annotations

import joblib

from src.inference import config as C

_CACHE: dict = {}


def _slug(subject: str) -> str:
    from src.data.topics import subject_slug
    return subject_slug(subject)


def has_topic_model(subject: str) -> bool:
    return (C.MODELS_DIR / "topics" / f"{_slug(subject)}.joblib").exists()


def assign_topics(subject: str, texts) -> list[int]:
    path = C.MODELS_DIR / "topics" / f"{_slug(subject)}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no topic model for '{subject}' (not a supported subject, or models/topics/ is "
            "missing: unzip the runtime bundle or run `python -m src.data.topics`)")
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    m = _CACHE[path]
    texts = list(texts)
    if not texts:
        return []
    return [int(t) for t in m["kmeans"].predict(m["vectorizer"].transform(texts))]
