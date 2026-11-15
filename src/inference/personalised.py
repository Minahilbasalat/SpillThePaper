"""Feature 5c: a personalised paper = the predicted paper, tilted towards the student's weak topics.

    generate_personalised(generator, subject, exam, year, analysis=None, store=None,
                          source_paper=None, **generate_kwargs) -> paper dict

The paper has the same structure as a predicted paper (so the same pages show it), plus
  paper["personalised"] = True and paper["focus"] = [{"topic_id", "topic_label", "weakness"}]
and every question has "focus" (True = it targets a weak topic). About half of the main questions
come from weak topics, the others still follow the prediction, and the "less-predicted"
questions stay, so a student is not drilled only on what went wrong. The paper they just
attempted can be excluded (source_paper) so they get new questions.
"""
from __future__ import annotations

WEAK_FROM = 0.4            # weakness >= 0.4 means 60% or less of the marks were earned
MAX_FOCUS = 6
CURRENT_WEIGHT = 0.7       # the latest attempt counts more than the history


def weak_focus(subject: str, analysis: dict | None = None, store=None) -> dict[int, dict]:
    """{topic_id: {"topic_label", "weakness"}} from this attempt (70%) and earlier ones (30%)."""
    now = {t["topic_id"]: t for t in (analysis or {}).get("topics", [])}
    hist = store.weakness(subject) if store is not None else {}
    out = {}
    for tid in set(now) | set(hist):
        if tid in now and tid in hist:
            w = CURRENT_WEIGHT * now[tid]["weakness"] + (1 - CURRENT_WEIGHT) * hist[tid]["weakness"]
        elif tid in now:
            w = now[tid]["weakness"]
        else:
            w = hist[tid]["weakness"] * 0.8          # not in this paper: slightly less certain
        if w >= WEAK_FROM:
            label = (now.get(tid) or hist.get(tid))["topic_label"]
            out[int(tid)] = {"topic_label": label, "weakness": round(w, 3)}
    top = sorted(out.items(), key=lambda kv: -kv[1]["weakness"])[:MAX_FOCUS]
    return dict(top)


def generate_personalised(generator, subject: str, exam: str, year: int, analysis: dict | None = None,
                          store=None, source_paper: dict | None = None, **kwargs) -> dict:
    if analysis is None and (store is None or not store.attempts(subject)):
        raise ValueError("there are no marked answers yet: mark a paper first")
    focus = weak_focus(subject, analysis, store)
    avoid = [q["text"] for q in source_paper["questions"]] if source_paper else None
    paper = generator.generate(subject, exam, year, focus={t: v["weakness"] for t, v in focus.items()},
                               avoid_texts=avoid, **kwargs)
    paper["title"] = paper["title"].replace("Predicted", "Personalised", 1)
    paper["personalised"] = True
    paper["focus"] = [{"topic_id": t, **v} for t, v in focus.items()]
    if not focus:
        paper["notes"].insert(0, "No clear weak area was found, so this is the normal predicted paper "
                                 "(without the questions you just attempted).")
    return paper
