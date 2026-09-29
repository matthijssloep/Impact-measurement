import pandas as pd

from impact.goals import Goal, classify, fold, score_text

GOALS = [
    Goal("G01", "Survivorship", terms=["quality of life", "kwaliteit van leven", "survivorship"]),
    Goal("G02", "Early detection", terms=["screening", "early detection", "vroege opsporing"],
         exclude=["lung screening trial"]),
]


def test_fold_removes_accents_and_case():
    assert fold("Kwaliteit  van Léven") == "kwaliteit van leven"


def test_word_boundaries_and_hyphens():
    g = Goal("X", "x", terms=["follow up", "screen"])
    assert score_text(g, "Long-term follow-up", "")[0] > 0
    assert score_text(g, "screening", "")[0] == 0  # 'screen' does not match inside 'screening'


def test_title_weighs_more_than_body():
    title_hit, _ = score_text(GOALS[0], "Quality of life", "")
    body_hit, _ = score_text(GOALS[0], "", "quality of life")
    assert title_hit > body_hit


def test_exclude_vetoes():
    assert score_text(GOALS[1], "Results of a lung screening trial", "")[0] == 0


def test_classify_multilabel_weights_sum_to_one():
    items = pd.DataFrame([
        {"id": "a", "title": "Screening and quality of life", "summary": ""},
        {"id": "b", "title": "Kwaliteit van leven na kanker", "summary": "survivorship"},
        {"id": "c", "title": "Unrelated", "summary": ""},
    ])
    links = classify(items, GOALS, "id", "title", ["summary"])
    assert set(links.id) == {"a", "b"}
    assert links.groupby("id").weight.sum().round(6).eq(1).all()
    assert set(links[links.id == "a"].goal_id) == {"G01", "G02"}
