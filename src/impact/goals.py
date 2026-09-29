"""Keyword-rule classifier that links projects and articles to Cancer Agenda goals.

Rules live in config/goals.yaml, per goal:

    - id: G01
      title_en: ...
      title_nl: ...
      keywords:            # English and Dutch terms, matched on word boundaries
        en: [survivorship, "quality of life"]
        nl: [overleving, "kwaliteit van leven"]
      exclude: [...]       # optional: any hit vetoes the goal for that item

A term starting with "re:" is used as a regular expression. Matching is case-
and accent-insensitive. Hits in the title weigh TITLE_WEIGHT, hits in the body
(abstract / summary) weigh 1. Each unique term counts once per field.
An item can link to several goals (multi-label).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config

TITLE_WEIGHT = 3.0
MIN_SCORE = 2.0      # minimum score for a link
CONF_SCALE = 4.0     # confidence = 1 - exp(-score / CONF_SCALE)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower())


def compile_term(term: str) -> re.Pattern:
    if term.startswith("re:"):
        return re.compile(term[3:], re.IGNORECASE)
    words = fold(term).split(" ")
    # Allow a hyphen or space between words: "follow-up" == "follow up"
    body = r"[\s\-]+".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![\w]){body}(?![\w])")


@dataclass
class Goal:
    id: str
    title_en: str
    title_nl: str = ""
    theme: str = ""
    featured: bool = False
    ambition: str = ""
    terms: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._terms = [(t, compile_term(t)) for t in self.terms]
        self._exclude = [compile_term(t) for t in self.exclude]


def load_goals(path: Path = config.GOALS_FILE) -> list[Goal]:
    raw = yaml.safe_load(path.read_text()) or {}
    goals = []
    for g in raw.get("goals") or []:
        kw = g.get("keywords") or {}
        terms = list(kw.get("en") or []) + list(kw.get("nl") or [])
        goals.append(Goal(id=str(g["id"]), title_en=g.get("title_en", ""), title_nl=g.get("title_nl", ""),
                          theme=g.get("theme", ""), featured=bool(g.get("featured")),
                          ambition=g.get("ambition", ""), terms=terms, exclude=list(g.get("exclude") or [])))
    return goals


def score_text(goal: Goal, title: str, body: str) -> tuple[float, list[str]]:
    title_f, body_f = fold(title), fold(body)
    full = f"{title_f} {body_f}"
    if any(p.search(full) for p in goal._exclude):
        return 0.0, []
    score, matched = 0.0, []
    for term, pat in goal._terms:
        in_title = bool(pat.search(title_f))
        in_body = bool(pat.search(body_f))
        if in_title or in_body:
            score += (TITLE_WEIGHT if in_title else 0) + (1 if in_body else 0)
            matched.append(term)
    return score, matched


def as_text(value) -> str:
    """Join list-like cells (Parquet lists come back as arrays); blank for missing."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value)
    return "" if pd.isna(value) else str(value)


def classify(items: pd.DataFrame, goals: list[Goal], id_col: str, title_col: str,
             body_cols: list[str], min_score: float = MIN_SCORE) -> pd.DataFrame:
    """Return one row per (item, goal) link with score, confidence and matched terms."""
    rows = []
    for rec in items.to_dict("records"):
        body = " ".join(as_text(rec.get(c)) for c in body_cols)
        for goal in goals:
            score, matched = score_text(goal, as_text(rec.get(title_col)), body)
            if score >= min_score:
                rows.append({
                    id_col: rec[id_col], "goal_id": goal.id, "score": score,
                    "confidence": round(1 - math.exp(-score / CONF_SCALE), 3),
                    "matched_terms": "; ".join(matched),
                })
    links = pd.DataFrame(rows, columns=[id_col, "goal_id", "score", "confidence", "matched_terms"])
    if not links.empty:
        # Fractional weight so totals across goals add up to one per item.
        links["weight"] = links["score"] / links.groupby(id_col)["score"].transform("sum")
        links["is_primary"] = links["score"] == links.groupby(id_col)["score"].transform("max")
    else:
        links["weight"] = pd.Series(dtype=float)
        links["is_primary"] = pd.Series(dtype=bool)
    return links


def goals_frame(goals: list[Goal]) -> pd.DataFrame:
    return pd.DataFrame([{"goal_id": g.id, "title_en": g.title_en, "title_nl": g.title_nl,
                          "theme": g.theme, "featured": g.featured, "ambition": g.ambition,
                          "n_terms": len(g.terms)} for g in goals])
