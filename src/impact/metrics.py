"""Impact-per-goal aggregation.

Each project/article can link to several goals. Totals use the fractional
`weight` from goals.classify (it sums to 1 per item), so goal totals add up to
the overall total without double counting. `*_any` columns count every item
linked to the goal at full weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def impact_per_goal(goals: pd.DataFrame, projects: pd.DataFrame, project_goals: pd.DataFrame,
                    works: pd.DataFrame, work_goals: pd.DataFrame) -> pd.DataFrame:
    p = project_goals.merge(projects, on="project_id", how="inner")
    w = work_goals.merge(works, on="openalex_id", how="inner")
    rows = []
    for g in goals.itertuples(index=False):
        pg = p[p.goal_id == g.goal_id]
        wg = w[w.goal_id == g.goal_id]
        rows.append({
            "goal_id": g.goal_id,
            "funding_eur": float((pg.amount_eur.fillna(0) * pg.weight).sum()),
            "funding_eur_iknl": float((pg.amount_eur.fillna(0) * pg.weight)[pg.is_iknl].sum()),
            "funding_eur_iknl_involved": float((pg.amount_eur.fillna(0) * pg.weight)[
                pg.get("iknl_involved", pg.is_iknl)].sum()),
            "projects": float(pg.weight.sum()),
            "projects_any": int(len(pg)),
            "articles": float(wg.weight.sum()),
            "articles_any": int(len(wg)),
            "articles_iknl": float(wg.weight[wg.has_iknl_author].sum()),
            "articles_kwf": float(wg.weight[wg.is_kwf].sum()),
            "articles_iknl_kwf": float(wg.weight[wg.has_iknl_author & wg.is_kwf].sum()),
            "citations": float((wg.cited_by_count * wg.weight).sum()),
            "mean_fwci": _wavg(wg.fwci, wg.weight),
            "share_top10": _wavg(wg.top10pct.astype(float), wg.weight),
            "share_oa": _wavg(wg.is_oa.astype(float), wg.weight),
            "policy_citations": float((wg.policy_citations.fillna(0) * wg.weight).sum())
            if "policy_citations" in wg else np.nan,
        })
    cols = [c for c in ["goal_id", "title_en", "title_nl", "theme", "featured", "ambition"] if c in goals]
    out = goals[cols].merge(pd.DataFrame(rows), on="goal_id")
    total_funding = out.funding_eur.sum()
    out["citations_per_meur"] = np.where(
        out.funding_eur > 0, out.citations / (out.funding_eur / 1e6), np.nan)
    out["funding_share"] = out.funding_eur / total_funding if total_funding else np.nan
    return out


def goal_year_trend(work_goals: pd.DataFrame, works: pd.DataFrame,
                    project_goals: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    w = work_goals.merge(works[["openalex_id", "year", "cited_by_count"]], on="openalex_id")
    w = w.groupby(["goal_id", "year"]).apply(
        lambda d: pd.Series({"articles": d.weight.sum(), "citations": (d.cited_by_count * d.weight).sum()}),
        include_groups=False).reset_index()
    p = project_goals.merge(projects[["project_id", "start_year", "amount_eur"]], on="project_id")
    p = p.assign(funding_eur=p.amount_eur.fillna(0) * p.weight).groupby(
        ["goal_id", "start_year"], as_index=False).agg(funding_eur=("funding_eur", "sum"), projects=("weight", "sum"))
    p = p.rename(columns={"start_year": "year"})
    out = w.merge(p, on=["goal_id", "year"], how="outer").fillna(0)
    out["year"] = out.year.astype(int)
    return out.sort_values(["goal_id", "year"]).reset_index(drop=True)
