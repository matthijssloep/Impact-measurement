"""End-to-end pipeline. Every step reads/writes Parquet in data/processed."""

from __future__ import annotations

import json
import logging
import shutil

import pandas as pd

from . import config, export, fetch, goals as goals_mod, kwf, metrics
from .openalex import OpenAlex

log = logging.getLogger(__name__)
P = config.PROCESSED_DIR
STLITE_VERSION = "1.9.2"  # @stlite/browser on npm; runs the Streamlit app in the browser


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(P / f"{name}.parquet")


def _write(df: pd.DataFrame, name: str) -> None:
    P.mkdir(parents=True, exist_ok=True)
    df.to_parquet(P / f"{name}.parquet", index=False)
    log.info("wrote %s (%d rows)", name, len(df))


def step_kwf() -> None:
    projects = kwf.add_iknl_flags(kwf.scrape_projects())
    _write(projects, "kwf_projects")
    # Human-readable copy (GitHub renders CSV as a table).
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    projects.assign(cancer_types=projects.cancer_types.map("; ".join)) \
        .to_csv(config.EXPORT_DIR / "kwf_projects.csv", index=False)


def step_openalex(start_year: int | None = None) -> None:
    client = OpenAlex()
    ids = fetch.resolve_ids(client)
    iknl_ids = [x["id"] for x in ids["iknl_institutions"]]
    funder_ids = [x["id"] for x in ids["kwf_funders"]]
    if not iknl_ids or not funder_ids:
        raise RuntimeError(f"Could not resolve IDs, check {config.RESOLVED_IDS_FILE}")

    projects = _read("kwf_projects") if (P / "kwf_projects.parquet").exists() else kwf.empty_projects()
    if start_year is None:
        start_year = config.START_YEAR
        if projects.start_year.notna().any():
            start_year = min(start_year, int(projects.start_year.min()))
    numbers = projects.project_number.dropna().tolist()

    iknl = fetch.fetch_iknl_works(client, iknl_ids, start_year)
    funded = fetch.fetch_kwf_funded_works(client, funder_ids, iknl_ids, start_year)
    # Grant-number search is resumable: earlier hits and searched numbers are kept.
    cache_path, done_path = P / "grant_search_hits.parquet", P / "grant_search_done.json"
    previous = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    new_hits, searched = fetch.fetch_grant_number_works(client, numbers, iknl_ids, start_year, done)
    by_number = pd.concat([previous, new_hits], ignore_index=True)
    if not by_number.empty:
        by_number = by_number.drop_duplicates(["openalex_id", "kwf_project_number"])
        _write(by_number, "grant_search_hits")
    done_path.write_text(json.dumps(sorted(done | searched)))
    remaining = len(set(numbers) - done - searched)
    if remaining:
        log.warning("Grant-number search incomplete: %d project numbers left; rerun `openalex` later", remaining)

    works, work_projects = assemble_works(iknl, funded, by_number, projects, funder_ids)
    _write(works, "works")
    _write(work_projects, "work_projects")


def assemble_works(iknl: pd.DataFrame, funded: pd.DataFrame, by_number: pd.DataFrame,
                   projects: pd.DataFrame, funder_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the three OpenAlex pulls into one works table plus article -> project links.

    Full-text grant-number hits published before the project's start year are
    dropped: a number that appears before the grant existed is a coincidence.
    """
    numbers = projects.project_number.dropna().tolist()
    if not by_number.empty:
        start = by_number.kwf_project_number.map(projects.set_index("project_number").start_year)
        keep = start.isna() | (by_number.year >= start)
        log.info("Grant-number hits: dropping %d of %d published before the project started",
                 int((~keep).sum()), len(by_number))
        by_number = by_number[keep.astype(bool)]

    links = [fetch.match_awards_to_projects(pd.concat([iknl, funded], ignore_index=True), numbers, funder_ids)]
    if not by_number.empty:
        links.append(by_number[["openalex_id", "kwf_project_number"]]
                     .rename(columns={"kwf_project_number": "project_number"})
                     .assign(evidence="grant_number_fulltext"))
    work_projects = pd.concat(links, ignore_index=True).drop_duplicates()
    work_projects = work_projects.merge(projects[["project_number", "project_id"]], on="project_number", how="left")

    evidence = pd.concat([
        funded[["openalex_id"]].assign(kwf_evidence="openalex_funder") if not funded.empty else None,
        by_number[["openalex_id"]].assign(kwf_evidence="grant_number_fulltext") if not by_number.empty else None,
    ]).drop_duplicates()
    evidence = evidence.groupby("openalex_id")["kwf_evidence"].agg(lambda s: ";".join(sorted(set(s))))

    works = pd.concat([iknl, funded.drop(columns="kwf_evidence", errors="ignore"),
                       by_number.drop(columns=["kwf_evidence", "kwf_project_number"], errors="ignore")],
                      ignore_index=True).drop_duplicates("openalex_id")
    works["kwf_evidence"] = works.openalex_id.map(evidence).fillna("")
    works["is_kwf"] = works.kwf_evidence != ""
    return works.reset_index(drop=True), work_projects


def step_classify() -> None:
    goal_list = goals_mod.load_goals()
    if not goal_list:
        raise RuntimeError(f"No goals defined in {config.GOALS_FILE}")
    _write(goals_mod.goals_frame(goal_list), "goals")

    projects = _read("kwf_projects")
    project_goals = goals_mod.classify(projects, goal_list, "project_id", "title",
                                       ["summary", "research_theme", "cancer_types"])
    _write(project_goals, "project_goals")

    works = _read("works")
    work_goals = goals_mod.classify(works, goal_list, "openalex_id", "title", ["abstract", "topic"])
    work_goals["goal_source"] = "keywords"

    # Articles without their own goal inherit the goals of the KWF project they came from.
    wp = _read("work_projects").dropna(subset=["project_id"])
    inherited = (wp.merge(project_goals, on="project_id")
                 .loc[lambda d: ~d.openalex_id.isin(work_goals.openalex_id)]
                 .groupby(["openalex_id", "goal_id"], as_index=False)
                 .agg(score=("score", "max"), confidence=("confidence", "max")))
    if not inherited.empty:
        inherited["matched_terms"] = ""
        inherited["weight"] = inherited.score / inherited.groupby("openalex_id").score.transform("sum")
        inherited["is_primary"] = inherited.score == inherited.groupby("openalex_id").score.transform("max")
        inherited["goal_source"] = "via_kwf_project"
        work_goals = pd.concat([work_goals, inherited], ignore_index=True)
    _write(work_goals, "work_goals")


def step_metrics() -> None:
    goals, projects, works = _read("goals"), _read("kwf_projects"), _read("works")
    project_goals, work_goals = _read("project_goals"), _read("work_goals")
    _write(metrics.impact_per_goal(goals, projects, project_goals, works, work_goals), "impact_per_goal")
    # KWF view: the same measures, counting only KWF-funded articles.
    kwf_works = works[works.is_kwf]
    _write(metrics.impact_per_goal(goals, projects, project_goals, kwf_works,
                                   work_goals[work_goals.openalex_id.isin(kwf_works.openalex_id)]),
           "impact_per_goal_kwf")
    _write(iknl_kwf_partners(works), "iknl_kwf_partners")


def iknl_kwf_partners(works: pd.DataFrame) -> pd.DataFrame:
    """Institutions co-authoring the IKNL articles that have KWF funding (IKNL itself excluded)."""
    joint = works[works.has_iknl_author & works.is_kwf].explode("institutions")
    joint = joint[~joint.institutions.fillna("").str.contains("Comprehensive Cancer Organisation|IKNL",
                                                             case=False)]
    return (joint.dropna(subset=["institutions"]).groupby("institutions").openalex_id.nunique()
            .rename("articles").reset_index().rename(columns={"institutions": "institution"})
            .sort_values("articles", ascending=False).reset_index(drop=True))
    _write(metrics.goal_year_trend(work_goals, works, project_goals, projects), "goal_year_trend")


def step_export() -> None:
    written = export.write_overton(_read("works"), _read("work_goals"))
    log.info("Overton export: %s", written)


APP_TABLES = {
    "goals": None,
    "impact_per_goal": None,
    "impact_per_goal_kwf": None,
    "iknl_kwf_partners": None,
    "goal_year_trend": None,
    "project_goals": ["project_id", "goal_id", "confidence", "weight", "is_primary"],
    "work_goals": ["openalex_id", "goal_id", "confidence", "weight", "is_primary", "goal_source"],
    "kwf_projects": ["project_id", "project_number", "title", "project_leader", "institution",
                     "amount_eur", "start_date", "start_year", "duration_months", "end_year", "status",
                     "research_theme", "funding_partner",
                     "is_iknl", "iknl_involved", "url"],
    "works": ["openalex_id", "doi", "pmid", "title", "year", "journal", "cited_by_count", "fwci",
              "top10pct", "is_oa", "has_iknl_author", "is_kwf", "kwf_evidence"],
    "work_projects": ["openalex_id", "project_id", "evidence"],
}


def step_site() -> None:
    """Copy the app and slim CSV copies of the tables into site/ for GitHub Pages.

    The in-browser runtime reads CSV, which avoids depending on pyarrow there.
    """
    out = config.SITE_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    files = {"streamlit_app.py": {"url": "./streamlit_app.py"}}
    for name, cols in APP_TABLES.items():
        path = P / f"{name}.parquet"
        if not path.exists():
            log.warning("skip %s: not built yet", name)
            continue
        df = pd.read_parquet(path)
        (df[cols] if cols else df).to_csv(out / f"{name}.csv", index=False)
        files[f"data/{name}.csv"] = {"url": f"./data/{name}.csv"}
    shutil.copy(config.ROOT / "app" / "streamlit_app.py", config.SITE_DIR / "streamlit_app.py")
    html = (config.ROOT / "app" / "index.template.html").read_text()
    html = html.replace("__STLITE_VERSION__", STLITE_VERSION).replace("__FILES__", json.dumps(files, indent=2))
    (config.SITE_DIR / "index.html").write_text(html)


STEPS = {
    "kwf": step_kwf,
    "openalex": step_openalex,
    "classify": step_classify,
    "metrics": step_metrics,
    "export": step_export,
    "site": step_site,
}
