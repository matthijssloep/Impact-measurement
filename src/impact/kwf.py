"""KWF research database (https://www.kwf.nl/onderzoek/onderzoeksdatabase).

KWF publishes no export. The database page is a React app on top of a Solr
index that the site exposes at POST /odb_query (filter index_id:kwf_odb); that
gives every project with its main fields. The project budget, duration and
team are only on each project page ("Projectgegevens"), so those pages are
fetched too. The database covers projects starting from 2017.
"""

from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://www.kwf.nl"
DATABASE_URL = f"{BASE_URL}/onderzoek/onderzoeksdatabase"
QUERY_URL = f"{BASE_URL}/odb_query"
USER_AGENT = "Impact-measurement research scraper (github.com/matthijssloep/Impact-measurement)"

# One row per project.
PROJECT_COLUMNS = {
    "project_id": "string",          # KWF project number
    "project_number": "string",
    "title": "string",
    "summary": "string",             # Dutch summary, plain text
    "project_leader": "string",
    "project_team": "string",
    "institution": "string",
    "amount_eur": "Float64",         # "Projectbudget" on the project page
    "start_date": "string",
    "start_year": "Int64",
    "duration_months": "Int64",
    "end_year": "Int64",             # start date + duration
    "status": "string",              # lopend / afgerond
    "research_theme": "string",      # "Onderzoeksthema" (primary modality)
    "funding_partner": "string",     # e.g. Alpe d'HuZes, Pink Ribbon, Gerichte Gift
    "cancer_types": "object",        # list[str]
    "url": "string",
    "scraped_at": "string",
}


def empty_projects() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in PROJECT_COLUMNS.items()})


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def fetch_index(session: requests.Session | None = None) -> list[dict]:
    session = session or _session()
    resp = session.post(QUERY_URL, data={"q": "*:*", "fq": "index_id:kwf_odb", "wt": "json",
                                         "rows": 100000, "start": 0}, timeout=120)
    resp.raise_for_status()
    body = resp.json()["response"]
    docs = body["docs"]
    if len(docs) != body["numFound"]:
        raise RuntimeError(f"Got {len(docs)} of {body['numFound']} projects")
    return docs


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _text(value) -> str:
    parts = value if isinstance(value, list) else [value or ""]
    return re.sub(r"\s+", " ", html.unescape(" ".join(p for p in parts if p))).strip()


def parse_index_doc(doc: dict) -> dict:
    start = doc.get("ds_field_project_start_date")
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
    number = doc.get("ss_field_project_number")
    return {
        "project_id": number,
        "project_number": number,
        "title": _text(_first(doc.get("tm_X3b_nl_field_project_title"))),
        "summary": _text(doc.get("tm_X3b_nl_project_summary")),
        "project_leader": _text(_first(doc.get("twm_X3b_nl_field_project_leader"))),
        "institution": doc.get("ss_project_institute"),
        "start_date": start_dt.date().isoformat() if start_dt else None,
        "start_year": start_dt.year if start_dt else None,
        "status": doc.get("ss_project_status"),
        "research_theme": doc.get("ss_project_primary_modality"),
        "funding_partner": doc.get("ss_name_funding_partner"),
        "cancer_types": list(doc.get("sm_project_disease_site_code_name") or []),
        "url": BASE_URL + doc["ss_url"] if doc.get("ss_url") else None,
    }


def parse_project_page(page: str) -> dict:
    """Read the "Projectgegevens" label/value list at the bottom of a project page."""
    start = page.find("Projectgegevens")
    if start < 0:
        return {}
    block = re.sub(r"<script.*?</script>|<svg.*?</svg>", "", page[start:start + 20000], flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", block)
    lines = [html.unescape(x).replace("\xa0", " ").strip() for x in text.split("\n")]
    lines = [x for x in lines if x]
    fields: dict[str, str] = {}
    label = None
    for line in lines[1:]:
        if line.endswith(":") and len(line) < 40:
            label = line[:-1].strip().lower()
            fields[label] = ""
        elif label:
            fields[label] = f"{fields[label]} {line}".strip()
        if line.startswith("Onze onderzoeken"):
            break
    out = {"project_team": fields.get("projectteam")}
    if m := re.search(r"(\d+)", fields.get("looptijd", "")):
        out["duration_months"] = int(m.group(1))
    if budget := fields.get("projectbudget"):
        digits = re.sub(r"[^\d,]", "", budget).split(",")[0]
        out["amount_eur"] = float(digits) if digits else None
    return out


def scrape_projects(pause: float = 0.4) -> pd.DataFrame:
    session = _session()
    docs = fetch_index(session)
    log.info("KWF index: %d projects", len(docs))
    rows = []
    for n, doc in enumerate(docs, 1):
        row = parse_index_doc(doc)
        if row["url"]:
            for attempt in range(4):
                try:
                    resp = session.get(row["url"], timeout=60)
                    if resp.status_code == 200:
                        row.update(parse_project_page(resp.text))
                        break
                    log.warning("%s -> %s", row["url"], resp.status_code)
                except requests.RequestException as exc:
                    log.warning("%s -> %s", row["url"], exc)
                time.sleep(2 ** attempt)
            time.sleep(pause)
        rows.append(row)
        if n % 100 == 0:
            log.info("  %d/%d project pages", n, len(docs))
    df = pd.DataFrame(rows)
    for col in PROJECT_COLUMNS:
        if col not in df:
            df[col] = None
    df["end_year"] = [
        (pd.Timestamp(s) + pd.DateOffset(months=int(d))).year if s and pd.notna(d) else None
        for s, d in zip(df.start_date, df.duration_months)
    ]
    df["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df = df[list(PROJECT_COLUMNS)].astype({k: v for k, v in PROJECT_COLUMNS.items() if v != "object"})
    return df.sort_values("project_number").reset_index(drop=True)


def is_iknl_institution(name: str | None) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ("iknl", "integraal kankercentrum", "comprehensive cancer organisation"))


def add_iknl_flags(projects: pd.DataFrame) -> pd.DataFrame:
    """is_iknl: IKNL leads the project; iknl_involved: IKNL leads or is in the project team."""
    projects = projects.copy()
    projects["is_iknl"] = projects.institution.map(is_iknl_institution).astype(bool)
    in_team = projects.project_team.fillna("").map(is_iknl_institution).astype(bool)
    projects["iknl_involved"] = projects.is_iknl | in_team
    return projects
