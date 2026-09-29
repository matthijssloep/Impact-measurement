"""KWF research database (https://www.kwf.nl/onderzoek/onderzoeksdatabase).

KWF publishes no export, so the database is scraped. The parser is written
against the live HTML, which this environment cannot reach yet; until then only
the target schema is defined here.
"""

from __future__ import annotations

import pandas as pd

DATABASE_URL = "https://www.kwf.nl/onderzoek/onderzoeksdatabase"

# One row per project. Lists are stored as Parquet list columns.
PROJECT_COLUMNS = {
    "project_id": "string",          # stable id: KWF project number, else URL slug
    "project_number": "string",      # KWF grant/project number as published
    "title": "string",
    "title_en": "string",
    "summary": "string",             # Dutch lay summary
    "summary_en": "string",          # English summary when published
    "project_leader": "string",
    "institution": "string",
    "city": "string",
    "amount_eur": "Float64",
    "start_year": "Int64",
    "end_year": "Int64",
    "status": "string",              # lopend / afgerond
    "funding_scheme": "string",      # e.g. Young Investigator Grant
    "cancer_types": "object",        # list[str]
    "research_areas": "object",      # list[str]
    "url": "string",
    "scraped_at": "string",
}


def empty_projects() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in PROJECT_COLUMNS.items()})


def scrape_projects() -> pd.DataFrame:
    raise NotImplementedError(
        "The KWF scraper needs network access to www.kwf.nl so the page structure "
        "can be inspected; allow that domain in the environment's network settings."
    )


def is_iknl_institution(name: str | None) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ("iknl", "integraal kankercentrum", "comprehensive cancer organisation"))
