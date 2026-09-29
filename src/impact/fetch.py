"""Pull IKNL and KWF-funded works from OpenAlex.

Scope (agreed):
* IKNL works: at least one author affiliated with IKNL (institution lineage).
* KWF works, two evidence routes:
  (a) OpenAlex lists KWF as funder;
  (b) a KWF project number from the KWF database appears in the full text
      together with a KWF phrase.
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from . import config
from .openalex import OpenAlex, OpenAlexError, normalize_award, short_id, works_frame

log = logging.getLogger(__name__)


def resolve_ids(client: OpenAlex, write: bool = True) -> dict:
    """Look up IKNL institution IDs and KWF funder IDs.

    If config/resolved_ids.json exists and has `"pinned": true`, it is used
    as-is so a human-reviewed selection is never overwritten.
    """
    if config.RESOLVED_IDS_FILE.exists():
        existing = json.loads(config.RESOLVED_IDS_FILE.read_text())
        if existing.get("pinned"):
            return existing

    def lookup(entity: str, queries: list[str], must_contain: list[str]) -> list[dict]:
        found: dict[str, dict] = {}
        for q in queries:
            for r in client.search_entities(entity, q):
                name = (r.get("display_name") or "")
                alts = " ".join(r.get("alternate_titles") or r.get("display_name_alternatives") or [])
                hay = f"{name} {alts}".lower()
                if any(m.lower() in hay for m in must_contain):
                    found[short_id(r["id"])] = {
                        "id": short_id(r["id"]), "name": name, "ror": r.get("ror"),
                        "works_count": r.get("works_count"),
                    }
        return list(found.values())

    ids = {
        "pinned": False,
        "iknl_institutions": lookup("institutions", config.IKNL_INSTITUTION_QUERIES,
                                    ["comprehensive cancer organisation", "integraal kankercentrum", "iknl"]),
        "kwf_funders": lookup("funders", config.KWF_FUNDER_QUERIES,
                              ["kwf", "dutch cancer society"]),
    }
    if write:
        config.RESOLVED_IDS_FILE.write_text(json.dumps(ids, indent=2))
    log.info("Resolved IDs: %s", json.dumps(ids, indent=2))
    return ids


def _date_filter(start_year: int) -> str:
    return f"from_publication_date:{start_year}-01-01"


def fetch_iknl_works(client: OpenAlex, iknl_ids: list[str], start_year: int) -> pd.DataFrame:
    flt = f"authorships.institutions.lineage:{'|'.join(iknl_ids)},{_date_filter(start_year)}"
    log.info("IKNL works: %s results", client.count("works", {"filter": flt}))
    return works_frame(client.iter_works(flt), set(iknl_ids))


def fetch_kwf_funded_works(client: OpenAlex, funder_ids: list[str], iknl_ids: list[str],
                           start_year: int) -> pd.DataFrame:
    """Route (a). Tries the current funder filter first, then the legacy one."""
    joined = "|".join(funder_ids)
    for key in ("funders.id", "awards.funder_id", "grants.funder"):
        flt = f"{key}:{joined},{_date_filter(start_year)}"
        try:
            n = client.count("works", {"filter": flt})
        except OpenAlexError as exc:
            log.info("Filter %s not supported (%s)", key, exc)
            continue
        log.info("KWF-funded works via %s: %s results", key, n)
        df = works_frame(client.iter_works(flt), set(iknl_ids))
        if not df.empty:
            df["kwf_evidence"] = "openalex_funder"
        return df
    raise OpenAlexError("No working funder filter found; check the OpenAlex docs")


def fetch_grant_number_works(client: OpenAlex, project_numbers: list[str], iknl_ids: list[str],
                             start_year: int) -> pd.DataFrame:
    """Route (b): full-text search for each KWF project number plus a KWF phrase.

    Returns one row per (work, project number) hit.
    """
    frames = []
    for number in sorted({n for n in project_numbers if n}):
        hits = []
        for phrase in ("KWF", "Dutch Cancer Society"):
            flt = f'fulltext.search:"{number}",fulltext.search:"{phrase}",{_date_filter(start_year)}'
            try:
                hits.extend(client.iter_works(flt))
            except OpenAlexError as exc:
                log.warning("Full-text search failed for %s: %s", number, exc)
        if hits:
            df = works_frame(hits, set(iknl_ids))
            df["kwf_project_number"] = number
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["kwf_evidence"] = "grant_number_fulltext"
    return out


def match_awards_to_projects(works: pd.DataFrame, project_numbers: list[str],
                             funder_ids: list[str]) -> pd.DataFrame:
    """Link works to KWF projects via KWF award IDs.

    A KWF award matches a project when, ignoring spaces and punctuation, it equals
    the project number, or its trailing number equals it ("KWF 10895", "KWF-UVA 10895").
    Only awards from the KWF funder IDs are used, so other funders' grant numbers
    cannot match by accident.
    """
    exact = {normalize_award(n): n for n in project_numbers if normalize_award(n)}
    by_number = {n.strip(): n for n in project_numbers if n and n.strip().isdigit()}
    funders = set(funder_ids)
    rows = []
    for rec in works[["openalex_id", "funder_awards"]].itertuples(index=False):
        for pair in rec.funder_awards:
            funder, _, award = pair.partition("|")
            if funder not in funders:
                continue
            project = exact.get(normalize_award(award))
            if project is None and (m := re.search(r"(\d{5})\s*$", award)):
                project = by_number.get(m.group(1))
            if project:
                rows.append({"openalex_id": rec.openalex_id, "project_number": project,
                             "evidence": "openalex_award_id"})
    return pd.DataFrame(rows, columns=["openalex_id", "project_number", "evidence"]).drop_duplicates()
