"""Minimal OpenAlex client and flattening of work records.

Docs: https://docs.openalex.org. Uses cursor paging (200 per page) and retries on
rate limits / server errors. Set OPENALEX_MAILTO and, if you have one,
OPENALEX_API_KEY in the environment.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"

WORK_FIELDS = [
    "id", "doi", "ids", "title", "publication_year", "publication_date", "type",
    "primary_location", "authorships", "cited_by_count", "fwci",
    "citation_normalized_percentile", "open_access", "primary_topic",
    "abstract_inverted_index", "grants", "funders", "awards", "language",
]


class OpenAlexError(RuntimeError):
    pass


class OpenAlexBudgetError(OpenAlexError):
    """Daily budget used up; retrying will not help until it resets."""


class InvalidSelectError(OpenAlexError):
    def __init__(self, message: str, valid: set[str]):
        super().__init__(message)
        self.valid = valid


def valid_select_fields(message: str) -> set[str]:
    m = re.search(r"Valid fields for select are:\s*(.+)", message)
    if not m:
        return set()
    return {f.strip(" .") for f in m.group(1).split(",") if f.strip(" .")}


class OpenAlex:
    def __init__(self, mailto: str | None = None, api_key: str | None = None,
                 session: requests.Session | None = None, pause: float = 0.12,
                 max_retries: int = 6):
        self.mailto = mailto if mailto is not None else config.OPENALEX_MAILTO
        self.api_key = api_key if api_key is not None else config.OPENALEX_API_KEY
        self.session = session or requests.Session()
        self.pause = pause
        self.max_retries = max_retries

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        params = dict(params or {})
        if self.mailto:
            params["mailto"] = self.mailto
        # Key goes in a header so it never appears in URLs, logs or error messages.
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(self.max_retries):
            resp = self.session.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 200:
                time.sleep(self.pause)
                return resp.json()
            if resp.status_code == 429 and "budget" in resp.text.lower():
                raise OpenAlexBudgetError(
                    "OpenAlex daily budget used up. Set OPENALEX_API_KEY (free key: "
                    "https://help.openalex.org/api/authentication/) or wait for the midnight UTC reset.")
            if resp.status_code == 400 and "valid select field" in resp.text:
                message = resp.json().get("message", resp.text)
                raise InvalidSelectError(message, valid_select_fields(message))
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                log.warning("OpenAlex %s -> %s, retrying in %ss", url, resp.status_code, wait)
                time.sleep(wait)
                continue
            raise OpenAlexError(f"{resp.status_code} for {resp.url}: {resp.text[:300]}")
        raise OpenAlexError(f"Gave up on {url} after {self.max_retries} attempts")

    def iter_results(self, path: str, params: dict[str, Any]) -> Iterator[dict]:
        params = {**params, "per-page": 200, "cursor": "*"}
        while params["cursor"]:
            page = self.get(path, params)
            yield from page.get("results", [])
            params["cursor"] = page.get("meta", {}).get("next_cursor")

    def count(self, path: str, params: dict[str, Any]) -> int:
        return self.get(path, {**params, "per-page": 1}).get("meta", {}).get("count", 0)

    def search_entities(self, entity: str, query: str) -> list[dict]:
        return self.get(entity, {"search": query, "per-page": 10}).get("results", [])

    def iter_works(self, filters: dict[str, str] | str, search: str | None = None,
                   fields: Iterable[str] = WORK_FIELDS) -> Iterator[dict]:
        fields = list(fields)
        params: dict[str, Any] = {"filter": build_filter(filters), "select": ",".join(fields)}
        if search:
            params["search"] = search
        try:
            first = self.get("works", {**params, "per-page": 1})
        except InvalidSelectError as exc:
            # OpenAlex renames fields over time: keep the ones it still accepts.
            if not exc.valid:
                raise
            dropped = [f for f in fields if f not in exc.valid]
            log.warning("OpenAlex no longer accepts select fields %s; dropping them", dropped)
            params["select"] = ",".join(f for f in fields if f in exc.valid)
            WORK_FIELDS[:] = [f for f in WORK_FIELDS if f in exc.valid]
        else:
            del first
        yield from self.iter_results("works", params)


def build_filter(filters: dict[str, str] | str) -> str:
    if isinstance(filters, str):
        return filters
    return ",".join(f"{k}:{v}" for k, v in filters.items())


def short_id(openalex_id: str | None) -> str | None:
    """'https://openalex.org/I123' -> 'I123'."""
    if not openalex_id:
        return None
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi or None


def normalize_pmid(pmid: str | None) -> str | None:
    if not pmid:
        return None
    m = re.search(r"(\d+)\s*$", str(pmid))
    return m.group(1) if m else None


def normalize_orcid(orcid: str | None) -> str | None:
    if not orcid:
        return None
    m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", orcid.upper())
    return m.group(1) if m else None


def normalize_award(award: str | None) -> str | None:
    """Uppercase alphanumerics only, so 'UVA 2014-7000' == 'uva2014 7000'."""
    if not award:
        return None
    norm = re.sub(r"[^0-9A-Z]", "", str(award).upper())
    return norm or None


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def funding_entries(work: dict) -> list[dict[str, str | None]]:
    """Funder/award pairs from the `grants`, `awards` and `funders` fields.

    OpenAlex has changed this schema over time, so every known shape is read.
    """
    out: list[dict[str, str | None]] = []
    for g in work.get("grants") or []:
        out.append({
            "funder_id": short_id(g.get("funder")),
            "funder_name": g.get("funder_display_name"),
            "award_id": g.get("award_id"),
        })
    for a in work.get("awards") or []:
        funder = a.get("funder") or {}
        out.append({
            "funder_id": short_id(a.get("funder_id") or (funder.get("id") if isinstance(funder, dict) else funder)),
            "funder_name": a.get("funder_display_name") or (funder.get("display_name") if isinstance(funder, dict) else None),
            "award_id": a.get("funder_award_id") or a.get("award_id"),
        })
    for f in work.get("funders") or []:
        out.append({"funder_id": short_id(f.get("id")), "funder_name": f.get("display_name"), "award_id": None})
    return out


def flatten_work(work: dict, iknl_ids: set[str] | None = None) -> dict:
    iknl_ids = iknl_ids or set()
    authorships = work.get("authorships") or []
    authors, orcids, iknl_orcids, institutions, countries = [], [], [], set(), set()
    has_iknl = False
    for a in authorships:
        author = a.get("author") or {}
        authors.append(author.get("display_name"))
        orcid = normalize_orcid(author.get("orcid"))
        if orcid:
            orcids.append(orcid)
        inst_ids = set()
        countries.update(c for c in a.get("countries") or [] if c)
        for inst in a.get("institutions") or []:
            institutions.add(inst.get("display_name"))
            if inst.get("country_code"):
                countries.add(inst["country_code"])
            inst_ids.add(short_id(inst.get("id")))
            inst_ids.update(short_id(x) for x in inst.get("lineage") or [])
        if inst_ids & iknl_ids:
            has_iknl = True
            if orcid:
                iknl_orcids.append(orcid)
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    pct = work.get("citation_normalized_percentile") or {}
    topic = work.get("primary_topic") or {}
    funding = funding_entries(work)
    ids = work.get("ids") or {}
    return {
        "openalex_id": short_id(work.get("id")),
        "doi": normalize_doi(work.get("doi")),
        "pmid": normalize_pmid(ids.get("pmid")),
        "title": work.get("title") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "language": work.get("language"),
        "journal": source.get("display_name"),
        "cited_by_count": work.get("cited_by_count") or 0,
        "fwci": work.get("fwci"),
        "citation_percentile": pct.get("value"),
        "top10pct": bool(pct.get("is_in_top_10_percent")),
        "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        "topic": topic.get("display_name"),
        "field": (topic.get("field") or {}).get("display_name"),
        "n_authors": len(authorships),
        "authors": [x for x in authors if x],
        "orcids": sorted(set(orcids)),
        "iknl_orcids": sorted(set(iknl_orcids)),
        "institutions": sorted(x for x in institutions if x),
        "countries": sorted(countries),  # ISO codes of author affiliations
        "has_iknl_author": has_iknl,
        "funder_ids": sorted({f["funder_id"] for f in funding if f["funder_id"]}),
        "funder_names": sorted({f["funder_name"] for f in funding if f["funder_name"]}),
        "award_ids": sorted({f["award_id"] for f in funding if f["award_id"]}),
        # "funder_id|award_id" so awards can be tied back to their funder
        "funder_awards": sorted({f"{f['funder_id']}|{f['award_id']}" for f in funding
                                 if f["funder_id"] and f["award_id"]}),
    }


def works_frame(works: Iterable[dict], iknl_ids: set[str] | None = None) -> pd.DataFrame:
    rows = [flatten_work(w, iknl_ids) for w in works]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("openalex_id").reset_index(drop=True)
    return df
