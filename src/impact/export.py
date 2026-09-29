"""Identifier lists for Overton (which matches research by DOI, PubMed ID, ORCID or ISBN)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def source_label(row) -> str:
    if row.has_iknl_author and row.is_kwf:
        return "IKNL+KWF"
    return "IKNL" if row.has_iknl_author else "KWF"


def overton_tables(works: pd.DataFrame, work_goals: pd.DataFrame) -> pd.DataFrame:
    goals = (work_goals.sort_values("score", ascending=False)
             .groupby("openalex_id")["goal_id"].agg(lambda s: ";".join(s)))
    df = works.copy()
    df["source"] = df.apply(source_label, axis=1)
    df["goals"] = df.openalex_id.map(goals).fillna("")
    df["orcids"] = df.orcids.apply(lambda xs: ";".join(xs) if len(xs) else "")
    return df[["doi", "pmid", "openalex_id", "orcids", "source", "year", "goals", "title"]] \
        .sort_values(["year", "doi"], na_position="last").reset_index(drop=True)


def overton_identifiers(table: pd.DataFrame) -> pd.DataFrame:
    """One identifier per article for Overton: the DOI, else the PubMed ID.

    OpenAlex has no ISBNs; articles with neither DOI nor PMID (mostly
    dissertations and conference papers) cannot be looked up and are left out.
    """
    ids = table.assign(
        identifier=table.doi.fillna(table.pmid),
        identifier_type=table.doi.notna().map({True: "DOI", False: "PMID"}),
    ).dropna(subset=["identifier"])
    return ids[["identifier", "identifier_type", "source", "year", "goals", "openalex_id", "title"]] \
        .drop_duplicates("identifier").reset_index(drop=True)


def write_overton(works: pd.DataFrame, work_goals: pd.DataFrame, out_dir: Path = config.EXPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = overton_tables(works, work_goals)
    table.to_csv(out_dir / "overton_works.csv", index=False)
    written = {"overton_works.csv": len(table)}

    ids = overton_identifiers(table)
    ids.to_csv(out_dir / "overton_identifiers.csv", index=False)
    written["overton_identifiers.csv"] = len(ids)
    # Single-column upload lists; IKNL+KWF articles appear in both subsets.
    subsets = {
        "overton_upload_all.csv": ids,
        "overton_upload_iknl.csv": ids[ids.source.str.contains("IKNL")],
        "overton_upload_kwf.csv": ids[ids.source.str.contains("KWF")],
    }
    for name, subset in subsets.items():
        subset[["identifier"]].to_csv(out_dir / name, index=False)
        written[name] = len(subset)

    for col, name in (("doi", "dois.txt"), ("pmid", "pmids.txt")):
        values = sorted(table[col].dropna().unique())
        (out_dir / name).write_text("\n".join(values) + "\n")
        written[name] = len(values)
    # ORCIDs of IKNL-affiliated authors only. In Overton an ORCID finds all of that
    # person's research, not only these articles, so keep them apart.
    orcids = sorted({o for xs in works.iknl_orcids for o in xs})
    (out_dir / "iknl_orcids.txt").write_text("\n".join(orcids) + "\n")
    pd.DataFrame({"identifier": orcids}).to_csv(out_dir / "overton_upload_iknl_orcids.csv", index=False)
    written["iknl_orcids.txt"] = written["overton_upload_iknl_orcids.csv"] = len(orcids)
    return written
