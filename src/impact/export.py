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


def write_overton(works: pd.DataFrame, work_goals: pd.DataFrame, out_dir: Path = config.EXPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = overton_tables(works, work_goals)
    table.to_csv(out_dir / "overton_works.csv", index=False)
    written = {"overton_works.csv": len(table)}
    for col, name in (("doi", "dois.txt"), ("pmid", "pmids.txt")):
        values = sorted(table[col].dropna().unique())
        (out_dir / name).write_text("\n".join(values) + "\n")
        written[name] = len(values)
    # ORCIDs of IKNL-affiliated authors only (all co-authors would be far broader).
    orcids = sorted({o for xs in works.iknl_orcids for o in xs})
    (out_dir / "iknl_orcids.txt").write_text("\n".join(orcids) + "\n")
    written["iknl_orcids.txt"] = len(orcids)
    return written
