"""Paths and identifiers shared by the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "export"
SITE_DIR = ROOT / "site"

GOALS_FILE = CONFIG_DIR / "goals.yaml"
RESOLVED_IDS_FILE = CONFIG_DIR / "resolved_ids.json"

# Earliest publication year pulled from OpenAlex. KWF projects may start earlier;
# the pipeline lowers this to the first KWF project year when that is older.
START_YEAR = 2010

OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# Search terms used to resolve OpenAlex entity IDs. Resolved IDs are written to
# config/resolved_ids.json so they can be reviewed and pinned by hand.
IKNL_INSTITUTION_QUERIES = [
    "Netherlands Comprehensive Cancer Organisation",
    "Integraal Kankercentrum Nederland",
]
KWF_FUNDER_QUERIES = [
    "KWF Kankerbestrijding",
    "Dutch Cancer Society",
]
# Phrases that signal KWF funding in acknowledgement / full text.
KWF_ACK_PHRASES = ["KWF", "Dutch Cancer Society", "KWF Kankerbestrijding"]
