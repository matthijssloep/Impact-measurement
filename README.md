# Impact-measurement

Measure the impact of KWF grant money, IKNL research and published articles on
the 20 goals of the [Netherlands Cancer Agenda 2.1](https://nederlandskankercollectief.nl/app/uploads/2026/07/2026-0730-The-Netherlands-Cancer-Agenda-2.1.pdf),
with a bilingual (NL/EN) Streamlit dashboard published on GitHub Pages.

## Scope

| Data | Source | Rule |
|---|---|---|
| KWF projects | [KWF onderzoeksdatabase](https://www.kwf.nl/onderzoek/onderzoeksdatabase) (scraped, no export exists) | all projects, all fields |
| IKNL articles | OpenAlex | at least one IKNL-affiliated author, from 2010 (or the first KWF project year) |
| KWF-funded articles | OpenAlex | (a) KWF listed as funder, or (b) a KWF project number in the full text next to "KWF" / "Dutch Cancer Society" |
| Goals | Cancer Agenda 2.1 | transparent EN + NL keyword rules in `config/goals.yaml` (multi-label); API classification can be added later |
| Policy impact | Overton (later) | via `data/export/` identifier lists |

## Pipeline

```bash
pip install -r requirements.txt
export OPENALEX_MAILTO=you@example.org      # polite pool; OPENALEX_API_KEY optional
python scripts/run_pipeline.py              # all steps, or name steps:
python scripts/run_pipeline.py kwf openalex classify metrics export site
```

| Step | Output (`data/processed/*.parquet` unless noted) |
|---|---|
| `kwf` | `kwf_projects` |
| `openalex` | `works`, `work_projects`; resolved IDs in `config/resolved_ids.json` (set `"pinned": true` after review) |
| `classify` | `goals`, `project_goals`, `work_goals` |
| `metrics` | `impact_per_goal`, `goal_year_trend` |
| `export` | `data/export/overton_works.csv`, `dois.txt`, `pmids.txt`, `iknl_orcids.txt` |
| `site` | `site/` (app + CSV tables + `index.html`) for GitHub Pages |

Goal totals are **fractional**: an item linked to several goals is split across
them, so totals add up without double counting.

## Dashboard

* Locally: `python scripts/run_pipeline.py site && streamlit run site/streamlit_app.py`
* GitHub Pages: `.github/workflows/pages.yml` builds `site/` from the committed
  Parquet files on every push to `main` and runs the same app in the browser
  with [stlite](https://github.com/whitphx/stlite). Enable it once under
  *Settings → Pages → Source: GitHub Actions*.

## Status

- [x] OpenAlex client, IKNL / KWF queries, grant-number matching
- [x] Keyword goal classifier, impact-per-goal metrics, Overton export
- [x] Bilingual dashboard (tested locally on sample data)
- [x] KWF scraper: 1,029 projects (start 2017+), €731M; `data/export/kwf_projects.csv`
- [x] 20 goals + EN/NL keyword rules in `config/goals.yaml`
- [x] First real data run (29 Sep 2026): 12,404 articles (3,749 IKNL, 9,074 KWF-funded, 419 both),
      4,336 article→project links to 584 KWF projects; Overton lists in `data/export/`
- [ ] Grant-number full-text search: 248 of 1,029 project numbers done (OpenAlex daily budget);
      rerun `python scripts/run_pipeline.py openalex classify metrics export site` to resume
- [ ] Known limitations: mean FWCI is skewed by outliers (use median); citations per €M
      compares articles since 2010 with KWF funding since 2017; 4,735 articles match no goal
