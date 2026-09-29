import pandas as pd

from impact.kwf import add_iknl_flags, parse_index_doc, parse_project_page

PAGE = """<main><h2>Projectgegevens</h2><dl>
<dt>Projectnummer:</dt><dd>14728</dd>
<dt>Projectleider:</dt><dd>dr. ir. Esther Warnert</dd>
<dt>Projectteam:</dt><dd>dr. A (UMC Utrecht) -<br> dr. B (IKNL)</dd>
<dt>Looptijd:</dt><dd>48 maanden</dd>
<dt>Projectbudget:</dt><dd>&euro; 781.776,-</dd>
</dl><h2>Onze onderzoeken</h2></main>"""


def test_parse_project_page():
    out = parse_project_page(PAGE)
    assert out["amount_eur"] == 781776.0
    assert out["duration_months"] == 48
    assert out["project_team"] == "dr. A (UMC Utrecht) - dr. B (IKNL)"


def test_parse_index_doc():
    row = parse_index_doc({
        "ss_field_project_number": "14728", "tm_X3b_nl_field_project_title": ["Titel"],
        "tm_X3b_nl_project_summary": ["Deel een.", "Deel &amp; twee."],
        "twm_X3b_nl_field_project_leader": ["dr. X"], "ds_field_project_start_date": "2023-10-01T12:00:00Z",
        "sm_project_disease_site_code_name": ["Hersentumor"], "ss_url": "/onderzoek/onderzoeksdatabase/x",
    })
    assert row["start_year"] == 2023 and row["start_date"] == "2023-10-01"
    assert row["summary"] == "Deel een. Deel & twee."
    assert row["url"] == "https://www.kwf.nl/onderzoek/onderzoeksdatabase/x"


def test_iknl_flags():
    df = add_iknl_flags(pd.DataFrame({
        "institution": ["Integraal Kankercentrum Nederland (IKNL)", "UMC Utrecht", "UMC Utrecht"],
        "project_team": [None, "dr. B (IKNL)", "dr. C (LUMC)"],
    }))
    assert df.is_iknl.tolist() == [True, False, False]
    assert df.iknl_involved.tolist() == [True, True, False]
