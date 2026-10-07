from impact.openalex import (flatten_work, funding_entries, normalize_award, normalize_doi,
                             normalize_orcid, normalize_pmid, reconstruct_abstract)

WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1000/ABC.1",
    "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345"},
    "title": "Survival after surgery",
    "publication_year": 2020,
    "abstract_inverted_index": {"Registry": [0], "based": [1], "study": [2]},
    "authorships": [
        {"author": {"display_name": "A", "orcid": "https://orcid.org/0000-0001-2345-678X"},
         "institutions": [{"id": "https://openalex.org/I9", "display_name": "IKNL", "lineage": ["https://openalex.org/I9"]}]},
        {"author": {"display_name": "B", "orcid": None},
         "institutions": [{"id": "https://openalex.org/I1", "display_name": "UMC", "lineage": []}]},
    ],
    "grants": [{"funder": "https://openalex.org/F5", "funder_display_name": "KWF", "award_id": "UVA 2014-7000"}],
    "citation_normalized_percentile": {"value": 0.95, "is_in_top_10_percent": True},
    "open_access": {"is_oa": True},
    "cited_by_count": 7,
}


def test_normalizers():
    assert normalize_doi("https://doi.org/10.1/X") == "10.1/x"
    assert normalize_doi("doi:10.1/x") == "10.1/x"
    assert normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/999") == "999"
    assert normalize_orcid("https://orcid.org/0000-0001-2345-678x") == "0000-0001-2345-678X"
    assert normalize_award("UVA 2014-7000") == normalize_award("uva2014 7000") == "UVA20147000"


def test_reconstruct_abstract():
    assert reconstruct_abstract({"b": [1], "a": [0], "c": [2]}) == "a b c"
    assert reconstruct_abstract(None) == ""


def test_funding_entries_handles_new_schema():
    work = {"awards": [{"funder_id": "https://openalex.org/F5", "funder_award_id": "12345"}],
            "funders": [{"id": "https://openalex.org/F5", "display_name": "KWF"}]}
    entries = funding_entries(work)
    assert {"funder_id": "F5", "funder_name": None, "award_id": "12345"} in entries
    assert any(e["funder_name"] == "KWF" for e in entries)


def test_flatten_work():
    row = flatten_work(WORK, iknl_ids={"I9"})
    assert row["openalex_id"] == "W1"
    assert row["doi"] == "10.1000/abc.1"
    assert row["pmid"] == "12345"
    assert row["abstract"] == "Registry based study"
    assert row["has_iknl_author"] is True
    assert row["iknl_orcids"] == ["0000-0001-2345-678X"]
    assert row["funder_ids"] == ["F5"]
    assert row["award_ids"] == ["UVA 2014-7000"]
    assert row["top10pct"] is True
    assert flatten_work(WORK, iknl_ids={"I1234"})["has_iknl_author"] is False


def test_valid_select_fields_parsed_from_error():
    from impact.openalex import valid_select_fields
    msg = ("grants is not a valid select field. Valid fields for select are: id, doi, title, "
           "funders, awards.")
    assert valid_select_fields(msg) == {"id", "doi", "title", "funders", "awards"}


def test_invalid_select_is_dropped(monkeypatch):
    from impact import openalex as oa
    client = oa.OpenAlex(mailto="", api_key="", pause=0)
    calls = []

    def fake_get(path, params):
        calls.append(params["select"])
        if "grants" in params["select"]:
            raise oa.InvalidSelectError("x", {"id", "title"})
        return {"results": [{"id": "W1"}], "meta": {"next_cursor": None}}

    monkeypatch.setattr(client, "get", fake_get)
    monkeypatch.setattr(oa, "WORK_FIELDS", ["id", "title", "grants"])
    works = list(client.iter_works("x:y", fields=["id", "title", "grants"]))
    assert works == [{"id": "W1"}]
    assert calls[-1] == "id,title"


def test_match_awards_only_kwf():
    import pandas as pd
    from impact.fetch import match_awards_to_projects
    works = pd.DataFrame({"openalex_id": ["W1", "W2", "W3"], "funder_awards": [
        ["F1|KWF 10895"], ["F2|10895"], ["F1|KUN 2015-7970", "F1|11788"]]})
    out = match_awards_to_projects(works, ["10895", "11788"], ["F1"])
    assert sorted(zip(out.openalex_id, out.project_number)) == [("W1", "10895"), ("W3", "11788")]


def test_grant_number_hits_need_dutch_author_and_timing():
    import pandas as pd
    from impact.pipeline import assemble_works
    projects = pd.DataFrame({"project_id": [1], "project_number": ["10004"], "start_year": [2018]})
    base = {"has_iknl_author": False, "funder_awards": [[]] * 3, "year": [2019, 2019, 2015]}
    hits = pd.DataFrame({"openalex_id": ["W1", "W2", "W3"], "kwf_project_number": ["10004"] * 3,
                         "countries": [["NL", "BE"], ["EG"], ["NL"]], **base})
    empty = pd.DataFrame(columns=["openalex_id", "funder_awards", "has_iknl_author", "year"])
    works, links = assemble_works(empty, empty, hits, projects, ["F1"])
    assert works.openalex_id.tolist() == ["W1"]          # W2: no Dutch author, W3: before project start
    assert links.project_number.tolist() == ["10004"]
