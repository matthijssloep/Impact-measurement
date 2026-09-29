"""KWF × IKNL impact dashboard, organised by the Netherlands Cancer Agenda goals.

Runs as a normal Streamlit app (`streamlit run app/streamlit_app.py`) and in the
browser via stlite on GitHub Pages. Reads the CSV tables written by
`scripts/run_pipeline.py site` into a `data/` folder next to this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cancer Agenda impact", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- data
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIRS = [HERE / "data", HERE.parent / "site" / "data", Path.cwd() / "data"]
DATA = next((d for d in DATA_DIRS if (d / "impact_per_goal.csv").exists()), DATA_DIRS[0])


@st.cache_data
def load(name: str) -> pd.DataFrame:
    path = DATA / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------- text
T = {
    "en": {
        "title": "Impact of KWF funding and IKNL research on the Netherlands Cancer Agenda",
        "lang": "Language",
        "no_data": "No data yet. Run `python scripts/run_pipeline.py` to build the tables.",
        "tabs": ["Overview", "Impact per goal", "KWF × IKNL", "Projects", "Articles", "Method"],
        "kpi_funding": "KWF funding", "kpi_projects": "KWF projects", "kpi_articles": "Articles",
        "kpi_iknl": "IKNL articles", "kpi_iknl_kwf": "IKNL articles with KWF funding",
        "metric": "Measure",
        "m_funding_eur": "Funding (€, fractional)", "m_projects": "Projects (fractional)",
        "m_articles": "Articles (fractional)", "m_citations": "Citations (fractional)",
        "m_mean_fwci": "Mean field-weighted citation impact", "m_share_top10": "Share in top 10% cited",
        "m_citations_per_meur": "Citations per € million", "m_policy_citations": "Policy citations (Overton)",
        "m_articles_iknl": "IKNL articles", "m_articles_iknl_kwf": "IKNL articles with KWF funding",
        "goal": "Goal", "year": "Year",
        "funding_vs_output": "Funding versus citations per goal",
        "trend": "Articles per goal per year",
        "overlap": "Articles per year by source",
        "src_iknl": "IKNL only", "src_kwf": "KWF only", "src_both": "IKNL + KWF",
        "iknl_funding": "KWF funding to projects led by or involving IKNL, per start year",
        "download": "Download CSV", "search": "Search title",
        "filter_goal": "Goal", "all": "All",
        "fractional_note": "Items linked to several goals are split across them, so goal totals add up to the overall total.",
        "method": """
**Sources.** KWF research database (scraped), OpenAlex (articles), Netherlands Cancer Agenda 2.1 (20 goals).

**IKNL articles**: at least one author affiliated with IKNL in OpenAlex.
**KWF-funded articles**: KWF listed as funder in OpenAlex, or a KWF project number found in the full text next to "KWF" / "Dutch Cancer Society".

**Goals** are assigned with transparent English and Dutch keyword rules (`config/goals.yaml`), matched in titles (weight 3) and abstracts/summaries (weight 1). An item can link to several goals. Articles with no keyword match inherit the goals of the KWF project they are linked to.

**Impact** measures: articles, citations, field-weighted citation impact (FWCI, 1.0 = world average), share in the top 10% most cited, open access share, and (later) policy citations from Overton.
""",
    },
    "nl": {
        "title": "Impact van KWF-financiering en IKNL-onderzoek op de Nederlandse Kankeragenda",
        "lang": "Taal",
        "no_data": "Nog geen data. Draai `python scripts/run_pipeline.py` om de tabellen te bouwen.",
        "tabs": ["Overzicht", "Impact per doel", "KWF × IKNL", "Projecten", "Artikelen", "Methode"],
        "kpi_funding": "KWF-financiering", "kpi_projects": "KWF-projecten", "kpi_articles": "Artikelen",
        "kpi_iknl": "IKNL-artikelen", "kpi_iknl_kwf": "IKNL-artikelen met KWF-geld",
        "metric": "Maat",
        "m_funding_eur": "Financiering (€, fractioneel)", "m_projects": "Projecten (fractioneel)",
        "m_articles": "Artikelen (fractioneel)", "m_citations": "Citaties (fractioneel)",
        "m_mean_fwci": "Gem. veldgewogen citatie-impact", "m_share_top10": "Aandeel in top 10% geciteerd",
        "m_citations_per_meur": "Citaties per miljoen €", "m_policy_citations": "Beleidscitaties (Overton)",
        "m_articles_iknl": "IKNL-artikelen", "m_articles_iknl_kwf": "IKNL-artikelen met KWF-geld",
        "goal": "Doel", "year": "Jaar",
        "funding_vs_output": "Financiering versus citaties per doel",
        "trend": "Artikelen per doel per jaar",
        "overlap": "Artikelen per jaar naar bron",
        "src_iknl": "Alleen IKNL", "src_kwf": "Alleen KWF", "src_both": "IKNL + KWF",
        "iknl_funding": "KWF-financiering aan projecten van of met IKNL, per startjaar",
        "download": "Download CSV", "search": "Zoek in titel",
        "filter_goal": "Doel", "all": "Alle",
        "fractional_note": "Items die bij meerdere doelen horen worden over die doelen verdeeld, zodat de doeltotalen optellen tot het totaal.",
        "method": """
**Bronnen.** KWF-onderzoeksdatabase (gescraped), OpenAlex (artikelen), Nederlandse Kankeragenda 2.1 (20 doelen).

**IKNL-artikelen**: minstens één auteur met een IKNL-affiliatie in OpenAlex.
**KWF-gefinancierde artikelen**: KWF staat als financier in OpenAlex, of een KWF-projectnummer staat in de volledige tekst naast "KWF" / "Dutch Cancer Society".

**Doelen** worden toegekend met transparante Engelse en Nederlandse trefwoordregels (`config/goals.yaml`), in titels (gewicht 3) en samenvattingen (gewicht 1). Een item kan bij meerdere doelen horen. Artikelen zonder trefwoordmatch erven de doelen van het gekoppelde KWF-project.

**Impact**: artikelen, citaties, veldgewogen citatie-impact (FWCI, 1,0 = wereldgemiddelde), aandeel in de top 10% meest geciteerd, open access, en (later) beleidscitaties uit Overton.
""",
    },
}

lang = st.sidebar.radio("Language / Taal", ["nl", "en"], format_func=lambda x: {"nl": "Nederlands", "en": "English"}[x],
                        horizontal=True)
t = T[lang]

# ---------------------------------------------------------------- chart style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), bargap=0.25, barcornerradius=4,
        font=dict(size=13), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def show(fig: go.Figure, title: str | None = None, height: int = 420) -> None:
    if title:
        st.subheader(title)
    st.plotly_chart(style(fig, height), use_container_width=True)


def fmt_eur(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"€{x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"€{x:,.0f}"


# ---------------------------------------------------------------- load
goals = load("goals")
impact = load("impact_per_goal")
trend = load("goal_year_trend")
projects = load("kwf_projects")
works = load("works")
work_goals = load("work_goals")
project_goals = load("project_goals")

st.title(t["title"])

if impact.empty:
    st.info(t["no_data"])
    st.stop()

goal_title = "title_nl" if lang == "nl" else "title_en"
impact["label"] = impact.goal_id + " · " + impact[goal_title].fillna(impact.title_en).fillna("")
labels = dict(zip(impact.goal_id, impact.label))

tabs = st.tabs(t["tabs"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric(t["kpi_funding"], fmt_eur(projects.amount_eur.sum()) if not projects.empty else "–")
    c[1].metric(t["kpi_projects"], f"{len(projects):,}")
    c[2].metric(t["kpi_articles"], f"{len(works):,}")
    if not works.empty:
        c[3].metric(t["kpi_iknl"], f"{int(works.has_iknl_author.sum()):,}")
        c[4].metric(t["kpi_iknl_kwf"], f"{int((works.has_iknl_author & works.is_kwf).sum()):,}")

    d = impact.sort_values("citations", ascending=True)
    fig = px.scatter(d, x="funding_eur", y="citations", hover_name="label", text="goal_id",
                     labels={"funding_eur": t["m_funding_eur"], "citations": t["m_citations"]})
    fig.update_traces(marker=dict(size=11, color=BLUE, line=dict(width=2, color="white")),
                      textposition="top center")
    show(fig, t["funding_vs_output"], 480)
    st.caption(t["fractional_note"])

# ---------------------------------------------------------------- impact per goal
with tabs[1]:
    metric_cols = [m for m in ["funding_eur", "articles", "citations", "mean_fwci", "share_top10",
                               "citations_per_meur", "articles_iknl", "articles_iknl_kwf",
                               "projects", "policy_citations"]
                   if m in impact and impact[m].notna().any()]
    metric = st.selectbox(t["metric"], metric_cols, format_func=lambda m: t[f"m_{m}"])
    d = impact.sort_values(metric, ascending=True)
    fig = px.bar(d, x=metric, y="label", orientation="h", labels={metric: t[f"m_{metric}"], "label": ""})
    fig.update_traces(marker_color=BLUE, marker_line_width=0,
                      hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>")
    show(fig, height=max(420, 28 * len(d)))
    st.caption(t["fractional_note"])

    if not trend.empty:
        h = trend.pivot_table(index="goal_id", columns="year", values="articles", aggfunc="sum").fillna(0)
        h.index = [labels.get(g, g) for g in h.index]
        fig = px.imshow(h, aspect="auto", color_continuous_scale=SEQ,
                        labels=dict(x=t["year"], y="", color=t["kpi_articles"]))
        show(fig, t["trend"], max(420, 26 * len(h)))

    table = impact.drop(columns=["label"]).copy()
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(t["download"], table.to_csv(index=False), "impact_per_goal.csv", "text/csv")

# ---------------------------------------------------------------- KWF × IKNL
with tabs[2]:
    if not works.empty:
        w = works.assign(source=works.apply(
            lambda r: t["src_both"] if r.has_iknl_author and r.is_kwf
            else (t["src_iknl"] if r.has_iknl_author else t["src_kwf"]), axis=1))
        per_year = w.groupby(["year", "source"], as_index=False).size()
        order = [t["src_iknl"], t["src_both"], t["src_kwf"]]
        fig = px.bar(per_year, x="year", y="size", color="source", category_orders={"source": order},
                     color_discrete_map=dict(zip(order, [BLUE, ORANGE, AQUA])),
                     labels={"year": t["year"], "size": t["kpi_articles"]})
        fig.update_traces(marker_line_width=0)
        show(fig, t["overlap"])

    iknl_col = "iknl_involved" if "iknl_involved" in projects else "is_iknl"
    if not projects.empty and projects[iknl_col].any():
        ip = projects[projects[iknl_col]].groupby("start_year", as_index=False).amount_eur.sum()
        fig = px.bar(ip, x="start_year", y="amount_eur", labels={"start_year": t["year"], "amount_eur": "€"})
        fig.update_traces(marker_color=BLUE, marker_line_width=0)
        show(fig, t["iknl_funding"], 360)

    if not works.empty:
        both = works[works.has_iknl_author & works.is_kwf].sort_values("cited_by_count", ascending=False)
        st.subheader(t["src_both"])
        st.dataframe(both[["year", "title", "journal", "doi", "cited_by_count", "fwci", "kwf_evidence"]],
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- explorers
def goal_filter(df: pd.DataFrame, links: pd.DataFrame, id_col: str, key: str) -> pd.DataFrame:
    choice = st.selectbox(t["filter_goal"], [t["all"], *impact.goal_id], key=key,
                          format_func=lambda g: labels.get(g, g))
    query = st.text_input(t["search"], key=f"{key}_q")
    if choice != t["all"] and not links.empty:
        df = df[df[id_col].isin(links.loc[links.goal_id == choice, id_col])]
    if query:
        df = df[df.title.fillna("").str.contains(query, case=False, regex=False)]
    return df


with tabs[3]:
    if not projects.empty:
        d = goal_filter(projects, project_goals, "project_id", "proj")
        st.dataframe(d, use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("url")})
        st.download_button(t["download"], d.to_csv(index=False), "kwf_projects.csv", "text/csv")

with tabs[4]:
    if not works.empty:
        d = goal_filter(works, work_goals, "openalex_id", "art")
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.download_button(t["download"], d.to_csv(index=False), "articles.csv", "text/csv")

with tabs[5]:
    st.markdown(t["method"])
