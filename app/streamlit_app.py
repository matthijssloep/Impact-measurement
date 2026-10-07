"""KWF impact dashboard, organised by the Netherlands Cancer Agenda (NKC) goals.

Runs as a normal Streamlit app (`streamlit run app/streamlit_app.py`) and in the
browser via stlite on GitHub Pages. Reads the CSV tables written by
`scripts/run_pipeline.py site` into a `data/` folder next to this file.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="KWF impact on the Cancer Agenda", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- data
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIRS = [HERE / "data", HERE.parent / "site" / "data", Path.cwd() / "data"]
DATA = next((d for d in DATA_DIRS if (d / "impact_per_goal.csv").exists()), DATA_DIRS[0])


@st.cache_data
def load(name: str) -> pd.DataFrame:
    path = DATA / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------- text
TITLE = "Impact of KWF funding on the Netherlands Cancer Agenda"
TABS = ["Overview", "Impact per goal", "KWF × IKNL", "Projects", "Articles", "Method"]
MEASURES = {
    "funding_eur": "KWF funding (€)",
    "projects": "KWF projects",
    "articles": "KWF-funded articles",
    "citations": "Citations to KWF-funded articles",
    "mean_fwci": "Mean field-weighted citation impact",
    "share_top10": "Share in the top 10% most cited",
    "citations_per_meur": "Citations per € million",
    "policy_citations": "Policy citations (Overton)",
    "share_policy": "Share cited in policy (Overton)",
}
RATIOS = ("mean_fwci", "share_top10", "share_policy")  # shown as bars, never as a log rose
THEMES = {
    "prevention": "Preventing cancer",
    "early_detection": "Early detection",
    "care": "Diagnostics, treatment & care",
    "quality_of_life": "Quality of life",
}
NO_GOAL = "No NKC goal (e.g. fundamental research)"
FRACTIONAL_NOTE = ("Projects and articles linked to several goals are split across them, "
                   "so goal totals add up to the overall total.")
POLAR_NOTE = ("Bar length on a log scale, so small goals stay visible; colour and hover show the real value.")
METHOD = """
**Sources.** KWF research database (all projects starting 2017 or later, scraped), OpenAlex (articles since 2010),
Netherlands Cancer Agenda 2.1 by the Nederlands Kanker Collectief (20 goals).

**KWF-funded articles**: OpenAlex lists KWF Kankerbestrijding as a funder, or a KWF project number appears in the
full text next to "KWF" / "Dutch Cancer Society" (matches published before the project started are dropped).

**Goals** are assigned with transparent English and Dutch keyword rules (`config/goals.yaml`), matched in titles
(weight 3) and abstracts/summaries (weight 1). A project or article can link to several goals; totals split it
across them. Articles without a keyword match inherit the goals of the KWF project they are linked to.

A grant-number match only counts when at least one author is at a Dutch institution.

**Impact measures**: funding, projects, articles, citations, field-weighted citation impact (FWCI, 1.0 = world
average), share in the top 10% most cited, and policy citations from **Overton** (number of policy documents —
guidelines, government and IGO reports, think-tank papers — citing an article; exported October 2026 via
"Find your research via its DOI"). Policy uptake takes years, so shares are shown for articles published at least
four years before the export.

**Known limitations**: the grant-number search covered 248 of 1,029 projects so far; mean FWCI is sensitive to a
few very highly cited papers; citations per € million compares articles since 2010 with funding since 2017;
Overton covers policy documents it indexes, so national documents that are not online are missed.
"""

# ---------------------------------------------------------------- chart style
# Viridis throughout, matching the original notebook charts.
VIRIDIS_BLUE, VIRIDIS_TEAL, VIRIDIS_GREEN = "#3b528b", "#21918c", "#5ec962"
THEME_COLOURS = {"prevention": "#440154", "early_detection": "#3b528b", "care": "#21918c",
                 "quality_of_life": "#5ec962", None: "#b8b8b8"}


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


def polar(d: pd.DataFrame, value: str, value_label: str, fmt: str = ",.0f") -> go.Figure:
    """Rose chart of one measure per goal, in the style of the original notebook:
    radius on a log10(n + 1) scale so small goals stay visible, colour on the
    real value (Viridis), goals clockwise 1 -> 20. Axis ticks and hover show real values."""
    d = d.sort_values("goal_num").assign(r=lambda x: np.log10(x[value].fillna(0).clip(lower=0) + 1))
    fig = px.bar_polar(d, r="r", theta="theta", color=value, color_continuous_scale="Viridis",
                       template="plotly_white", custom_data=["label", value],
                       labels={value: value_label})
    fig.update_traces(hovertemplate=f"%{{customdata[0]}}<br>{value_label}: %{{customdata[1]:{fmt}}}<extra></extra>",
                      marker_line_color="white", marker_line_width=1)
    top = float(d.r.max() or 1)
    ticks = [v for v in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)
             if np.log10(v + 1) <= top * 1.05]
    if len(ticks) > 5:  # every other power of ten, so the money scale stays readable
        ticks = ticks[::2]
    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=True, ticks="", showline=False, gridcolor="lightgray",
                            tickvals=[np.log10(v + 1) for v in ticks],
                            ticktext=[f"{v:,.0f}" for v in ticks], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11), direction="clockwise", rotation=90),
        ),
        coloraxis_colorbar=dict(title=dict(text=value_label, side="top"), outlinewidth=0, ticks="",
                                orientation="h", x=0.5, xanchor="center", y=-0.08, yanchor="top",
                                len=0.6, thickness=14),
        margin=dict(l=60, r=60, t=40, b=90),
    )
    return fig


def show_polar(fig: go.Figure, title: str | None = None, height: int = 700) -> None:
    if title:
        st.subheader(title)
    st.plotly_chart(fig.update_layout(height=height), use_container_width=True)


def hex_rgba(colour: str, alpha: float) -> str:
    c = colour.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


def sankey(flows: pd.DataFrame, columns: list[list[str]], node_colours: dict[str, str],
           value_fmt: str) -> go.Figure:
    """flows: columns source, target, value, colour (link colour).

    `columns` gives the nodes of each column, top to bottom; nodes are placed
    explicitly so the order is stable (Agenda order) and nothing spills out.
    """
    flows = flows[flows.value > 0]
    inflow = flows.groupby("target").value.sum()
    outflow = flows.groupby("source").value.sum()
    size = {n: max(inflow.get(n, 0), outflow.get(n, 0)) for col in columns for n in col}
    nodes, xs, ys = [], [], []
    for i, col in enumerate(columns):
        col = [n for n in col if size.get(n, 0) > 0]
        total = sum(size[n] for n in col)
        gap = 0.02 if len(col) < 8 else 0.004
        # Each node gets at least a minimum slot so small goals' labels do not overlap;
        # the rest of the height is shared in proportion to value.
        min_slot = 0.034 if len(col) >= 8 else 0.0
        raw = {n: size[n] / total for n in col}
        slots = {n: max(raw[n], min_slot) for n in col}
        scale = (1 - gap * (len(col) - 1)) / sum(slots.values())
        y = 0.0
        for n in col:
            h = slots[n] * scale
            nodes.append(n)
            xs.append(min(max(i / (len(columns) - 1), 0.001), 0.999))
            ys.append(min(max(y + h / 2, 0.001), 0.999))
            y += h + gap
    index = {n: i for i, n in enumerate(nodes)}
    flows = flows[flows.source.isin(index) & flows.target.isin(index)]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=nodes, x=xs, y=ys, pad=6, thickness=16, line=dict(color="white", width=0.5),
                  color=[node_colours.get(n, "#8c8c8c") for n in nodes],
                  hovertemplate=f"%{{label}}<br>%{{value:{value_fmt}}}<extra></extra>"),
        link=dict(source=[index[s] for s in flows.source], target=[index[t] for t in flows.target],
                  value=flows.value.tolist(), color=[hex_rgba(c, 0.35) for c in flows.colour],
                  hovertemplate=f"%{{source.label}} → %{{target.label}}<br>%{{value:{value_fmt}}}<extra></extra>"),
    ))
    fig.update_layout(font=dict(size=12), margin=dict(l=8, r=8, t=8, b=8))
    return fig


# In the browser build (stlite on GitHub Pages) st.download_button cannot fetch its file,
# so there the CSV is handed to the browser directly as a Blob link.
IN_BROWSER = sys.platform == "emscripten" or os.environ.get("IMPACT_HTML_DOWNLOADS") == "1"


def download_csv(df: pd.DataFrame, filename: str) -> None:
    data = df.to_csv(index=False)
    if not IN_BROWSER:
        st.download_button("Download CSV", data, filename, "text/csv", key=f"dl_{filename}")
        return
    b64 = base64.b64encode(data.encode("utf-8")).decode("ascii")
    components.html(f"""
<a id="dl" href="#" style="display:inline-block;padding:6px 14px;border:1px solid rgba(49,51,63,.2);
   border-radius:8px;font:14px 'Source Sans Pro',sans-serif;color:#31333f;text-decoration:none;">
   ⬇ Download CSV</a>
<script>
  const bin = atob({json.dumps(b64)});
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const a = document.getElementById("dl");
  a.href = URL.createObjectURL(new Blob([bytes], {{type: "text/csv;charset=utf-8"}}));
  a.download = {json.dumps(filename)};
</script>""", height=48)


def fmt_eur(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"€{x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"€{x:,.0f}"


# ---------------------------------------------------------------- load
goals = load("goals")
impact = load("impact_per_goal_kwf")
if impact.empty:  # older data builds
    impact = load("impact_per_goal")
projects = load("kwf_projects")
works_all = load("works")
work_goals = load("work_goals")
project_goals = load("project_goals")
work_projects = load("work_projects")

st.title(TITLE)

if impact.empty:
    st.info("No data yet. Run `python scripts/run_pipeline.py` to build the tables.")
    st.stop()

works = works_all[works_all.is_kwf] if not works_all.empty else works_all
kwf_goals = work_goals[work_goals.openalex_id.isin(works.openalex_id)] if not work_goals.empty else work_goals

impact["goal_num"] = impact.goal_id.str.extract(r"(\d+)", expand=False).astype(int)
impact["label"] = impact.goal_num.astype(str) + ". " + impact.title_en
impact["theta"] = impact.label
labels = dict(zip(impact.goal_id, impact.label))
goal_theme = dict(zip(goals.goal_id, goals.theme)) if "theme" in goals else {}

tabs = st.tabs(TABS)

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("KWF funding", fmt_eur(projects.amount_eur.sum()) if not projects.empty else "–")
    c[1].metric("KWF projects", f"{len(projects):,}")
    c[2].metric("KWF-funded articles", f"{len(works):,}")
    c[3].metric("Citations to these articles", f"{int(works.cited_by_count.sum()):,}" if not works.empty else "–")

    theme_order = [*THEMES.values(), NO_GOAL]
    goal_order = [labels[g] for g in impact.sort_values("goal_num").goal_id] + [NO_GOAL]
    node_colours = {THEMES[k]: v for k, v in THEME_COLOURS.items() if k}
    node_colours[NO_GOAL] = THEME_COLOURS[None]
    node_colours.update({labels[g]: THEME_COLOURS.get(goal_theme.get(g)) for g in labels})

    # 1. KWF funding: funding stream -> NKC theme -> goal (€, split across goals)
    if not projects.empty:
        p = projects.assign(stream=projects.funding_partner.fillna("KWF (no partner listed)"))
        pg = project_goals.merge(p[["project_id", "stream", "amount_eur"]], on="project_id")
        pg = pg.assign(eur=pg.amount_eur.fillna(0) * pg.weight, theme=pg.goal_id.map(goal_theme))
        unlinked = p[~p.project_id.isin(project_goals.project_id)]
        total = f"KWF funding {fmt_eur(p.amount_eur.sum())}"
        streams = p.groupby("stream").amount_eur.sum().sort_values(ascending=False)
        flows = [pd.DataFrame({"source": total, "target": streams.index, "value": streams.values,
                               "colour": VIRIDIS_BLUE})]
        to_theme = pg.groupby(["stream", "theme"], as_index=False).eur.sum()
        flows.append(pd.DataFrame({"source": to_theme.stream, "target": to_theme.theme.map(THEMES),
                                   "value": to_theme.eur, "colour": to_theme.theme.map(THEME_COLOURS)}))
        nog = unlinked.groupby("stream").amount_eur.sum()
        flows.append(pd.DataFrame({"source": nog.index, "target": NO_GOAL, "value": nog.values,
                                   "colour": THEME_COLOURS[None]}))
        to_goal = pg.groupby(["theme", "goal_id"], as_index=False).eur.sum()
        flows.append(pd.DataFrame({"source": to_goal.theme.map(THEMES), "target": to_goal.goal_id.map(labels),
                                   "value": to_goal.eur, "colour": to_goal.theme.map(THEME_COLOURS)}))
        st.subheader("1 · Where KWF funding goes")
        st.caption("KWF funding by funding stream, through the Cancer Agenda themes to the 20 NKC goals (€).")
        columns = [[total], list(streams.index), theme_order[:-1] + [NO_GOAL], goal_order[:-1]]
        st.plotly_chart(sankey(pd.concat(flows), columns, node_colours, ",.0f").update_layout(height=760),
                        use_container_width=True)

    # 2. KWF-funded articles: route -> NKC theme -> goal (article counts, split across goals)
    if not works.empty:
        linked = set(work_projects.dropna(subset=["project_id"]).openalex_id) if not work_projects.empty else set()
        routes = pd.Series(np.where(works.openalex_id.isin(linked), "From a KWF database project",
                                    "KWF-acknowledged (no project match)"), index=works.openalex_id)
        wg = kwf_goals.assign(route=kwf_goals.openalex_id.map(routes), theme=kwf_goals.goal_id.map(goal_theme))
        total = f"KWF-funded articles ({len(works):,})"
        rc = routes.value_counts()
        flows = [pd.DataFrame({"source": total, "target": rc.index, "value": rc.values, "colour": VIRIDIS_BLUE})]
        to_theme = wg.groupby(["route", "theme"], as_index=False).weight.sum()
        flows.append(pd.DataFrame({"source": to_theme.route, "target": to_theme.theme.map(THEMES),
                                   "value": to_theme.weight, "colour": to_theme.theme.map(THEME_COLOURS)}))
        nog = routes[~routes.index.isin(kwf_goals.openalex_id)].value_counts()
        flows.append(pd.DataFrame({"source": nog.index, "target": NO_GOAL, "value": nog.values,
                                   "colour": THEME_COLOURS[None]}))
        to_goal = wg.groupby(["theme", "goal_id"], as_index=False).weight.sum()
        flows.append(pd.DataFrame({"source": to_goal.theme.map(THEMES), "target": to_goal.goal_id.map(labels),
                                   "value": to_goal.weight, "colour": to_goal.theme.map(THEME_COLOURS)}))
        st.subheader("2 · What KWF funding produced: articles")
        st.caption("KWF-funded articles since 2010, by how they are linked to KWF, through the themes to the NKC goals.")
        columns = [[total], list(rc.index), theme_order, goal_order[:-1]]
        st.plotly_chart(sankey(pd.concat(flows), columns, node_colours, ",.0f").update_layout(height=760),
                        use_container_width=True)

    has_policy = "policy_citations" in works and not works.empty
    policy_cutoff = int(works.year.max()) - 4 if has_policy else None  # policy uptake takes years
    if has_policy:
        # 3. KWF-funded articles -> theme -> cited in policy or not
        pol = pd.Series(np.where(works.policy_citations > 0, "Cited in policy", "Not (yet) cited in policy"),
                        index=works.openalex_id)
        wg = kwf_goals.assign(theme=kwf_goals.goal_id.map(goal_theme).map(THEMES),
                              out=kwf_goals.openalex_id.map(pol))
        total = f"KWF-funded articles ({len(works):,})"
        order = {t: i for i, t in enumerate(theme_order)}
        by_theme = wg.groupby("theme").weight.sum().sort_index(key=lambda i: i.map(order))
        nog = pol[~pol.index.isin(kwf_goals.openalex_id)]
        to_out = (wg.groupby(["theme", "out"], as_index=False).weight.sum()
                  .sort_values(["theme", "out"], ascending=[True, True],
                               key=lambda c: c.map(order) if c.name == "theme" else c))
        nog_out = nog.value_counts()
        policy_colour = {"Not (yet) cited in policy": "#d9d9d9", "Cited in policy": "#fde725"}
        flows = pd.concat([
            pd.DataFrame({"source": to_out.theme, "target": to_out.out, "value": to_out.weight,
                          "colour": ["#e3c800" if o == "Cited in policy" else node_colours.get(t, "#999")
                                     for t, o in zip(to_out.theme, to_out.out)]}),
            pd.DataFrame({"source": NO_GOAL, "target": nog_out.index, "value": nog_out.values,
                          "colour": ["#e3c800" if o == "Cited in policy" else THEME_COLOURS[None]
                                     for o in nog_out.index]}),
        ])
        st.subheader("3 · From research to policy")
        n_pol = int((works.policy_citations > 0).sum())
        st.caption(f"The {len(works):,} KWF-funded articles by Agenda theme, and whether policy documents cite them: "
                   f"{n_pol:,} articles are cited {int(works.policy_citations.sum()):,} times in policy (Overton).")
        st.plotly_chart(sankey(flows, [theme_order, list(policy_colour)],
                               {**node_colours, **policy_colour}, ",.0f")
                        .update_layout(height=560), use_container_width=True)

        # 4. Impact ladder
        kk = works[works.year <= policy_cutoff]
        steps = [("KWF-funded articles", len(kk)),
                 ("Cited by other research", int((kk.cited_by_count > 0).sum())),
                 ("Top 10% most cited in their field", int(kk.top10pct.sum())),
                 ("Cited in policy", int((kk.policy_citations > 0).sum())),
                 ("Cited in 3 or more policy documents", int((kk.policy_citations >= 3).sum()))]
        fig = go.Figure(go.Funnel(y=[a for a, _ in steps], x=[b for _, b in steps], textinfo="value+percent initial",
                                  marker=dict(color=["#3b528b", "#2c728e", "#21918c", "#5ec962", "#fde725"])))
        fig.update_layout(template="plotly_white")
        show(fig, "4 · Impact ladder: from publication to policy", 420)
        st.caption(f"KWF-funded articles 2010–{policy_cutoff}; each step is a subset of the one above.")

    # 5. Impact per NKC goal (rose chart)
    st.subheader("5 · Impact per NKC goal")
    options = [m for m in MEASURES if m in impact and impact[m].notna().any()]
    c1, c2 = st.columns([3, 1])
    metric = c1.selectbox("Measure", options, format_func=MEASURES.get)
    view = c2.radio("View", ["Rose", "Bars"], horizontal=True)
    if view == "Rose" and metric not in RATIOS:
        show_polar(polar(impact, metric, MEASURES[metric]))
        st.caption(POLAR_NOTE)
    else:
        d = impact.sort_values(metric, ascending=True)
        fig = px.bar(d, x=metric, y="label", orientation="h", color=metric, color_continuous_scale="Viridis",
                     template="plotly_white", labels={metric: MEASURES[metric], "label": ""})
        fig.update_traces(marker_line_width=0, hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>")
        show(fig, height=max(420, 28 * len(d)))
        if view == "Rose":
            st.caption("Averages and shares are shown as bars; a log rose chart only suits counts and amounts.")
    st.caption(FRACTIONAL_NOTE)

    if has_policy:
        # 6. Most policy-cited KWF research
        top = works.nlargest(12, "policy_citations").iloc[::-1]
        top = top.assign(short=top.title.str.slice(0, 80) + np.where(top.title.str.len() > 80, "…", ""))
        fig = go.Figure(go.Bar(x=top.policy_citations, y=top.short, orientation="h", text=top.policy_citations,
                               textposition="outside", marker=dict(color=top.policy_citations, colorscale="Viridis"),
                               customdata=top[["journal", "year", "doi"]],
                               hovertemplate="%{y}<br>%{customdata[0]} (%{customdata[1]})<br>doi: %{customdata[2]}"
                                             "<br>%{x} policy documents<extra></extra>"))
        fig.update_layout(template="plotly_white", xaxis_title="Policy documents citing the article")
        show(fig, "6 · KWF-funded research most cited in policy", 520)

# ---------------------------------------------------------------- impact per goal (table)
with tabs[1]:
    if has_policy:
        old = works[works.year <= policy_cutoff]
        y = kwf_goals.merge(old[["openalex_id", "policy_citations", "fwci"]], on="openalex_id")
        y["in_policy"] = (y.policy_citations > 0).astype(float)
        avg = float((old.policy_citations > 0).mean() * 100)
        per_goal = (y.groupby("goal_id")
                    .apply(lambda d: pd.Series({"share": np.average(d.in_policy, weights=d.weight) * 100,
                                                "n": d.weight.sum(), "pc": (d.policy_citations * d.weight).sum(),
                                                "fwci": d.fwci.median()}), include_groups=False)
                    .reset_index())
        per_goal = per_goal[per_goal.n >= 5]
        per_goal["label"] = per_goal.goal_id.map(labels)
        per_goal["theme"] = per_goal.goal_id.map(goal_theme).map(THEMES)
        theme_colours = {THEMES[k]: v for k, v in THEME_COLOURS.items() if k}

        # Policy uptake per goal
        d = per_goal.assign(num=per_goal.goal_id.str.extract(r"(\d+)", expand=False).astype(int)).sort_values(
            "num", ascending=False)
        fig = px.bar(d, x="share", y="label", orientation="h", color="theme", color_discrete_map=theme_colours,
                     custom_data=["n", "pc"], template="plotly_white",
                     labels={"share": "% of KWF-funded articles cited in policy", "label": "", "theme": ""})
        fig.update_traces(hovertemplate="%{y}: %{x:.0f}% (n=%{customdata[0]:.0f}; "
                                        "%{customdata[1]:.0f} policy citations)<extra></extra>")
        fig.add_vline(x=avg, line_dash="dot", line_color="#666", annotation_text=f"KWF average {avg:.0f}%",
                      annotation_position="top")
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, x=0))
        show(fig, "Which goals reach policy?", 700)
        st.caption(f"Share of KWF-funded articles (2010–{policy_cutoff}) cited in at least one policy document; "
                   "goals with fewer than 5 articles left out.")

        # Academic vs policy impact
        fig = px.scatter(per_goal, x="fwci", y="share", size="n", color="theme", color_discrete_map=theme_colours,
                         text=per_goal.goal_id.str.extract(r"(\d+)", expand=False).astype(int).astype(str),
                         hover_name="label", size_max=45, template="plotly_white",
                         labels={"fwci": "Academic impact: median FWCI", "share": "Policy uptake: % cited in policy",
                                 "theme": "", "n": "Articles"})
        fig.update_traces(textposition="middle center", textfont=dict(color="white", size=11),
                          marker=dict(line=dict(width=1, color="white"), sizemin=18))
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
        fig.add_vline(x=1, line_dash="dot", line_color="#999")
        fig.add_hline(y=avg, line_dash="dot", line_color="#999")
        show(fig, "Academic impact versus policy impact per goal", 560)
        st.caption("Bubble = NKC goal (number), size = number of KWF-funded articles. Dotted lines: world-average "
                   "FWCI (1.0) and the KWF average policy uptake.")

        # Uptake takes time
        t = (works[works.year < works.year.max()].groupby("year").policy_citations
             .apply(lambda s: (s > 0).mean() * 100).reset_index(name="share"))
        fig = px.line(t, x="year", y="share", markers=True, template="plotly_white",
                      labels={"share": "% cited in policy", "year": "Publication year"})
        fig.update_traces(line=dict(width=2, color=VIRIDIS_TEAL), marker=dict(size=8))
        show(fig, "Policy uptake takes time", 380)
        st.caption("Share of KWF-funded articles cited in policy, by publication year: recent work has not had "
                   "time to be picked up yet.")

    st.subheader("Impact per NKC goal (KWF-funded work)")
    cols = ["goal_id", "title_en", "theme", "featured", "ambition", "funding_eur", "funding_share", "projects",
            "projects_any", "articles", "articles_any", "citations", "mean_fwci", "share_top10", "share_oa",
            "citations_per_meur", "policy_citations", "share_policy"]
    table = impact[[c for c in cols if c in impact]].copy()
    st.dataframe(table, use_container_width=True, hide_index=True)
    download_csv(table, "impact_per_goal_kwf.csv")
    st.caption(FRACTIONAL_NOTE + " `*_any` columns count every linked item at full weight.")

# ---------------------------------------------------------------- KWF × IKNL
GROUP_COLOURS = {"IKNL only": VIRIDIS_BLUE, "IKNL + KWF": VIRIDIS_TEAL, "KWF only": VIRIDIS_GREEN}

with tabs[2]:
    if works_all.empty:
        st.info("No article data yet.")
    else:
        wa = works_all.assign(group=np.select(
            [works_all.has_iknl_author & works_all.is_kwf, works_all.has_iknl_author],
            ["IKNL + KWF", "IKNL only"], "KWF only"))
        iknl = wa[wa.has_iknl_author]
        c = st.columns(4)
        c[0].metric("IKNL articles", f"{len(iknl):,}")
        c[1].metric("…with KWF funding", f"{int(iknl.is_kwf.sum()):,}")
        c[2].metric("KWF projects with IKNL", f"{int(projects.iknl_involved.sum()) if 'iknl_involved' in projects else 0}")
        c[3].metric("KWF funding to those projects",
                    fmt_eur(projects.loc[projects.iknl_involved, 'amount_eur'].sum()) if 'iknl_involved' in projects else "–")

        # A · share of IKNL research that is KWF-funded
        last_full_year = int(wa.year.max()) - 1
        a = (iknl[iknl.year <= last_full_year].groupby("year")
             .agg(n=("openalex_id", "size"), k=("is_kwf", "sum")).reset_index())
        a["share"] = a.k / a.n * 100
        fig = go.Figure()
        fig.add_bar(x=a.year, y=a.k, name="KWF-funded", marker_color=VIRIDIS_TEAL,
                    hovertemplate="%{x}: %{y} KWF-funded<extra></extra>")
        fig.add_bar(x=a.year, y=a.n - a.k, name="Other funding", marker_color="#c9d3e6",
                    text=[f"{v:.0f}%" for v in a.share], textposition="outside",
                    textfont=dict(color=VIRIDIS_TEAL, size=12), customdata=a.share,
                    hovertemplate="%{x}: %{y} other · %{customdata:.0f}% KWF-funded<extra></extra>")
        fig.update_layout(barmode="stack", yaxis_title="IKNL articles", template="plotly_white")
        show(fig, "A · How much of IKNL's research is KWF-funded?", 440)
        st.caption(f"IKNL articles per year up to {last_full_year}, split by funding; label = share KWF-funded.")

        # B · Sankey: IKNL articles -> funding -> Agenda theme
        ik_goals = work_goals[work_goals.openalex_id.isin(iknl.openalex_id)].merge(
            iknl[["openalex_id", "is_kwf"]], on="openalex_id")
        ik_goals = ik_goals.assign(mid=np.where(ik_goals.is_kwf, "KWF-funded", "Other funding"),
                                   theme=ik_goals.goal_id.map(goal_theme))
        src = f"IKNL articles ({len(iknl):,})"
        mids = pd.Series(np.where(iknl.is_kwf, "KWF-funded", "Other funding"), index=iknl.openalex_id)
        mc = mids.value_counts()
        to_theme = ik_goals.groupby(["mid", "theme"], as_index=False).weight.sum()
        nog = mids[~mids.index.isin(work_goals.openalex_id)].value_counts()
        flows = pd.concat([
            pd.DataFrame({"source": src, "target": mc.index, "value": mc.values,
                          "colour": [VIRIDIS_TEAL if m == "KWF-funded" else "#9aa7c4" for m in mc.index]}),
            pd.DataFrame({"source": to_theme.mid, "target": to_theme.theme.map(THEMES), "value": to_theme.weight,
                          "colour": [VIRIDIS_TEAL if m == "KWF-funded" else "#9aa7c4" for m in to_theme.mid]}),
            pd.DataFrame({"source": nog.index, "target": NO_GOAL, "value": nog.values, "colour": "#b8b8b8"}),
        ])
        colours = {THEMES[k]: v for k, v in THEME_COLOURS.items() if k}
        colours.update({NO_GOAL: "#b8b8b8", src: VIRIDIS_BLUE, "KWF-funded": VIRIDIS_TEAL, "Other funding": "#9aa7c4"})
        st.subheader("B · IKNL research: KWF-funded or not, and on which Agenda themes")
        st.plotly_chart(sankey(flows, [[src], ["Other funding", "KWF-funded"], [*THEMES.values(), NO_GOAL]],
                               colours, ",.0f").update_layout(height=520), use_container_width=True)
        st.caption("IKNL articles since 2010; articles linked to several goals are split across them.")

        # C · portfolio profile per goal
        k = work_goals.merge(wa[["openalex_id", "has_iknl_author", "is_kwf"]], on="openalex_id")
        prof = pd.DataFrame({
            "IKNL": k[k.has_iknl_author].groupby("goal_id").weight.sum() / k[k.has_iknl_author].weight.sum() * 100,
            "KWF": k[k.is_kwf].groupby("goal_id").weight.sum() / k[k.is_kwf].weight.sum() * 100,
        }).fillna(0).reset_index()
        prof["label"] = prof.goal_id.map(labels)
        prof["n"] = prof.goal_id.str.extract(r"(\d+)", expand=False).astype(int)
        prof = prof.sort_values("n", ascending=False)
        fig = go.Figure()
        fig.add_bar(y=prof.label, x=-prof.IKNL, orientation="h", name="IKNL articles", marker_color=VIRIDIS_BLUE,
                    customdata=prof.IKNL, hovertemplate="%{y}: %{customdata:.1f}% of IKNL articles<extra></extra>")
        fig.add_bar(y=prof.label, x=prof.KWF, orientation="h", name="KWF-funded articles", marker_color=VIRIDIS_GREEN,
                    hovertemplate="%{y}: %{x:.1f}% of KWF-funded articles<extra></extra>")
        lim = float(max(prof.IKNL.max(), prof.KWF.max()) * 1.1)
        ticks = [v for v in range(-50, 51, 10) if abs(v) <= lim]
        fig.update_layout(barmode="relative", template="plotly_white",
                          xaxis=dict(range=[-lim, lim], tickvals=ticks, ticktext=[f"{abs(v)}%" for v in ticks],
                                     title="Share of each portfolio"))
        show(fig, "C · Different strengths: IKNL and KWF portfolios per goal", 700)
        st.caption("Share of IKNL articles (left) and of KWF-funded articles (right) per NKC goal.")

        # D · citation impact of joint work
        cutoff = int(wa.year.max()) - 3  # recent articles have not had time to be cited
        d = (wa[wa.year <= cutoff].groupby("group")
             .agg(median_fwci=("fwci", "median"), top10=("top10pct", "mean"), n=("openalex_id", "size"))
             .reindex(list(GROUP_COLOURS)).reset_index())
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure(go.Bar(x=d.group, y=d.median_fwci, marker_color=[GROUP_COLOURS[g] for g in d.group],
                                   text=d.median_fwci.round(2), textposition="outside", customdata=d.n,
                                   hovertemplate="%{x}: %{y:.2f} (n=%{customdata:,})<extra></extra>"))
            fig.add_hline(y=1, line_dash="dot", line_color="#999")
            fig.update_layout(template="plotly_white", yaxis_title="Median FWCI")
            show(fig, "D · Citation impact of joint work", 400)
        with c2:
            fig = go.Figure(go.Bar(x=d.group, y=d.top10 * 100, marker_color=[GROUP_COLOURS[g] for g in d.group],
                                   text=[f"{v:.0f}%" for v in d.top10 * 100], textposition="outside",
                                   hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
            fig.add_hline(y=10, line_dash="dot", line_color="#999")
            fig.update_layout(template="plotly_white", yaxis_title="% in top 10% most cited")
            show(fig, "\u00a0", 400)
        with c3:
            if "policy_citations" in wa:
                pol_cut = int(wa.year.max()) - 4
                pp = (wa[wa.year <= pol_cut].groupby("group").policy_citations
                      .apply(lambda s: (s > 0).mean() * 100).reindex(list(GROUP_COLOURS)))
                fig = go.Figure(go.Bar(x=pp.index, y=pp.values, marker_color=[GROUP_COLOURS[g] for g in pp.index],
                                       text=[f"{v:.0f}%" for v in pp.values], textposition="outside",
                                       hovertemplate="%{x}: %{y:.0f}% cited in policy<extra></extra>"))
                fig.update_layout(template="plotly_white", yaxis_title="% cited in policy (Overton)")
                show(fig, "\u00a0", 400)
        st.caption(f"Articles 2010–{cutoff} (policy: 2010–{int(wa.year.max()) - 4}). Dotted line = world average "
                   "(FWCI 1.0; 10% in the top 10%).")

        # E · partner institutions
        partners = load("iknl_kwf_partners")
        if not partners.empty:
            top = partners.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top.articles, y=top.institution, orientation="h",
                                   marker=dict(color=top.articles, colorscale="Viridis"),
                                   hovertemplate="%{y}: %{x} joint articles<extra></extra>"))
            fig.update_layout(template="plotly_white", xaxis_title="IKNL articles with KWF funding")
            show(fig, "E · Who IKNL works with on KWF-funded research", 560)
            st.caption(f"Partner institutions on the {int(iknl.is_kwf.sum()):,} IKNL articles with KWF funding "
                       "(top 15; IKNL itself excluded).")

        # F · KWF projects with IKNL
        if "iknl_involved" in projects and projects.iknl_involved.any() and "start_date" in projects:
            q = projects[projects.iknl_involved].copy()
            q["start"] = pd.to_datetime(q.start_date)
            q["end"] = q.start + pd.to_timedelta(q.duration_months.fillna(48) * 30.44, unit="D")
            n_art = (work_projects.dropna(subset=["project_id"]).groupby("project_id").size()
                     if not work_projects.empty else pd.Series(dtype=int))
            q["articles"] = q.project_id.astype(str).map(n_art.rename(index=str)).fillna(0).astype(int)
            q["role"] = np.where(q.is_iknl, "IKNL leads", "IKNL in project team")
            q["short"] = q.title.str.slice(0, 60) + np.where(q.title.str.len() > 60, "…", "")
            q = q.sort_values(["role", "start"])
            fig = px.timeline(q, x_start="start", x_end="end", y="short", color="role",
                              color_discrete_map={"IKNL leads": "#440154", "IKNL in project team": VIRIDIS_TEAL},
                              hover_data={"institution": True, "amount_eur": ":,.0f", "articles": True,
                                          "short": False, "start": False, "end": False})
            for r in q.itertuples():
                fig.add_annotation(x=r.end, y=r.short, text=f" {fmt_eur(r.amount_eur)} · {r.articles} art.",
                                   showarrow=False, xanchor="left", font=dict(size=10, color="#555"))
            fig.update_yaxes(autorange="reversed", title="")
            fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=-0.08, title=""))
            show(fig, "F · KWF projects with IKNL: timeline, budget and output", max(420, 26 * len(q) + 120))
            st.caption("KWF projects led by or involving IKNL; label = budget · articles linked to the project.")

        both = wa[wa.group == "IKNL + KWF"].sort_values("cited_by_count", ascending=False)
        st.subheader("IKNL articles with KWF funding")
        st.dataframe(both[["year", "title", "journal", "doi", "cited_by_count", "fwci", "kwf_evidence"]],
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- explorers
def goal_filter(df: pd.DataFrame, links: pd.DataFrame, id_col: str, key: str) -> pd.DataFrame:
    choice = st.selectbox("NKC goal", ["All", *impact.sort_values("goal_num").goal_id], key=key,
                          format_func=lambda g: labels.get(g, g))
    query = st.text_input("Search title", key=f"{key}_q")
    if choice != "All" and not links.empty:
        df = df[df[id_col].isin(links.loc[links.goal_id == choice, id_col])]
    if query:
        df = df[df.title.fillna("").str.contains(query, case=False, regex=False)]
    return df


with tabs[3]:
    if not projects.empty:
        d = goal_filter(projects.drop(columns=["is_iknl", "iknl_involved"], errors="ignore"),
                        project_goals, "project_id", "proj")
        st.dataframe(d, use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("url")})
        download_csv(d, "kwf_projects.csv")

with tabs[4]:
    if not works.empty:
        d = goal_filter(works.drop(columns=["has_iknl_author", "is_kwf"], errors="ignore"),
                        kwf_goals, "openalex_id", "art")
        st.caption(f"{len(d):,} KWF-funded articles")
        st.dataframe(d, use_container_width=True, hide_index=True)
        download_csv(d, "kwf_articles.csv")

with tabs[5]:
    st.markdown(METHOD)
