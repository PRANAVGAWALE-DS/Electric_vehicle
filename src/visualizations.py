"""
Plotly-based visualization library for the EV dataset.

All functions return ``plotly.graph_objects.Figure`` objects so they can be
rendered in notebooks, Streamlit, or exported as HTML/PNG.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.features import (
    CITY_COL,
    COUNTY_COL,
    EV_TYPE_COL,
    MAKE_COL,
    RANGE_COL,
    STATE_COL,
    YEAR_COL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared theme
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Vivid
TEMPLATE = "plotly_white"

# ──────────────────────────────────────────────────────────────────────────────
# Bundled GeoJSON (loaded once, avoids network fetch at render time)
# Download once with: make download-geojson  (see Makefile)
# ──────────────────────────────────────────────────────────────────────────────
_GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "us_counties.json"
_GEOJSON_REMOTE = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)


@lru_cache(maxsize=1)
def _load_counties_geojson() -> dict:
    """Return the US counties GeoJSON dict, preferring the local bundle."""
    if _GEOJSON_PATH.exists():
        with _GEOJSON_PATH.open() as fh:
            return json.load(fh)
    import logging
    import urllib.request

    logging.getLogger(__name__).warning(
        "Local GeoJSON not found at %s — fetching from remote. "
        "Run `make download-geojson` to bundle it locally.",
        _GEOJSON_PATH,
    )
    with urllib.request.urlopen(_GEOJSON_REMOTE) as response:  # noqa: S310
        return json.loads(response.read())


# ──────────────────────────────────────────────────────────────────────────────
# 1. EV Model Year Distribution
# ──────────────────────────────────────────────────────────────────────────────


def plot_model_year_distribution(df: pd.DataFrame) -> go.Figure:
    """Animated-style count of registrations per model year."""
    counts = (
        df[YEAR_COL]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "Count"})
        .sort_values(YEAR_COL)
    )
    fig = px.bar(
        counts,
        x=YEAR_COL,
        y="Count",
        title="EV Registrations by Model Year",
        labels={YEAR_COL: "Model Year", "Count": "Registrations"},
        color="Count",
        color_continuous_scale="Blues",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False, bargap=0.1)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Top N EV Makes
# ──────────────────────────────────────────────────────────────────────────────


def plot_top_makes(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top N manufacturers."""
    counts = df[MAKE_COL].value_counts().head(n).reset_index().rename(columns={"count": "Count"})
    fig = px.bar(
        counts,
        x="Count",
        y=MAKE_COL,
        orientation="h",
        title=f"Top {n} EV Manufacturers",
        labels={"Count": "Registrations", MAKE_COL: "Manufacturer"},
        color="Count",
        color_continuous_scale="Tealgrn",
        template=TEMPLATE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. BEV vs PHEV share over time
# ──────────────────────────────────────────────────────────────────────────────


def plot_ev_type_over_time(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing BEV/PHEV share per model year."""
    pivot = df.groupby([YEAR_COL, EV_TYPE_COL]).size().reset_index(name="Count")
    # Filter to years with meaningful data
    pivot = pivot[pivot[YEAR_COL] >= 2010]
    pivot["EV Type"] = pivot[EV_TYPE_COL].map(
        {
            "Battery Electric Vehicle (BEV)": "BEV",
            "Plug-in Hybrid Electric Vehicle (PHEV)": "PHEV",
        }
    )
    fig = px.area(
        pivot,
        x=YEAR_COL,
        y="Count",
        color="EV Type",
        title="BEV vs PHEV Share Over Time",
        labels={YEAR_COL: "Model Year", "Count": "Registrations"},
        color_discrete_map={"BEV": "#1f77b4", "PHEV": "#ff7f0e"},
        template=TEMPLATE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Market share over time (top 5 makes)
# ──────────────────────────────────────────────────────────────────────────────


def plot_market_share_over_time(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Line chart: YoY registration share for the top-N makes."""
    top_makes = df[MAKE_COL].value_counts().head(top_n).index.tolist()
    sub = df[df[MAKE_COL].isin(top_makes) & (df[YEAR_COL] >= 2013)].copy()
    pivot = sub.groupby([YEAR_COL, MAKE_COL]).size().reset_index(name="Count")
    year_totals = pivot.groupby(YEAR_COL)["Count"].transform("sum")
    pivot["Share %"] = 100 * pivot["Count"] / year_totals

    fig = px.line(
        pivot,
        x=YEAR_COL,
        y="Share %",
        color=MAKE_COL,
        markers=True,
        title=f"YoY Market Share — Top {top_n} Makes",
        labels={YEAR_COL: "Model Year", MAKE_COL: "Make"},
        template=TEMPLATE,
    )
    fig.update_layout(legend_title_text="Manufacturer")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Electric Range by Make — box plot
# ──────────────────────────────────────────────────────────────────────────────


def plot_range_by_make(df: pd.DataFrame, min_count: int = 200) -> go.Figure:
    """Box plot of Electric Range distribution per make (non-zero only)."""
    df_r = df[df[RANGE_COL] > 0].copy()
    valid_makes = df_r[MAKE_COL].value_counts().loc[lambda s: s >= min_count].index
    df_r = df_r[df_r[MAKE_COL].isin(valid_makes)]

    median_order = (
        df_r.groupby(MAKE_COL)[RANGE_COL].median().sort_values(ascending=False).index.tolist()
    )

    fig = px.box(
        df_r,
        x=MAKE_COL,
        y=RANGE_COL,
        category_orders={MAKE_COL: median_order},
        title="Electric Range Distribution by Manufacturer (non-zero ranges)",
        labels={MAKE_COL: "Manufacturer", RANGE_COL: "Electric Range (miles)"},
        color=MAKE_COL,
        template=TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Range progression over model years
# ──────────────────────────────────────────────────────────────────────────────


def plot_range_progression(df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Median electric range per year for top-N makes."""
    top_makes = df[df[RANGE_COL] > 0][MAKE_COL].value_counts().head(top_n).index
    sub = df[(df[RANGE_COL] > 0) & df[MAKE_COL].isin(top_makes) & (df[YEAR_COL] >= 2013)]
    agg = sub.groupby([YEAR_COL, MAKE_COL])[RANGE_COL].median().reset_index()

    fig = px.line(
        agg,
        x=YEAR_COL,
        y=RANGE_COL,
        color=MAKE_COL,
        markers=True,
        title=f"Median Electric Range Progression — Top {top_n} Makes",
        labels={
            YEAR_COL: "Model Year",
            RANGE_COL: "Median Range (miles)",
            MAKE_COL: "Make",
        },
        template=TEMPLATE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. CAFV Eligibility — sunburst (EV Type → CAFV label)
# ──────────────────────────────────────────────────────────────────────────────


def plot_cafv_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst: outer ring = CAFV label, inner ring = EV type."""
    df_plot = df.copy()
    df_plot["EV Type"] = df_plot[EV_TYPE_COL].map(
        {
            "Battery Electric Vehicle (BEV)": "BEV",
            "Plug-in Hybrid Electric Vehicle (PHEV)": "PHEV",
        }
    )
    df_plot["CAFV"] = df_plot["cafv_label"].map(
        {
            "eligible": "Eligible",
            "not_eligible": "Not Eligible",
            "unknown": "Unknown",
        }
    )
    agg = df_plot.groupby(["EV Type", "CAFV"]).size().reset_index(name="Count")
    fig = px.sunburst(
        agg,
        path=["EV Type", "CAFV"],
        values="Count",
        title="CAFV Eligibility by EV Type",
        color="CAFV",
        color_discrete_map={
            "Eligible": "#2ca02c",
            "Not Eligible": "#d62728",
            "Unknown": "#aec7e8",
        },
        template=TEMPLATE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. Washington State Choropleth (county-level EV density)
# ──────────────────────────────────────────────────────────────────────────────

# WA county FIPS mapping (required for plotly choropleth with US counties GeoJSON)
WA_COUNTY_FIPS: dict[str, str] = {
    "Adams": "53001",
    "Asotin": "53003",
    "Benton": "53005",
    "Chelan": "53007",
    "Clallam": "53009",
    "Clark": "53011",
    "Columbia": "53013",
    "Cowlitz": "53015",
    "Douglas": "53017",
    "Ferry": "53019",
    "Franklin": "53021",
    "Garfield": "53023",
    "Grant": "53025",
    "Grays Harbor": "53027",
    "Island": "53029",
    "Jefferson": "53031",
    "King": "53033",
    "Kitsap": "53035",
    "Kittitas": "53037",
    "Klickitat": "53039",
    "Lewis": "53041",
    "Lincoln": "53043",
    "Mason": "53045",
    "Okanogan": "53047",
    "Pacific": "53049",
    "Pend Oreille": "53051",
    "Pierce": "53053",
    "San Juan": "53055",
    "Skagit": "53057",
    "Skamania": "53059",
    "Snohomish": "53061",
    "Spokane": "53063",
    "Stevens": "53065",
    "Thurston": "53067",
    "Wahkiakum": "53069",
    "Walla Walla": "53071",
    "Whatcom": "53073",
    "Whitman": "53075",
    "Yakima": "53077",
}


def plot_county_choropleth(df: pd.DataFrame) -> go.Figure:
    """
    Washington State choropleth of EV registrations per county.
    Uses the Plotly built-in US counties GeoJSON (FIPS codes).
    """
    wa = df[df[STATE_COL] == "WA"].copy()
    county_counts = (
        wa[COUNTY_COL].value_counts().reset_index().rename(columns={"count": "EV Count"})
    )
    county_counts["FIPS"] = county_counts[COUNTY_COL].map(WA_COUNTY_FIPS)
    county_counts = county_counts.dropna(subset=["FIPS"])

    fig = px.choropleth(
        county_counts,
        geojson=_load_counties_geojson(),
        locations="FIPS",
        color="EV Count",
        color_continuous_scale="Blues",
        scope="usa",
        title="EV Registrations by County — Washington State",
        labels={"EV Count": "Registrations"},
        hover_name=COUNTY_COL,
        hover_data={"EV Count": True, "FIPS": False},
        template=TEMPLATE,
    )
    fig.update_geos(
        center={"lat": 47.5, "lon": -120.5},
        projection_scale=5.5,
        visible=False,
        showland=True,
        landcolor="LightGrey",
        showlakes=True,
        lakecolor="LightBlue",
    )
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9. Top Cities — interactive treemap
# ──────────────────────────────────────────────────────────────────────────────


def plot_top_cities_treemap(df: pd.DataFrame, n: int = 30) -> go.Figure:
    """Treemap of top N cities by EV registrations."""
    top = df[CITY_COL].value_counts().head(n).reset_index()
    top.columns = [CITY_COL, "Count"]
    fig = px.treemap(
        top,
        path=[CITY_COL],
        values="Count",
        title=f"Top {n} Cities by EV Registrations",
        color="Count",
        color_continuous_scale="Teal",
        template=TEMPLATE,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 10. SHAP summary (from pre-computed values — keeps viz module standalone)
# ──────────────────────────────────────────────────────────────────────────────


def plot_shap_bar(
    shap_values: pd.Series, title: str = "Feature Importance (mean |SHAP|)"
) -> go.Figure:
    """
    Horizontal bar chart of mean absolute SHAP values.

    Parameters
    ----------
    shap_values : pd.Series
        Index = feature name, values = mean(|SHAP|)
    """
    df_shap = shap_values.sort_values(ascending=True).reset_index()
    df_shap.columns = ["Feature", "Mean |SHAP|"]
    fig = px.bar(
        df_shap,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title=title,
        color="Mean |SHAP|",
        color_continuous_scale="Oranges",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 11. Range residuals scatter
# ──────────────────────────────────────────────────────────────────────────────


def plot_residuals(y_true: pd.Series, y_pred, title: str = "Residual Plot") -> go.Figure:
    """Scatter: actual vs predicted with residual colouring."""
    residuals = y_true.values - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(
                color=residuals,
                colorscale="RdBu",
                opacity=0.5,
                size=4,
                colorbar=dict(title="Residual"),
            ),
            name="Residual",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Range (miles)",
        yaxis_title="Residual (actual − predicted)",
        template=TEMPLATE,
    )
    return fig


if __name__ == "__main__":
    # Basic smoke test to verify imports work
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Visualizations module loaded successfully and imports resolved.")
