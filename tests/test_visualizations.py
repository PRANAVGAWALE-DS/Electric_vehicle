"""
Unit tests for src/visualizations.py.

Run with: pytest tests/test_visualizations.py -v
(requires `pip install -e .` from the project root so `src` is importable)
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src import visualizations as viz
from src.features import (
    CAFV_COL,
    CITY_COL,
    COUNTY_COL,
    EV_TYPE_COL,
    MAKE_COL,
    MODEL_COL,
    RANGE_COL,
    STATE_COL,
    YEAR_COL,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_counties_geojson(monkeypatch):
    """
    Prevent plot_county_choropleth from hitting the network or requiring a
    local data/us_counties.json file during tests.

    Returns an empty FeatureCollection: Plotly accepts it and renders an
    empty choropleth without error, which is sufficient for the chart-type
    assertion tests below.
    """
    monkeypatch.setattr(
        viz,
        "_load_counties_geojson",
        lambda: {"type": "FeatureCollection", "features": []},
    )


@pytest.fixture
def ev_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            YEAR_COL: [2018, 2019, 2020, 2021, 2022, 2023],
            MAKE_COL: ["TESLA", "TESLA", "NISSAN", "FORD", "FORD", "CHEVROLET"],
            MODEL_COL: ["3", "Y", "LEAF", "MACH-E", "F-150", "BOLT"],
            EV_TYPE_COL: [
                "Battery Electric Vehicle (BEV)",
                "Battery Electric Vehicle (BEV)",
                "Battery Electric Vehicle (BEV)",
                "Plug-in Hybrid Electric Vehicle (PHEV)",
                "Battery Electric Vehicle (BEV)",
                "Plug-in Hybrid Electric Vehicle (PHEV)",
            ],
            CAFV_COL: [
                "Clean Alternative Fuel Vehicle Eligible",
                "Clean Alternative Fuel Vehicle Eligible",
                "Not eligible due to low battery range",
                "Eligibility unknown as battery range has not been researched",
                "Clean Alternative Fuel Vehicle Eligible",
                "Eligibility unknown as battery range has not been researched",
            ],
            "cafv_label": [
                "eligible",
                "eligible",
                "not_eligible",
                "unknown",
                "eligible",
                "unknown",
            ],
            RANGE_COL: [220, 330, 150, 25, 300, 90],
            STATE_COL: ["WA", "WA", "WA", "WA", "OR", "WA"],
            COUNTY_COL: ["King", "King", "Pierce", "Snohomish", "Multnomah", "Yakima"],
            CITY_COL: ["Seattle", "Seattle", "Tacoma", "Everett", "Portland", "Yakima"],
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Return-type and title smoke tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("plotter", "expected_title"),
    [
        (viz.plot_model_year_distribution, "EV Registrations by Model Year"),
        (viz.plot_top_makes, "Top 10 EV Manufacturers"),
        (viz.plot_ev_type_over_time, "BEV vs PHEV Share Over Time"),
        (viz.plot_market_share_over_time, "YoY Market Share"),
        (viz.plot_range_by_make, "Electric Range Distribution"),
        (viz.plot_range_progression, "Median Electric Range Progression"),
        (viz.plot_cafv_sunburst, "CAFV Eligibility by EV Type"),
        (viz.plot_county_choropleth, "EV Registrations by County"),
        (viz.plot_top_cities_treemap, "Top 30 Cities"),
    ],
)
def test_plot_functions_return_figures(ev_df, plotter, expected_title):
    fig = plotter(ev_df)

    assert isinstance(fig, go.Figure)
    assert expected_title in fig.layout.title.text


# ──────────────────────────────────────────────────────────────────────────────
# Behaviour tests
# ──────────────────────────────────────────────────────────────────────────────


def test_plot_top_makes_honors_n(ev_df):
    fig = viz.plot_top_makes(ev_df, n=2)

    assert fig.layout.title.text == "Top 2 EV Manufacturers"
    assert len(fig.data[0].y) == 2


def test_plot_range_by_make_filters_by_min_count(ev_df):
    # TESLA×2, FORD×2 pass min_count=2; NISSAN×1, CHEVROLET×1 do not.
    fig = viz.plot_range_by_make(ev_df, min_count=2)

    plotted_makes = {make for trace in fig.data for make in trace.x}
    assert plotted_makes == {"FORD", "TESLA"}


def test_plot_shap_bar_sorts_values_ascending():
    shap_values = pd.Series({"range": 0.4, "age": 0.2, "bev": 0.8})

    fig = viz.plot_shap_bar(shap_values)

    assert isinstance(fig, go.Figure)
    assert list(fig.data[0].y) == ["age", "range", "bev"]
    assert fig.layout.title.text == "Feature Importance (mean |SHAP|)"


def test_plot_residuals_adds_scatter_and_zero_line():
    y_true = pd.Series([100, 120, 140])
    y_pred = pd.Series([90, 125, 130])

    fig = viz.plot_residuals(y_true, y_pred, title="Residual Check")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Residual Check"
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [10, -5, 10]


def test_choropleth_uses_local_geojson_not_network(ev_df, monkeypatch):
    """
    Confirm that plot_county_choropleth calls _load_counties_geojson() and
    never falls back to a raw URL string in the geojson= argument.
    The autouse fixture already injects the mock; this test just makes the
    contract explicit by verifying the mock was actually invoked.
    """
    call_log = []

    def recording_loader():
        call_log.append(True)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(viz, "_load_counties_geojson", recording_loader)
    viz.plot_county_choropleth(ev_df)

    assert len(call_log) == 1, "_load_counties_geojson must be called exactly once per render"
