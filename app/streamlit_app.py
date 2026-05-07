"""
Washington State EV Population Dashboard
────────────────────────────────────────
Run locally:   streamlit run app/streamlit_app.py
Deploy:        streamlit cloud (connect GitHub repo, set main file = app/streamlit_app.py)

Requirements: The CSV must be at data/raw/Electric_Vehicle_Population_Data.csv
              (or upload it via the sidebar file uploader).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import visualizations as viz  # noqa: E402
from src.features import (
    COUNTY_COL,
    EV_TYPE_COL,
    MAKE_COL,
    MODEL_COL,
    RANGE_COL,
    STATE_COL,
    YEAR_COL,
    full_preprocessing_pipeline,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WA EV Population Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading — cached
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PATH = (
    Path(__file__).parent.parent / "data" / "raw" / "Electric_Vehicle_Population_Data.csv"
)


@st.cache_data(show_spinner="Loading & preprocessing dataset…")
def load_data(path: str) -> pd.DataFrame:
    return full_preprocessing_pipeline(path)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — data source + filters
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ EV Dashboard")
    st.caption("Washington State DOL Electric Vehicle Population Data")
    st.divider()

    # Data source
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        file_bytes = uploaded.read()
        content_hash = hashlib.md5(file_bytes).hexdigest()
        tmp_path = Path(tempfile.gettempdir()) / f"ev_upload_{content_hash}.csv"
        if not tmp_path.exists():
            tmp_path.write_bytes(file_bytes)
        data_source = str(tmp_path)
    elif DEFAULT_PATH.exists():
        data_source = str(DEFAULT_PATH)
    else:
        st.error("No data found. Upload the CSV using the file uploader above.")
        st.stop()

    df_raw = load_data(data_source)

    st.divider()
    st.subheader("🔍 Filters")

    # Year range
    min_yr, max_yr = int(df_raw[YEAR_COL].min()), int(df_raw[YEAR_COL].max())
    year_range = st.slider("Model Year", min_yr, max_yr, (2013, max_yr))

    # EV Type
    ev_types = ["All"] + df_raw[EV_TYPE_COL].unique().tolist()
    ev_type_sel = st.selectbox("EV Type", ev_types)

    # Make
    all_makes = sorted(df_raw[MAKE_COL].unique())
    make_sel = st.multiselect("Manufacturer (leave blank = all)", all_makes)

    # County (WA only)
    wa_counties = sorted(df_raw[df_raw[STATE_COL] == "WA"][COUNTY_COL].unique())
    county_sel = st.multiselect("County (WA, leave blank = all)", wa_counties)

    st.divider()
    st.caption("Built with Plotly · Streamlit · XGBoost · SHAP")


# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def apply_filters(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    ev_type: str,
    makes: list[str],
    counties: list[str],
) -> pd.DataFrame:
    mask = (df[YEAR_COL] >= year_range[0]) & (df[YEAR_COL] <= year_range[1])
    if ev_type != "All":
        mask &= df[EV_TYPE_COL] == ev_type
    if makes:
        mask &= df[MAKE_COL].isin(makes)
    if counties:
        mask &= df[COUNTY_COL].isin(counties)
    return df[mask].copy()


df = apply_filters(df_raw, year_range, ev_type_sel, make_sel, county_sel)

# ──────────────────────────────────────────────────────────────────────────────
# KPI row
# ──────────────────────────────────────────────────────────────────────────────
st.title("⚡ Washington State EV Population Dashboard")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Registrations", f"{len(df):,}")
k2.metric("Unique Makes", f"{df[MAKE_COL].nunique()}")
k3.metric("Unique Models", f"{df[MODEL_COL].nunique()}")
k4.metric("Avg Range (non-zero)", f"{df[df[RANGE_COL] > 0][RANGE_COL].mean():.0f} mi")
k5.metric("% BEV", f"{100 * df['is_bev'].mean():.1f}%")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Trends",
        "🗺️ Map",
        "🏭 Makes & Models",
        "🔋 Range Analysis",
        "📊 Statistical Tests",
    ]
)


# ── Tab 1: Trends ─────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_model_year_distribution(df), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_ev_type_over_time(df), use_container_width=True)

    st.plotly_chart(viz.plot_market_share_over_time(df), use_container_width=True)

    # CAFV eligibility
    st.subheader("CAFV Eligibility Distribution")
    col_c, col_d = st.columns([1, 2])
    with col_c:
        cafv_counts = (
            df["cafv_label"].value_counts().rename_axis("Status").reset_index(name="Count")
        )
        st.dataframe(cafv_counts, use_container_width=True)
    with col_d:
        st.plotly_chart(viz.plot_cafv_sunburst(df), use_container_width=True)


# ── Tab 2: Map ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("EV Registrations by County — Washington State")
    st.info(
        "Choropleth uses Plotly's built-in US counties GeoJSON. "
        "Filters above apply — county selection overrides the map focus."
    )
    st.plotly_chart(viz.plot_county_choropleth(df), use_container_width=True)

    st.subheader("Top Cities")
    st.plotly_chart(viz.plot_top_cities_treemap(df), use_container_width=True)

    # County table
    wa_df = df[df[STATE_COL] == "WA"]
    county_tbl = (
        wa_df.groupby(COUNTY_COL)
        .agg(
            registrations=(RANGE_COL, "count"),
            pct_bev=("is_bev", lambda x: f"{x.mean() * 100:.1f}%"),
            median_range=(
                RANGE_COL,
                lambda x: f"{x[x > 0].median():.0f} mi" if (x > 0).any() else "—",
            ),
        )
        .reset_index()
        .sort_values("registrations", ascending=False)
    )
    st.dataframe(county_tbl, use_container_width=True)


# ── Tab 3: Makes & Models ─────────────────────────────────────────────────────
with tab3:
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(viz.plot_top_makes(df), use_container_width=True)
    with col_f:
        # Top models
        top_models = df[MODEL_COL].value_counts().head(15).reset_index()
        top_models.columns = [MODEL_COL, "Count"]
        fig_models = px.bar(
            top_models,
            x="Count",
            y=MODEL_COL,
            orientation="h",
            title="Top 15 EV Models",
            color="Count",
            color_continuous_scale="Teal",
            template="plotly_white",
        )
        fig_models.update_layout(
            coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_models, use_container_width=True)

    # Market share
    st.subheader("Market Concentration")
    shares = df[MAKE_COL].value_counts(normalize=True)
    hhi = (shares**2).sum() * 10_000
    col_g, col_h, col_i = st.columns(3)
    col_g.metric(
        "HHI",
        f"{hhi:.0f}",
        help="<1500 competitive, 1500–2500 moderate, >2500 concentrated",
    )
    col_h.metric("Tesla Share", f"{100 * shares.get('TESLA', 0):.1f}%")
    col_i.metric("Unique Manufacturers", df[MAKE_COL].nunique())

    # Make × Year heatmap
    st.subheader("Make × Model Year Heatmap")
    top_makes_heat = df[MAKE_COL].value_counts().head(10).index.tolist()
    heat_data = (
        df[df[MAKE_COL].isin(top_makes_heat) & (df[YEAR_COL] >= 2015)]
        .groupby([MAKE_COL, YEAR_COL])
        .size()
        .reset_index(name="Count")
    )
    fig_heat = px.density_heatmap(
        heat_data,
        x=YEAR_COL,
        y=MAKE_COL,
        z="Count",
        title="Registration Density — Top 10 Makes × Model Year",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 4: Range Analysis ─────────────────────────────────────────────────────
with tab4:
    st.info(
        f"**Zero-range records**: {df['range_is_zero'].sum():,} records "
        f"({100 * df['range_is_zero'].mean():.1f}% of filtered set). "
        "These represent vehicles whose EPA range has not yet been entered into the DOL database. "
        "All range charts below exclude them."
    )
    col_j, col_k = st.columns(2)
    with col_j:
        st.plotly_chart(viz.plot_range_by_make(df), use_container_width=True)
    with col_k:
        st.plotly_chart(viz.plot_range_progression(df), use_container_width=True)

    # Range histogram
    df_r = df[df[RANGE_COL] > 0].copy()
    df_r["EV Type"] = df_r["is_bev"].map({1: "BEV", 0: "PHEV"})
    fig_hist = px.histogram(
        df_r,
        x=RANGE_COL,
        color="EV Type",
        nbins=50,
        title="Electric Range Distribution — BEV vs PHEV",
        labels={RANGE_COL: "Electric Range (miles)"},
        color_discrete_map={"BEV": "#1f77b4", "PHEV": "#ff7f0e"},
        barmode="overlay",
        opacity=0.7,
        template="plotly_white",
    )
    fig_hist.update_layout(legend_title_text="EV Type")
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 5: Statistical Tests ──────────────────────────────────────────────────
with tab5:
    st.subheader("Statistical Test Results")
    st.caption("Run on the currently filtered dataset.")

    from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu

    results_out = []

    # Test 1: Chi-square EV type vs county
    try:
        top_c = df[COUNTY_COL].value_counts().head(10).index
        sub_c = df[df[COUNTY_COL].isin(top_c)]
        ct = pd.crosstab(sub_c[COUNTY_COL], sub_c["is_bev"])
        chi2, p, dof, _ = chi2_contingency(ct)
        results_out.append(
            {
                "Test": "Chi-square: BEV/PHEV vs County",
                "Statistic": f"χ²={chi2:.2f}, df={dof}",
                "p-value": f"{p:.2e}",
                "Significant (p<0.05)": "✅ Yes" if p < 0.05 else "❌ No",
                "Interpretation": (
                    "EV type distribution varies significantly by county"
                    if p < 0.05
                    else "No significant difference in EV type across counties"
                ),
            }
        )
    except Exception as e:
        results_out.append({"Test": "Chi-square: BEV/PHEV vs County", "Error": str(e)})

    # Test 2: Mann-Whitney — Tesla vs non-Tesla range
    try:
        df_nz = df[df[RANGE_COL] > 0]
        t_r = df_nz[df_nz[MAKE_COL] == "TESLA"][RANGE_COL]
        nt_r = df_nz[df_nz[MAKE_COL] != "TESLA"][RANGE_COL]
        if len(t_r) > 10 and len(nt_r) > 10:
            u, p_mw = mannwhitneyu(t_r, nt_r, alternative="greater")
            results_out.append(
                {
                    "Test": "Mann-Whitney U: Tesla vs Non-Tesla Range",
                    "Statistic": f"U={u:.0f}",
                    "p-value": f"{p_mw:.2e}",
                    "Significant (p<0.05)": "✅ Yes" if p_mw < 0.05 else "❌ No",
                    "Interpretation": f"Tesla median={t_r.median():.0f} mi vs non-Tesla median={nt_r.median():.0f} mi",
                }
            )
    except Exception as e:
        results_out.append({"Test": "Mann-Whitney U: Tesla vs Non-Tesla Range", "Error": str(e)})

    # Test 3: ANOVA — range across years
    try:
        years_test = [
            yr
            for yr in range(2018, 2024)
            if len(df[(df[YEAR_COL] == yr) & (df[RANGE_COL] > 0)]) > 30
        ]
        groups_test = [
            df[(df[YEAR_COL] == yr) & (df[RANGE_COL] > 0)][RANGE_COL].values for yr in years_test
        ]
        if len(groups_test) >= 2:
            f_stat, p_an = f_oneway(*groups_test)
            results_out.append(
                {
                    "Test": f"One-way ANOVA: Range across model years ({years_test[0]}–{years_test[-1]})",
                    "Statistic": f"F={f_stat:.2f}",
                    "p-value": f"{p_an:.2e}",
                    "Significant (p<0.05)": "✅ Yes" if p_an < 0.05 else "❌ No",
                    "Interpretation": (
                        "Significant variation in range across model years"
                        if p_an < 0.05
                        else "No significant range variation across model years"
                    ),
                }
            )
    except Exception as e:
        results_out.append({"Test": "ANOVA: Range across years", "Error": str(e)})

    st.dataframe(pd.DataFrame(results_out), use_container_width=True)

    # Quick contingency table viewer
    st.subheader("Contingency Table Explorer")
    col_x = st.selectbox("Row variable", [COUNTY_COL, MAKE_COL, EV_TYPE_COL, "cafv_label"])
    col_y = st.selectbox("Column variable", ["is_bev", "cafv_label", EV_TYPE_COL], index=0)
    if col_x != col_y:
        top_vals = df[col_x].value_counts().head(8).index
        sub_view = df[df[col_x].isin(top_vals)]
        ct_view = pd.crosstab(sub_view[col_x], sub_view[col_y])
        st.dataframe(ct_view, use_container_width=True)
