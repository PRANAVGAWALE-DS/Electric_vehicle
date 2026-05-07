# ⚡ Washington State EV Population Analysis

[![CI](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions/workflows/ci.yml/badge.svg)](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/PRANAVGAWALE-DS/Electric_vehicle)

Production-grade analysis of Washington State's electric vehicle population — from EDA and statistical testing through XGBoost modelling, SHAP explainability, and an interactive Streamlit dashboard.

---

## Dataset

| Property | Value |
|---|---|
| Source | [WA State DOL Electric Vehicle Population Data](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2) |
| Records | ~177,866 (snapshot varies — check `len(df)` after load) |
| Geographic scope | **~99.8% Washington State** (not a national dataset) |
| Key columns | Make, Model, Model Year, Electric Vehicle Type, CAFV Eligibility, Electric Range, County, City |

> ⚠️ **Note:** Record counts vary across DOL snapshots. Two CSV files are present in `data/raw/` (41 MB and 73 MB). `features.py` was last validated on the 177,866-record snapshot. Run `01_eda.ipynb` cell 1 to confirm the current row count and update this table accordingly.

### Known Data Issues

| Issue | Detail |
|---|---|
| `Electric Range = 0` | ~41% of records. Vehicles whose EPA range has **not been entered** in the DOL database (mostly post-2021 models), not actually zero-range vehicles. Flagged and excluded from range analyses. |
| `Base MSRP = 0` | ~95% zeros. Column not populated for most registrations. Dropped. |
| `County/City` missing | ~3 records. Filled with "Unknown". |
| State distribution | WA=130,138, CA=83, VA=35 — claims about "California leads" in older documentation were incorrect for this dataset. |

---

## Project Structure

```
ev_analysis/
├── data/
│   ├── raw/                    ← Place the CSV here (not tracked by git)
│   ├── processed/              ← Outputs from preprocessing
│   └── models/                 ← Saved model artifacts (.joblib)
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Enhanced EDA + statistical tests + choropleth
│   ├── 02_modeling.ipynb       ← CAFV classifier + range regressor + SHAP
│   └── 03_forecasting.ipynb    ← Adoption trend forecasting (ARIMA + LightGBM)
│
├── src/
│   ├── __init__.py
│   ├── features.py             ← Preprocessing pipeline & feature engineering
│   ├── models.py               ← XGBoost model training & evaluation
│   └── visualizations.py       ← Plotly chart library
│
├── app/
│   └── streamlit_app.py        ← Interactive dashboard
│
├── tests/
│   └── test_preprocessing.py   ← Unit tests (pytest, no dataset required)
│
├── .github/
│   └── workflows/ci.yml        ← GitHub Actions: lint + test + notebook validation
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quick Start

```bash
# 1 — Clone and install
git clone https://github.com/YOUR_USERNAME/ev_analysis.git
cd ev_analysis
pip install -r requirements.txt

# 2 — Place the dataset
# Download from: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2
# Save as: data/raw/Electric_Vehicle_Population_Data.csv

# 3 — Run the dashboard
streamlit run app/streamlit_app.py

# 4 — Or open notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

**Using the Makefile:**
```bash
make install       # Install all dependencies
make test          # Run unit tests
make coverage      # Run tests with coverage report
make run-app       # Launch Streamlit dashboard
make lint          # Lint with ruff
make format        # Auto-format with ruff
```

---

## ML Models

### 1 · CAFV Eligibility Classifier

**Problem**: 41% of vehicles (53,446 records) have CAFV eligibility listed as "unknown". Can we predict eligibility from vehicle attributes?

| | |
|---|---|
| Algorithm | XGBoost multiclass |
| Target | 3 classes: eligible / not_eligible / unknown |
| Features | Electric Range, Model Year, EV Type, vehicle age, make market share, Tesla flag |
| Validation | 5-fold stratified CV, final eval on 20% holdout |
| Explainability | SHAP TreeExplainer — feature importance + beeswarm plots |

### 2 · Electric Range Regressor

| | |
|---|---|
| Algorithm | XGBoost regression |
| Target | Electric Range (miles), zero-range records excluded |
| Baseline | Group-median by BEV/PHEV type |
| Validation | 80/20 train-test split, MAE/RMSE/R² reported vs baseline |

---

## Analysis Highlights

### Statistical Tests
- **Chi-square** — BEV/PHEV type distribution is not uniform across counties (p < 0.05)
- **Mann-Whitney U** — Tesla electric ranges are significantly higher than non-Tesla BEVs (p < 0.05)
- **One-way ANOVA** — Significant improvement in electric range across model years 2017–2023 (p < 0.05)
- **Chi-square** — Strong association between EV type and CAFV eligibility (p < 0.05)

### Key Findings
| Finding | Value |
|---|---|
| Most registered model year | 2022 (28,013 vehicles) |
| Tesla market share | ~45.7% |
| Market HHI | >2,500 (highly concentrated — Tesla dominant) |
| BEV vs PHEV split | 76.8% BEV, 23.2% PHEV |
| Max recorded range | 337 miles (Tesla Model S, 2020) |
| Median range (non-zero BEV, 2020+) | ~220 miles |
| Top county | King County (68,477) — 52.5% of all WA registrations |

---

## Dashboard Features

The Streamlit dashboard provides:
- **5 KPI tiles** — total registrations, unique makes/models, avg range, % BEV
- **Trends tab** — model year distribution, BEV/PHEV over time, market share trajectory, CAFV sunburst
- **Map tab** — Washington State choropleth by county, top cities treemap, county data table
- **Makes & Models tab** — top manufacturers, top models, market HHI, make×year heatmap
- **Range Analysis tab** — box plots by make, range progression over years, BEV vs PHEV histogram
- **Statistical Tests tab** — live test results on filtered data, contingency table explorer

All charts and tests respond to sidebar filters (year range, EV type, manufacturer, county).

---

## Forecasting

Notebook `03_forecasting.ipynb` implements:
- **ARIMA(2,1,1)** — statistical baseline forecast trained on 2010–2021
- **LightGBM with lag features** — lag-1/2/3 + rolling mean, walk-forward validation
- **Per-make trajectory** analysis for top 5 manufacturers
- **BEV penetration rate** trend 2010–2023

---

## Development

```bash
# Run tests (no dataset required — uses synthetic fixtures)
make test

# Run with coverage
make coverage

# Lint
make lint
```

Tests cover all preprocessing steps using synthetic DataFrames that match the dataset schema. No real data download required to run the test suite.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (ensure `data/raw/` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path**: `app/streamlit_app.py`
4. Add the CSV as a Streamlit secret or upload via the sidebar file uploader
5. The app will rebuild automatically on every push to `main`

---

## License

MIT — see [LICENSE](LICENSE).
