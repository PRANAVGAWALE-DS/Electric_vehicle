# ⚡ Washington State EV Population Analysis

[![CI](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions/workflows/ci.yml/badge.svg)](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://electricvehicle-oa89dgvf6dw52cd2kcxiqq.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-grade end-to-end data science pipeline on Washington State's electric vehicle population — EDA and statistical testing, XGBoost modelling with Optuna HPO, SHAP explainability, adoption forecasting, and a live interactive Streamlit dashboard.

> **Live demo →** [electricvehicle-oa89dgvf6dw52cd2kcxiqq.streamlit.app](https://electricvehicle-oa89dgvf6dw52cd2kcxiqq.streamlit.app/)

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Notebooks](#notebooks)
5. [ML Models](#ml-models)
6. [Analysis Highlights](#analysis-highlights)
7. [Dashboard](#dashboard)
8. [Forecasting](#forecasting)
9. [Development & Testing](#development--testing)
10. [Deployment](#deployment)

---

## Dataset

| Property | Value |
|---|---|
| Source | [WA State DOL Electric Vehicle Population Data](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2) |
| Records | ~177,866 (snapshot varies — run `01_eda.ipynb` cell 1 to confirm) |
| Geographic scope | ~99.8% Washington State |
| Key columns | Make, Model, Model Year, EV Type, CAFV Eligibility, Electric Range, County, City |

### Known Data Issues

| Issue | Detail | Handling |
|---|---|---|
| `Electric Range = 0` | ~41% of records — EPA range not entered by DOL, not true zero-range vehicles | Flagged; excluded from range analyses |
| `Base MSRP = 0` | ~95% zeros — column unpopulated for most registrations | Dropped entirely |
| `County / City` missing | ~3 records | Filled with `"Unknown"` |
| State distribution | WA = 130,138 · CA = 83 · VA = 35 — not a national dataset | Noted in all geographic claims |

---

## Project Structure

```
ev_analysis/
├── data/
│   ├── raw/                        ← place the CSV here (git-ignored)
│   ├── processed/                  ← outputs from preprocessing
│   └── models/                     ← saved model artefacts (.joblib)
│
├── notebooks/
│   ├── 01_eda.ipynb                ← EDA, statistical tests, choropleths
│   ├── 02_modeling.ipynb           ← CAFV classifier + range regressor + SHAP
│   └── 03_forecasting.ipynb        ← ARIMA + LightGBM adoption forecasting
│
├── src/
│   ├── __init__.py
│   ├── features.py                 ← preprocessing pipeline & feature engineering
│   ├── models.py                   ← XGBoost training & evaluation
│   └── visualizations.py           ← Plotly chart library
│
├── app/
│   └── streamlit_app.py            ← interactive dashboard
│
├── tests/
│   └── test_preprocessing.py       ← pytest unit tests (no dataset required)
│
├── .github/
│   └── workflows/ci.yml            ← lint + test + notebook validation
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quick Start

```bash
# 1 — Clone and install
git clone https://github.com/your-username/Electric_vehicle.git
cd Electric_vehicle
pip install -r requirements.txt

# 2 — Place the dataset
# Download from: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2
# Save as: data/raw/Electric_Vehicle_Population_Data.csv

# 3 — Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb

# 4 — Or launch the dashboard directly
streamlit run app/streamlit_app.py
```

**Makefile targets:**

```bash
make install       # install all dependencies
make test          # run unit tests
make coverage      # run tests with coverage report
make lint          # lint with ruff
make format        # auto-format with ruff
make run-app       # launch Streamlit dashboard
```

---

## Notebooks

### `01_eda.ipynb` — Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data Quality | Missing-value audit, zero-range and zero-MSRP flagging, dtype inspection |
| Cleaning | Zero-range exclusion, county/city imputation, column pruning |
| Feature Engineering | Vehicle age, make market share, Tesla flag, BEV/PHEV encoding |
| Distributions | Model year histogram, EV type split, range distribution by type |
| Statistical Tests | Chi-square (type × county) · Mann-Whitney U (Tesla vs non-Tesla range) · ANOVA (range × model year) · Chi-square (type × CAFV eligibility) |
| Geospatial | Washington State choropleth by county, top cities treemap |
| Market Analysis | Market HHI, make × year heatmap, CAFV eligibility sunburst |

### `02_modeling.ipynb` — Machine Learning

| Section | Detail |
|---|---|
| Classification | Predict CAFV eligibility — XGBoost multiclass + Optuna HPO |
| Regression | Predict electric range — XGBoost + Optuna HPO (zero-range excluded) |
| Evaluation | Confusion matrix · classification report · actual-vs-predicted scatter |
| Explainability | SHAP `TreeExplainer` — beeswarm + bar for both models |

### `03_forecasting.ipynb` — Adoption Forecasting

| Section | Detail |
|---|---|
| ARIMA(2,1,1) | Statistical baseline trained on 2010–2021 annual registration counts |
| LightGBM | Lag-1/2/3 + rolling mean features, walk-forward validation |
| Per-make trends | Registration trajectory for top 5 manufacturers |
| Penetration rate | BEV share trend 2010–2023 |

---

## ML Models

### 1 · CAFV Eligibility Classifier

41% of vehicles (53,446 records) have CAFV eligibility listed as `"unknown"`. The classifier predicts eligibility from vehicle attributes alone.

| Property | Detail |
|---|---|
| Algorithm | XGBoost multiclass + Optuna HPO (40 trials, 5-fold stratified CV) |
| Target | 3 classes: `eligible` / `not_eligible` / `unknown` |
| Split | 70% train / 15% val (early stopping, `rounds=20`) / 15% test |
| Features | Electric Range, Model Year, EV Type, vehicle age, make market share, Tesla flag |
| Optimised metric | ROC-AUC (OvR) |
| Explainability | SHAP `TreeExplainer` — beeswarm + bar summary |

### 2 · Electric Range Regressor

Predicts electric range in miles. Zero-range records excluded — those represent missing DOL entries, not true zero-range vehicles.

| Property | Detail |
|---|---|
| Algorithm | XGBoost regression + Optuna HPO (40 trials, 5-fold KFold CV) |
| Target | Electric Range (miles) |
| Split | 70% train / 15% val (early stopping, `rounds=20`) / 15% test |
| Baseline | Group-median by BEV / PHEV type |
| Reported metrics | MAE · RMSE · R² vs baseline |
| Explainability | SHAP `TreeExplainer` — feature importance dot plot |

---

## Analysis Highlights

### Statistical Tests

| Test | Question | Result |
|---|---|---|
| Chi-square | Is BEV/PHEV type distribution uniform across counties? | Significant — p < 0.05 |
| Mann-Whitney U | Do Tesla BEVs have higher range than non-Tesla BEVs? | Significant — p < 0.05 |
| One-way ANOVA | Has electric range improved across model years 2017–2023? | Significant — p < 0.05 |
| Chi-square | Is there an association between EV type and CAFV eligibility? | Significant — p < 0.05 |

### Key Findings

| Metric | Value |
|---|---|
| Most registered model year | 2022 — 28,013 vehicles |
| Tesla market share | ~45.7% |
| Market HHI | > 2,500 (highly concentrated) |
| BEV vs PHEV split | 76.8% BEV · 23.2% PHEV |
| Max recorded range | 337 miles (Tesla Model S, 2020) |
| Median range (non-zero BEV, 2020+) | ~220 miles |
| Top county | King County — 68,477 registrations (52.5% of WA total) |

---

## Dashboard

Six tabs, all responding to sidebar filters (year range, EV type, manufacturer, county):

| Tab | Content |
|---|---|
| **Overview** | 5 KPI tiles — total registrations, unique makes/models, avg range, % BEV |
| **Trends** | Model year distribution, BEV/PHEV over time, market share trajectory, CAFV sunburst |
| **Map** | Washington State choropleth by county, top cities treemap, county data table |
| **Makes & Models** | Top manufacturers, top models, market HHI, make × year heatmap |
| **Range Analysis** | Box plots by make, range progression over years, BEV vs PHEV histogram |
| **Statistical Tests** | Live test results on filtered data, contingency table explorer |

---

## Forecasting

`03_forecasting.ipynb` implements two complementary approaches:

| Model | Detail |
|---|---|
| **ARIMA(2,1,1)** | Statistical baseline trained on 2010–2021 annual registration counts |
| **LightGBM** | Lag-1/2/3 + rolling mean features, walk-forward validation |

Additional analyses: per-make trajectory for top 5 manufacturers, BEV penetration rate trend 2010–2023.

---

## Development & Testing

```bash
# Run tests — no dataset required, uses synthetic fixtures
make test

# With coverage report
make coverage

# Lint
make lint
```

Tests cover all preprocessing steps using synthetic DataFrames that match the dataset schema exactly. The full suite runs in CI on every push via GitHub Actions.

---

## Deployment

### Streamlit Cloud

1. Push to GitHub — ensure `data/raw/` is in `.gitignore`
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path**: `app/streamlit_app.py`
4. Upload the CSV via the sidebar file uploader or configure as a Streamlit secret
5. The app rebuilds automatically on every push to `main`

---

## License

MIT — see [LICENSE](LICENSE).
