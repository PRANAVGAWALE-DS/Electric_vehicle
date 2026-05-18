# ⚡ Washington State EV Population Analysis

[![CI](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions/workflows/ci.yml/badge.svg)](https://github.com/PRANAVGAWALE-DS/Electric_vehicle/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-40_trials-blueviolet.svg)](https://optuna.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-inference_layer-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-94_passing-brightgreen.svg)](#development--testing)
[![HuggingFace](https://img.shields.io/badge/🤗_Spaces-live_demo-FFD21E.svg)](https://huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-grade ML pipeline on 177,866 Washington State EV registrations — end-to-end from raw DOL data to deployed inference API and interactive dashboard. XGBoost models tuned with 40-trial Optuna HPO, SHAP explainability, MLflow experiment tracking, FastAPI inference layer, and a 6-tab Streamlit dashboard live on HuggingFace Spaces.

> **Live dashboard →** [huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard](https://huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard)

---

## Model Results

| Model | Metric | Value | Baseline |
|---|---|---|---|
| CAFV Eligibility Classifier | CV F1-macro | **0.9725 ± 0.0006** | — |
| CAFV Eligibility Classifier | Test F1-macro | **0.9725** | — |
| Electric Range Regressor | Test MAE | **5.28 miles** | 35.90 miles |
| Electric Range Regressor | Test RMSE | **11.88 miles** | — |
| Electric Range Regressor | Test R² | **0.9858** | — |
| Electric Range Regressor | MAE improvement | **85.3%** over group-median baseline | — |

Both models use XGBoost + Optuna HPO (40 trials, 5-fold CV, 60% subsample) with nested MLflow run logging.

---

## Stack

```
Data          Pandas · NumPy · SciPy · statsmodels
ML            XGBoost · scikit-learn · Optuna · SHAP
Tracking      MLflow (SQLite backend, nested Optuna runs)
Forecasting   ARIMA(2,1,1) · LightGBM lag-feature model
API           FastAPI · Pydantic · uvicorn
Dashboard     Streamlit · Plotly
Testing       pytest (94 tests) · pytest-cov · synthetic fixtures
CI/CD         GitHub Actions (lint + test + notebook validation)
Deployment    HuggingFace Spaces (Streamlit) · Streamlit Cloud
```

---

## Project Structure

```
ml_projects/
├── src/
│   ├── features.py          ← preprocessing pipeline & feature engineering
│   ├── models.py            ← XGBoost + Optuna HPO training functions
│   └── visualizations.py   ← Plotly chart library
│
├── api/
│   └── main.py              ← FastAPI: /predict/cafv · /predict/range · /health
│
├── app/
│   └── streamlit_app.py     ← 6-tab interactive dashboard (HF Spaces ready)
│
├── notebooks/
│   ├── 01_eda.ipynb         ← EDA, statistical tests, choropleths
│   ├── 02_modeling.ipynb    ← CAFV classifier + range regressor + SHAP
│   └── 03_forecasting.ipynb ← ARIMA + LightGBM adoption forecasting
│
├── tests/
│   ├── test_api.py          ← 28 FastAPI tests (mocked models)
│   ├── test_models.py       ← training pipeline tests (n_trials=0)
│   ├── test_preprocessing.py← 45 preprocessing unit tests
│   └── test_visualizations.py← 11 chart tests
│
├── data/
│   ├── raw/                 ← CSV here (git-ignored)
│   └── models/              ← .joblib artefacts (git-ignored)
│
├── conftest.py              ← sys.path fix for api/ package in pytest
├── .github/workflows/ci.yml ← GitHub Actions CI
├── Makefile
└── requirements.txt
```

---

## Quick Start

```bash
# 1 — Clone and install
git clone https://github.com/PRANAVGAWALE-DS/Electric_vehicle.git
cd Electric_vehicle
pip install -r requirements.txt

# 2 — Place the dataset
# Download: https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2
# Save as:  data/raw/Electric_Vehicle_Population_Data.csv

# 3 — Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb

# 4 — Train models (saves .joblib to data/models/)
jupyter notebook notebooks/02_modeling.ipynb

# 5 — Launch dashboard
streamlit run app/streamlit_app.py

# 6 — Start inference API
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

**Makefile targets:**

```bash
make install    # install all dependencies
make test       # run 94 unit tests
make coverage   # tests + coverage report
make lint       # lint with ruff
make run-app    # launch Streamlit dashboard
```

---

## ML Models

### 1 · CAFV Eligibility Classifier

41% of vehicles (91,950 records) have CAFV eligibility listed as `"unknown"`. The classifier predicts eligibility from vehicle attributes alone, resolving ambiguity at inference time.

| Property | Detail |
|---|---|
| Algorithm | XGBoost multiclass + Optuna HPO (40 trials) |
| Target | 3 classes: `eligible` / `not_eligible` / `unknown` |
| Split | 70% train / 15% val (early stopping `rounds=20`) / 15% test |
| HPO subsample | 60% of train slice per trial (~72K rows) |
| CV | 5-fold stratified, F1-macro objective |
| **CV F1-macro** | **0.9725 ± 0.0006** |
| **Test F1-macro** | **0.9725** |
| Best iteration | 156 / 487 (early stopped) |
| Explainability | SHAP `TreeExplainer` — beeswarm + bar summary |

**SHAP insight:** `is_bev` is the dominant predictor of CAFV eligibility by a large margin, followed by `Model Year` and `make_market_share`. Whether a vehicle is a BEV is more predictive than the manufacturer, model year, or market position combined.

**Applied to unknowns:** Of 91,950 `unknown` records, the classifier predicts 204 (0.2%) as eligible — confirming most unknowns lack sufficient range data for CAFV qualification.

### 2 · Electric Range Regressor

Zero-range records (51.7% of dataset) are excluded — they represent vehicles whose EPA range has not been entered into the DOL database, not true zero-range vehicles.

| Property | Detail |
|---|---|
| Algorithm | XGBoost regression + Optuna HPO (40 trials) |
| Target | Electric Range (miles) |
| Split | 70% train / 15% val (early stopping `rounds=20`) / 15% test |
| HPO subsample | 60% of train slice per trial (~35K rows) |
| CV | 5-fold, neg-MAE objective |
| Baseline | Group-median by BEV / PHEV type (train only) |
| **Test MAE** | **5.28 miles** (baseline: 35.90 miles) |
| **Test RMSE** | **11.88 miles** |
| **Test R²** | **0.9858** |
| **MAE improvement** | **85.3% over baseline** |
| Explainability | SHAP `TreeExplainer` — feature importance dot plot |

**SHAP insight:** `is_bev` dominates range prediction. `is_tesla` is the second strongest predictor — Tesla BEVs drive the high-range tail of the distribution independently of model year.

### MLflow Experiment Tracking

Both models are tracked under the `ev-population-analysis` experiment with nested runs:

- **Parent run** — final model hyperparams, test metrics, `agg_transformer.joblib` artifact
- **40 nested trial runs** — per-trial hyperparameters and CV score (sortable in the MLflow UI)

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

---

## FastAPI Inference Layer

Three endpoints, Pydantic-validated, loads `.joblib` artefacts once at startup:

```
GET  /health            → liveness + model load status
POST /predict/cafv      → CAFV eligibility (3-class) + probabilities + feature vector
POST /predict/range     → electric range in miles + feature vector
```

```bash
uvicorn api.main:app --reload --port 8000

# Example
curl -X POST http://localhost:8000/predict/cafv \
  -H "Content-Type: application/json" \
  -d '{"make": "TESLA", "model_year": 2022, "ev_type": "BEV"}'
```

```json
{
  "prediction": "eligible",
  "predicted_code": 0,
  "probabilities": {"eligible": 0.9821, "not_eligible": 0.0134, "unknown": 0.0045},
  "input_features": {"Model Year": 2022, "is_bev": 1, "vehicle_age": 3, "is_tesla": 1, "make_market_share": 0.4571},
  "model_version": "cafv-xgboost-optuna-hpo-v1"
}
```

---

## Notebooks

### `01_eda.ipynb` — Exploratory Data Analysis

| Section | Detail |
|---|---|
| Data Quality | Missing-value audit, zero-range and zero-MSRP flagging, dtype inspection |
| Cleaning | Zero-range flagging, county/city imputation, column pruning |
| Feature Engineering | Vehicle age, make market share, Tesla flag, BEV/PHEV encoding |
| Distributions | Model year histogram, EV type split, range distribution by type |
| Statistical Tests | Chi-square · Mann-Whitney U · One-way ANOVA · Chi-square (type × CAFV) |
| Geospatial | Washington State choropleth by county, top cities treemap |
| Market Analysis | HHI, make × year heatmap, CAFV eligibility sunburst |

### `02_modeling.ipynb` — Machine Learning

| Section | Detail |
|---|---|
| Setup | MLflow experiment init, `mlflow.xgboost.autolog()`, Optuna verbosity |
| CAFV Classifier | XGBoost + Optuna HPO, confusion matrix, SHAP beeswarm + bar |
| Range Regressor | XGBoost + Optuna HPO, residual plot, SHAP feature importance |
| Unknown prediction | CAFV classifier applied to 91,950 unknown records |

### `03_forecasting.ipynb` — Adoption Forecasting

| Section | Detail |
|---|---|
| ARIMA(2,1,1) | Baseline on 2010–2021 annual counts; test MAE 25,773 vehicles (53.8% MAPE — expected underestimation of exponential growth) |
| LightGBM | Lag-1/2/3 + rolling mean features, walk-forward validation, iterative 2024–2026 forecast |
| Per-make trends | Registration trajectory for top 5 manufacturers |
| BEV penetration | Share trend 2010–2023 |

---

## Analysis Highlights

### Statistical Tests

| Test | Question | Result |
|---|---|---|
| Chi-square | Is BEV/PHEV distribution uniform across counties? | Significant — p < 0.05 |
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
| CAFV unknown resolved eligible | ~204 of 91,950 (0.2%) |

---

## Dashboard

Live at **[huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard](https://huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard)**

Data fetched automatically from the WA State DOL open-data API on first load — no CSV upload required.

Six tabs, all responding to sidebar filters (year range, EV type, manufacturer, county):

| Tab | Content |
|---|---|
| **📈 Trends** | Model year distribution, BEV/PHEV over time, market share trajectory, CAFV sunburst |
| **🗺️ Map** | Washington State choropleth by county, top cities treemap, county data table |
| **🏭 Makes & Models** | Top manufacturers, top models, market HHI, make × year heatmap |
| **🔋 Range Analysis** | Box plots by make, range progression over years, BEV vs PHEV histogram |
| **📊 Statistical Tests** | Live test results on filtered data, contingency table explorer |

---

## Development & Testing

```bash
make test       # 94 tests across 4 modules, ~46s
make coverage   # coverage report (65%+ on src/)
make lint       # ruff
```

| Module | Tests | Coverage |
|---|---|---|
| `test_preprocessing.py` | 45 | `src/features.py` 96% |
| `test_visualizations.py` | 13 | `src/visualizations.py` 88% |
| `test_api.py` | 28 | `api/main.py` (mocked) |
| `test_models.py` | 5 | `src/models.py` 25%* |

*`src/models.py` coverage is intentionally lower — Optuna HPO training functions are integration-tested via notebooks, not unit tests (40-trial runs take ~10 min and are not appropriate for CI).

Tests use synthetic DataFrames matching the dataset schema exactly — no CSV required. The full suite runs in CI on every push via GitHub Actions.

---

## Dataset

| Property | Value |
|---|---|
| Source | [WA State DOL Electric Vehicle Population Data](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2) |
| Records | ~177,866 |
| Geographic scope | ~99.8% Washington State |
| Key columns | Make, Model, Model Year, EV Type, CAFV Eligibility, Electric Range, County, City |

### Known Data Issues

| Issue | Detail | Handling |
|---|---|---|
| `Electric Range = 0` | ~51.7% of records — EPA range not entered by DOL | Flagged; excluded from range model and range charts |
| `Base MSRP = 0` | ~95% zeros — column unpopulated | Dropped entirely |
| `County / City` missing | ~5 records | Filled with `"Unknown"` |

---

## Deployment

### HuggingFace Spaces (live)

**[huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard](https://huggingface.co/spaces/PG-AIML/wa-ev-population-dashboard)**

Data fetched from the WA DOL open-data API at startup, cached 24 hours. No CSV upload needed.

### Local API

```bash
uvicorn api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

### Local Dashboard

```bash
streamlit run app/streamlit_app.py
# Reads from data/raw/ if present, otherwise fetches from DOL API
```

---

## License

MIT — see [LICENSE](LICENSE).