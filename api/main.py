"""
FastAPI inference service — Washington State EV Population Analysis.

Endpoints
---------
GET  /health           → liveness + model load status
POST /predict/cafv     → CAFV eligibility classification (3-class XGBoost)
POST /predict/range    → Electric range regression (XGBoost)

Feature construction at inference time
---------------------------------------
Both models were trained on five features derived from three user inputs:

    User input          →  Derived feature
    ──────────────────────────────────────────────────────────────────
    make (str)          →  is_tesla   (int, 1 if TESLA else 0)
                        →  Make       (passed to AggregateFeatureTransformer
                                       for make_market_share lookup)
    model_year (int)    →  Model Year (raw, used by model directly)
                        →  vehicle_age (REFERENCE_YEAR − model_year)
    ev_type (BEV/PHEV)  →  is_bev    (int, 1 if BEV else 0)

    AggregateFeatureTransformer (fitted on training data, loaded from disk)
    appends make_market_share using the training distribution.
    Unseen makes → 0.0 (handled inside the transformer).

Model artefacts
---------------
All four .joblib files are loaded once at startup via the FastAPI lifespan
context manager.  If any file is missing the service starts but the affected
endpoint returns HTTP 503 until the file is present and the app is restarted.

Running locally
---------------
    cd <project_root>
    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

import datetime
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── src imports (project root must be on PYTHONPATH or installed as package) ──
from src.features import (
    MAKE_COL,
    REFERENCE_YEAR,
    YEAR_COL,
    AggregateFeatureTransformer,
)
from src.models import CAFV_FEATURES, CLASS_NAMES, RANGE_FEATURES

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# api/main.py → api/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"

_ARTEFACTS = {
    "cafv_model": MODEL_DIR / "cafv_classifier.joblib",
    "cafv_transformer": MODEL_DIR / "cafv_classifier_agg_transformer.joblib",
    "range_model": MODEL_DIR / "range_regressor.joblib",
    "range_transformer": MODEL_DIR / "range_regressor_agg_transformer.joblib",
}

# ──────────────────────────────────────────────────────────────────────────────
# Startup / shutdown — load models once into module-level state dict
# ──────────────────────────────────────────────────────────────────────────────

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all four .joblib artefacts at startup; release on shutdown."""
    for key, path in _ARTEFACTS.items():
        try:
            _state[key] = joblib.load(path)
            logger.info("Loaded %-20s ← %s", key, path.name)
        except FileNotFoundError:
            _state[key] = None
            logger.error("Artefact NOT FOUND: %s — endpoint will return 503", path)

    logger.info(
        "API ready. REFERENCE_YEAR=%d  MODEL_DIR=%s",
        REFERENCE_YEAR,
        MODEL_DIR,
    )
    yield
    _state.clear()


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WA State EV Analysis — ML Inference API",
    description=(
        "Production inference layer for the Washington State EV Population ML pipeline. "
        "Models: XGBoost CAFV eligibility classifier (F1-macro 0.9724) and "
        "XGBoost electric range regressor (R² 0.9857, MAE 5.28 miles)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

_CURRENT_YEAR = datetime.date.today().year


class VehicleInput(BaseModel):
    """
    Minimal vehicle attributes required for both ML endpoints.

    Both CAFV classification and range regression were trained on the same
    five features derived from these three inputs.
    """

    make: str = Field(
        ...,
        description="Vehicle manufacturer (e.g. TESLA, CHEVROLET, NISSAN). "
        "Case-insensitive. Unseen makes get make_market_share=0.",
        examples=["TESLA", "CHEVROLET", "NISSAN"],
    )
    model_year: int = Field(
        ...,
        ge=1990,
        le=_CURRENT_YEAR + 1,
        description=f"Model year of the vehicle. Must be between 1990 and {_CURRENT_YEAR + 1}.",
        examples=[2022, 2020, 2018],
    )
    ev_type: Literal["BEV", "PHEV"] = Field(
        ...,
        description="Battery Electric Vehicle (BEV) or Plug-in Hybrid Electric Vehicle (PHEV).",
        examples=["BEV", "PHEV"],
    )

    @field_validator("make")
    @classmethod
    def normalise_make(cls, v: str) -> str:
        """Uppercase and strip — matches training data convention."""
        return v.strip().upper()


class CAFVProbabilities(BaseModel):
    eligible: float = Field(..., ge=0.0, le=1.0)
    not_eligible: float = Field(..., ge=0.0, le=1.0)
    unknown: float = Field(..., ge=0.0, le=1.0)


class CAFVResponse(BaseModel):
    prediction: str = Field(..., description="Predicted CAFV eligibility class.")
    predicted_code: int = Field(
        ..., description="Integer class code: 0=eligible, 1=not_eligible, 2=unknown."
    )
    probabilities: CAFVProbabilities
    input_features: dict = Field(
        ..., description="Exact feature vector sent to the model."
    )
    model_version: str = "cafv-xgboost-optuna-hpo-v1"


class RangeResponse(BaseModel):
    predicted_range_miles: float = Field(
        ..., description="Predicted electric range in miles."
    )
    input_features: dict = Field(
        ..., description="Exact feature vector sent to the model."
    )
    note: str = Field(
        default=(
            "Trained on non-zero range records only. "
            "Predictions for PHEV vehicles reflect plug-in hybrid range, not all-electric."
        )
    )
    model_version: str = "range-xgboost-optuna-hpo-v1"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: dict[str, bool]
    reference_year: int
    api_version: str


# ──────────────────────────────────────────────────────────────────────────────
# Feature construction helper (shared by both endpoints)
# ──────────────────────────────────────────────────────────────────────────────


def _build_feature_row(inp: VehicleInput) -> pd.DataFrame:
    """
    Convert a VehicleInput into a single-row DataFrame with MAKE_COL and all
    base feature columns.  AggregateFeatureTransformer.transform() is applied
    by each endpoint individually (different transformers for CAFV vs range).

    Mirrors the exact transformations in src.features.engineer_features() and
    src.features.encode_ev_type() — no full pipeline needed at inference time
    since the saved agg_transformer already encodes the training distribution.
    """
    return pd.DataFrame(
        [
            {
                MAKE_COL: inp.make,  # AggregateFeatureTransformer needs this
                YEAR_COL: inp.model_year,  # "Model Year"
                "is_bev": int(inp.ev_type == "BEV"),
                "is_tesla": int(inp.make == "TESLA"),
                "vehicle_age": REFERENCE_YEAR - inp.model_year,
            }
        ]
    )


def _check_artefacts(*keys: str) -> None:
    """Raise HTTP 503 if any required artefact failed to load at startup."""
    missing = [k for k in keys if _state.get(k) is None]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artefact(s) not loaded: {missing}. "
                "Check that all .joblib files exist under data/models/ "
                "and restart the service."
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """
    Liveness + readiness check.

    Returns the load status of each model artefact.  A ``status`` of
    ``"degraded"`` means at least one artefact is missing; the corresponding
    endpoint will return HTTP 503.
    """
    loaded = {k: (_state.get(k) is not None) for k in _ARTEFACTS}
    return HealthResponse(
        status="ok" if all(loaded.values()) else "degraded",
        models_loaded=loaded,
        reference_year=REFERENCE_YEAR,
        api_version=app.version,
    )


@app.post("/predict/cafv", response_model=CAFVResponse, tags=["predictions"])
def predict_cafv(inp: VehicleInput) -> CAFVResponse:
    """
    Predict CAFV (Clean Alternative Fuel Vehicle) eligibility.

    Returns the predicted class, per-class probabilities, and the exact
    feature vector used so callers can verify the inference inputs.

    **Note on `make_market_share`:** The transformer was fitted on ~142K
    training records.  Unseen makes (post-2024 or non-WA-registered) receive
    a market share of 0.0, which biases predictions toward `not_eligible`.
    This is the correct conservative default.
    """
    _check_artefacts("cafv_model", "cafv_transformer")

    model: Any = _state["cafv_model"]
    agg: AggregateFeatureTransformer = _state["cafv_transformer"]

    row = _build_feature_row(inp)

    # Apply training-distribution statistics, then select model features only
    row_transformed = agg.transform(row)[CAFV_FEATURES]

    # Predicted class code (0=eligible, 1=not_eligible, 2=unknown)
    pred_code = int(model.predict(row_transformed)[0])
    pred_label = CLASS_NAMES[pred_code]

    # Per-class probabilities
    proba = model.predict_proba(row_transformed)[0]  # shape: (3,)
    probabilities = CAFVProbabilities(
        eligible=round(float(proba[0]), 4),
        not_eligible=round(float(proba[1]), 4),
        unknown=round(float(proba[2]), 4),
    )

    # Surface the exact feature vector for transparency / debugging
    feature_dict = row_transformed.iloc[0].to_dict()

    return CAFVResponse(
        prediction=pred_label,
        predicted_code=pred_code,
        probabilities=probabilities,
        input_features={
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in feature_dict.items()
        },
    )


@app.post("/predict/range", response_model=RangeResponse, tags=["predictions"])
def predict_range(inp: VehicleInput) -> RangeResponse:
    """
    Predict electric range in miles.

    The model was trained exclusively on vehicles where the DOL-recorded
    electric range > 0 (approximately 103K records after filtering).
    Zero-range records in the source dataset represent vehicles whose EPA
    range has not been entered into the DOL database — not actual zero-range
    vehicles.

    Test-set performance: MAE 5.28 miles · RMSE 11.88 miles · R² 0.9857
    Baseline (group-median by EV type): MAE 35.90 miles.
    """
    _check_artefacts("range_model", "range_transformer")

    model: Any = _state["range_model"]
    agg: AggregateFeatureTransformer = _state["range_transformer"]

    row = _build_feature_row(inp)
    row_transformed = agg.transform(row)[RANGE_FEATURES]

    predicted_range = float(model.predict(row_transformed)[0])
    # Clamp to physically plausible range — XGBoost can extrapolate slightly
    # below 0 for very old or unusual vehicles.
    predicted_range = max(0.0, round(predicted_range, 2))

    feature_dict = row_transformed.iloc[0].to_dict()

    return RangeResponse(
        predicted_range_miles=predicted_range,
        input_features={
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in feature_dict.items()
        },
    )
