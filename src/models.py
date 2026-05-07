"""
ML models for the Washington State EV dataset.

Model 1 — CAFV Eligibility Classifier
    3-class XGBoost: eligible / not_eligible / unknown
    Primary ML problem: ~51% of records have unresearched battery ranges
    (cafv_code == 2).  Can we predict eligibility from vehicle characteristics
    alone — without access to the range value that defines the target?

    Feature set is intentionally range-free:
        "Electric Range" and "median_range_by_make" are excluded because
        Electric Range = 0 is the *definition* of cafv_code == 2 (unknown).
        Including it produces F1 ≈ 1.0 via tautological lookup, not learning.

Model 2 — Electric Range Regressor
    XGBoost regression on non-zero range records only.
    Baseline: median range stratified by EV type (BEV / PHEV).
    Reports whether the model beats this strong domain-aware baseline.

Aggregate feature handling (leakage-free)
-----------------------------------------
Both trainers instantiate AggregateFeatureTransformer, fit it on X_train only,
and apply it to X_test — never on the full dataframe before splitting.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBClassifier, XGBRegressor

from src.features import MAKE_COL, RANGE_COL, AggregateFeatureTransformer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Feature lists — explicit, no "select by dtype" magic
# ──────────────────────────────────────────────────────────────────────────────

# Intentionally excludes:
#   "Electric Range"        — encodes the target directly (0 ↔ class "unknown")
#   "median_range_by_make"  — range-derived; secondary leakage into CAFV target
#   "make_market_share"     — added by AggregateFeatureTransformer after split
#
# AggregateFeatureTransformer.fit_transform(X_train) appends
# "make_market_share" in-place; this list reflects the state BEFORE that step.
# The trainer adds the column and references it explicitly.
CAFV_BASE_FEATURES = [
    "Model Year",
    "is_bev",
    "vehicle_age",
    "is_tesla",
]

# "make_market_share" is added post-transform; listed here for documentation.
CAFV_FEATURES = CAFV_BASE_FEATURES + ["make_market_share"]

# Range regressor does not use "median_range_by_make" (not in RANGE_FEATURES)
# because it would leak the range target distribution.  "make_market_share"
# is added by AggregateFeatureTransformer after splitting.
RANGE_BASE_FEATURES = [
    "Model Year",
    "is_bev",
    "vehicle_age",
    "is_tesla",
]

RANGE_FEATURES = RANGE_BASE_FEATURES + ["make_market_share"]

CAFV_TARGET = "cafv_code"
CLASS_NAMES = ["eligible", "not_eligible", "unknown"]


# ──────────────────────────────────────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────────────────────────────────────


def build_cafv_classifier(
    random_state: int = 42,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    """
    XGBoost multiclass classifier for CAFV eligibility.

    Parameters
    ----------
    early_stopping_rounds :
        Pass None for CV use (sklearn calls .fit() with no eval_set —
        XGBoost raises ValueError if early_stopping_rounds is set without one).
        Pass an int for the final fit, which supplies eval_set explicitly.
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
    )


def build_range_regressor(
    random_state: int = 42,
    early_stopping_rounds: int | None = None,
) -> XGBRegressor:
    """
    XGBoost regressor for Electric Range prediction.

    Parameters
    ----------
    early_stopping_rounds :
        Pass None for CV use. Pass an int for the final fit with eval_set.
    """
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training routines
# ──────────────────────────────────────────────────────────────────────────────


def train_cafv_classifier(df: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    """
    Train and evaluate the CAFV eligibility classifier.

    Leakage controls applied here:
      1. "Electric Range" and "median_range_by_make" are absent from
         CAFV_BASE_FEATURES — they encode the target definition.
      2. AggregateFeatureTransformer is fit on X_train only; the same
         learned mapping is applied to X_test.
      3. eval_set for early stopping uses a validation slice carved from
         X_train — the test set never touches .fit().

    Returns a dict with keys:
        model, agg_transformer, X_train, X_test, y_train, y_test,
        cv_scores, report
    """
    # Include MAKE_COL so AggregateFeatureTransformer.transform() can look up
    # market share per make.  It is dropped from X_train / X_test after
    # transformation — the model never sees the raw make string.
    _cols_for_split = CAFV_BASE_FEATURES + [MAKE_COL]
    X = df[_cols_for_split].copy()
    y = df[CAFV_TARGET].copy()

    # ── Primary train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # ── Aggregate features: fit on X_train only ───────────────────────────────
    # fit() on the train-index slice of df (which has RANGE_COL) so that
    # median_range_by_make is computed from real range values, not the
    # feature-only X_train frame.
    agg = AggregateFeatureTransformer()
    agg.fit(df.loc[X_train.index])  # full df rows → has Make + Electric Range

    # transform() reads Make from X_train / X_test and appends make_market_share
    # (and median_range_by_make).  Make is then dropped; only CAFV_FEATURES
    # (which excludes Make and range-derived columns) reach the model.
    X_train = agg.transform(X_train)[CAFV_FEATURES]
    X_test = agg.transform(X_test)[CAFV_FEATURES]

    # ── Validation slice for early stopping ───────────────────────────────────
    # Carved BEFORE CV so val rows are never seen inside any CV fold.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=random_state
    )

    # ── 5-fold stratified CV on X_tr only (val rows excluded) ────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_model = build_cafv_classifier(random_state, early_stopping_rounds=None)
    cv_scores = cross_val_score(cv_model, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info("CAFV CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Final fit with early stopping on val slice (NOT the test set) ─────────
    model = build_cafv_classifier(random_state, early_stopping_rounds=20)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "CAFV classifier stopped at iteration %d / %d",
        model.best_iteration,
        model.n_estimators,
    )

    # ── Test-set evaluation ───────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return dict(
        model=model,
        agg_transformer=agg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv_scores=cv_scores,
        report=report,
    )


def train_range_regressor(df: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    """
    Train and evaluate the Electric Range regressor.

    Only uses records where Electric Range > 0 — zero values represent
    unverified ranges, not actual zero-range vehicles.

    Leakage controls applied here:
      - AggregateFeatureTransformer is fit on X_train (non-zero slice) only.
        make_market_share reflects the training distribution exclusively.
      - eval_set for early stopping uses a val slice from X_train; test set
        never touches .fit().

    Baseline: group-median of training set stratified by is_bev.
    """
    df_nonzero = df[df[RANGE_COL] > 0].copy()
    logger.info(
        "Range regressor training set: %s records (zero-range excluded)",
        f"{len(df_nonzero):,}",
    )

    # Include MAKE_COL so AggregateFeatureTransformer.transform() can look up
    # market share per make.  Dropped after transformation.
    _cols_for_split = RANGE_BASE_FEATURES + [MAKE_COL]
    X = df_nonzero[_cols_for_split].copy()
    y = df_nonzero[RANGE_COL].copy()

    # ── Primary train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # ── Aggregate features: fit on X_train (non-zero slice) only ─────────────
    # fit() on the non-zero train rows of df so range medians are computed on
    # real range values; Make column is present in both df_nonzero and X_train.
    agg = AggregateFeatureTransformer()
    agg.fit(df_nonzero.loc[X_train.index])

    X_train = agg.transform(X_train)[RANGE_FEATURES]
    X_test = agg.transform(X_test)[RANGE_FEATURES]

    # ── Baseline: median by BEV/PHEV group (computed on y_train only) ─────────
    group_medians = y_train.groupby(X_train["is_bev"]).median()
    baseline_pred = X_test["is_bev"].map(group_medians).fillna(y_train.median())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    logger.info("Baseline MAE (median by EV type): %.2f miles", baseline_mae)

    # ── Validation slice for early stopping ───────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )

    # ── Final fit ─────────────────────────────────────────────────────────────
    model = build_range_regressor(random_state, early_stopping_rounds=20)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "Range regressor stopped at iteration %d / %d",
        model.best_iteration,
        model.n_estimators,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = dict(mae=mae, rmse=rmse, r2=r2, baseline_mae=baseline_mae)
    logger.info(
        "Range Regressor — MAE: %.2f miles, RMSE: %.2f miles, R²: %.4f "
        "(vs baseline MAE: %.2f miles)",
        mae,
        rmse,
        r2,
        baseline_mae,
    )

    return dict(
        model=model,
        agg_transformer=agg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        metrics=metrics,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper — apply saved transformer + model to new data
# ──────────────────────────────────────────────────────────────────────────────


def predict_cafv(
    df_new: pd.DataFrame,
    model: XGBClassifier,
    agg: AggregateFeatureTransformer,
) -> np.ndarray:
    """
    Run CAFV classification on new (post-pipeline) data using a fitted model
    and a previously fitted AggregateFeatureTransformer.

    Parameters
    ----------
    df_new : DataFrame produced by full_preprocessing_pipeline()
    model  : trained XGBClassifier from train_cafv_classifier()
    agg    : fitted AggregateFeatureTransformer from train_cafv_classifier()

    Returns
    -------
    np.ndarray of integer class codes (0=eligible, 1=not_eligible, 2=unknown)
    """
    # Include MAKE_COL for the transformer lookup, then drop via final select
    X = df_new[CAFV_BASE_FEATURES + [MAKE_COL]].copy()
    X = agg.transform(X)[CAFV_FEATURES]
    return model.predict(X)


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────


def save_model(model: Any, directory: Path, name: str) -> Path:
    """Persist a model artefact with joblib."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}.joblib"
    joblib.dump(model, out)
    logger.info("Model saved → %s", out)
    return out


def save_transformer(agg: AggregateFeatureTransformer, directory: Path, name: str) -> Path:
    """Persist a fitted AggregateFeatureTransformer alongside its model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}_agg_transformer.joblib"
    joblib.dump(agg, out)
    logger.info("Transformer saved → %s", out)
    return out


def load_model(path: Path) -> Any:
    """Load a model artefact previously saved with joblib."""
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


if __name__ == "__main__":
    import sys

    # root is the project directory (two levels above src/models.py)
    root = Path(__file__).resolve().parent.parent

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_path = root / "data" / "raw" / "Electric_Vehicle_Population_Data.csv"

    if not data_path.exists():
        logger.error("Dataset not found at %s. Skipping test run.", data_path)
        sys.exit(1)

    from src.features import full_preprocessing_pipeline

    logger.info("Starting test training run...")
    df_clean = full_preprocessing_pipeline(data_path)

    logger.info("--- Training CAFV Classifier ---")
    cafv_results = train_cafv_classifier(df_clean)

    logger.info("--- Training Range Regressor ---")
    range_results = train_range_regressor(df_clean)

    model_dir = root / "data" / "models"
    save_model(cafv_results["model"], model_dir, "cafv_classifier")
    save_transformer(cafv_results["agg_transformer"], model_dir, "cafv")
    logger.info("Test run complete.")
