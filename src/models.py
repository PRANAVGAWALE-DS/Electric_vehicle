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
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBClassifier, XGBRegressor

from src.features import MAKE_COL, RANGE_COL, AggregateFeatureTransformer

# mlflow.xgboost.autolog() inspects _estimator_type to identify model class.
# XGBoost's sklearn API omits this, causing autolog to silently skip
# per-iteration metric logging. Patching here fixes it globally.
XGBClassifier._estimator_type = "classifier"
XGBRegressor._estimator_type = "regressor"

# Suppress per-trial INFO spam; WARNING still shows study-level summaries.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

# Fraction of the training slice used for Optuna CV objective.
# HPO generalises from a subsample — avoids 40 × 5 full-dataset fits.
# At ~60% of a 120K-row train slice ≈ 72K rows per CV fold.
HPO_SUBSAMPLE_FRAC: float = 0.60

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
# Training routines
# ──────────────────────────────────────────────────────────────────────────────


def train_cafv_classifier(
    df: pd.DataFrame,
    random_state: int = 42,
    n_trials: int = 40,
) -> dict[str, Any]:
    """
    Train and evaluate the CAFV eligibility classifier.

    When n_trials > 0, Optuna searches the XGBoost hyperparameter space using
    5-fold stratified CV on a 60% subsample of the training slice (fast), then
    retrains on the full training slice with best_params + early stopping.
    Each trial is logged as a nested MLflow run — requires an active parent run
    (started in the notebook with ``mlflow.start_run()``).

    When n_trials == 0, defaults from build_cafv_classifier() are used with no
    search (backward-compatible with the original implementation).

    Leakage controls:
      1. "Electric Range" and "median_range_by_make" absent from CAFV_BASE_FEATURES.
      2. AggregateFeatureTransformer fit on X_train only.
      3. eval_set for early stopping uses a val slice, never the test set.
      4. HPO subsample drawn from X_tr only (val rows already excluded).

    Returns a dict with keys:
        model, agg_transformer, X_train, X_test, y_train, y_test,
        cv_scores, report, study (None when n_trials == 0)
    """
    _cols_for_split = CAFV_BASE_FEATURES + [MAKE_COL]
    X = df[_cols_for_split].copy()
    y = df[CAFV_TARGET].copy()

    # ── Primary train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # ── Aggregate features: fit on X_train only ───────────────────────────────
    agg = AggregateFeatureTransformer()
    agg.fit(df.loc[X_train.index])

    X_train = agg.transform(X_train)[CAFV_FEATURES]
    X_test = agg.transform(X_test)[CAFV_FEATURES]

    # ── Validation slice for early stopping (carved before HPO) ──────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=random_state
    )

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    study: optuna.Study | None = None

    if n_trials > 0:
        # Subsample X_tr for fast CV inside each trial.
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=HPO_SUBSAMPLE_FRAC, random_state=random_state
        )
        hpo_idx, _ = next(sss.split(X_tr, y_tr))
        X_hpo = X_tr.iloc[hpo_idx]
        y_hpo = y_tr.iloc[hpo_idx]
        logger.info(
            "HPO subsample: %s / %s rows (%.0f%%)",
            f"{len(X_hpo):,}",
            f"{len(X_tr):,}",
            100 * HPO_SUBSAMPLE_FRAC,
        )

        cv_hpo = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        def _cafv_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }
            clf = XGBClassifier(
                **params,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,  # sklearn owns the parallelism via cross_val_score
            )
            scores = cross_val_score(
                clf,
                X_hpo,
                y_hpo,
                cv=cv_hpo,
                scoring="f1_macro",
                n_jobs=-1,  # parallelise across 5 CV folds, not inside XGBoost
                error_score=0.0,  # NaN folds (undefined F1 from degenerate params) → 0.0
            )
            score = float(scores.mean())

            # Log each trial as a nested MLflow run.
            # Requires an active parent run in the calling notebook cell.
            with mlflow.start_run(nested=True, run_name=f"cafv-trial-{trial.number}"):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "cv_f1_macro": score,
                        "cv_f1_macro_std": float(scores.std()),
                    }
                )

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
            study_name="cafv-classifier-hpo",
        )
        study.optimize(
            _cafv_objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,),  # log failed trials, never abort the study
        )

        best_params = study.best_params
        logger.info(
            "Optuna CAFV — best CV F1-macro: %.4f  best params: %s",
            study.best_value,
            best_params,
        )
    else:
        # Fallback defaults (no HPO)
        best_params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    # ── Final fit on full X_tr with best_params + early stopping ─────────────
    model = XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(
        "CAFV classifier stopped at iteration %d / %d",
        model.best_iteration,
        model.n_estimators,
    )

    # ── 5-fold CV on full X_tr with best params (final reported CV score) ─────
    cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_model = XGBClassifier(
        **best_params, eval_metric="mlogloss", random_state=random_state, n_jobs=-1
    )
    cv_scores = cross_val_score(cv_model, X_tr, y_tr, cv=cv_final, scoring="f1_macro", n_jobs=-1)
    logger.info("CAFV final CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

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
        study=study,
    )


def train_range_regressor(
    df: pd.DataFrame,
    random_state: int = 42,
    n_trials: int = 40,
) -> dict[str, Any]:
    """
    Train and evaluate the Electric Range regressor.

    Only uses records where Electric Range > 0 — zero values represent
    unverified ranges, not actual zero-range vehicles.

    When n_trials > 0, Optuna searches the XGBoost hyperparameter space using
    5-fold CV on a 60% subsample of X_tr (neg-MAE objective), then retrains
    on the full X_tr with best_params + early stopping. Each trial is logged
    as a nested MLflow run.

    When n_trials == 0, defaults from build_range_regressor() are used.

    Leakage controls:
      - AggregateFeatureTransformer fit on X_train (non-zero slice) only.
      - Baseline group-medians computed on y_train only.
      - HPO subsample drawn from X_tr only (val rows excluded).

    Returns a dict with keys:
        model, agg_transformer, X_train, X_test, y_train, y_test,
        y_pred, metrics, study (None when n_trials == 0)
    """
    df_nonzero = df[df[RANGE_COL] > 0].copy()
    logger.info(
        "Range regressor training set: %s records (zero-range excluded)",
        f"{len(df_nonzero):,}",
    )

    _cols_for_split = RANGE_BASE_FEATURES + [MAKE_COL]
    X = df_nonzero[_cols_for_split].copy()
    y = df_nonzero[RANGE_COL].copy()

    # ── Primary train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # ── Aggregate features: fit on X_train (non-zero slice) only ─────────────
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

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    study: optuna.Study | None = None

    if n_trials > 0:
        # Subsample X_tr for fast CV inside each trial.
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=HPO_SUBSAMPLE_FRAC, random_state=random_state
        )
        # StratifiedShuffleSplit requires integer-like y — bin range into deciles.
        y_tr_bins = pd.qcut(y_tr, q=10, labels=False, duplicates="drop")
        hpo_idx, _ = next(sss.split(X_tr, y_tr_bins))
        X_hpo = X_tr.iloc[hpo_idx]
        y_hpo = y_tr.iloc[hpo_idx]
        logger.info(
            "HPO subsample: %s / %s rows (%.0f%%)",
            f"{len(X_hpo):,}",
            f"{len(X_tr):,}",
            100 * HPO_SUBSAMPLE_FRAC,
        )

        cv_hpo = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        def _range_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }
            reg = XGBRegressor(
                **params,
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=1,  # sklearn owns the parallelism via cross_val_score
            )
            # neg_mean_absolute_error: cross_val_score maximises, so higher = better.
            scores = cross_val_score(
                reg,
                X_hpo,
                y_hpo,
                cv=cv_hpo,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,  # parallelise across 5 CV folds, not inside XGBoost
                error_score=0.0,  # degenerate params → 0.0 (worst neg-MAE = 0)
            )
            neg_mae = float(scores.mean())  # negative MAE; optimise to maximise

            with mlflow.start_run(nested=True, run_name=f"range-trial-{trial.number}"):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "cv_neg_mae": neg_mae,
                        "cv_neg_mae_std": float(scores.std()),
                        "cv_mae": -neg_mae,
                    }
                )

            return neg_mae

        study = optuna.create_study(
            direction="maximize",  # maximise neg-MAE = minimise MAE
            sampler=optuna.samplers.TPESampler(seed=random_state),
            study_name="range-regressor-hpo",
        )
        study.optimize(
            _range_objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,),  # log failed trials, never abort the study
        )

        best_params = study.best_params
        logger.info(
            "Optuna Range — best CV MAE: %.4f  best params: %s",
            -study.best_value,
            best_params,
        )
    else:
        best_params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    # ── Final fit on full X_tr with best_params + early stopping ─────────────
    model = XGBRegressor(
        **best_params,
        eval_metric="rmse",
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
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
        study=study,
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
