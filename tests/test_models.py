"""
Unit tests for src/models.py.

The training tests monkeypatch expensive estimator work so they exercise the
module's data splitting, metric plumbing, and return shapes without fitting
real XGBoost models.

Run with: pytest tests/test_models.py -v
(requires `pip install -e .` from the project root so `src` is importable)
"""

# import numpy as np
import pandas as pd

from src import models

# DummyClassifier / DummyRegressor removed — training tests now use
# n_trials=0 which skips Optuna and runs the real XGBoost pipeline on
# synthetic DataFrames.  The hardcoded fallback params complete in < 2s
# on the tiny datasets used here.


# ──────────────────────────────────────────────────────────────────────────────
# train_cafv_classifier
# ──────────────────────────────────────────────────────────────────────────────


def test_train_cafv_classifier_returns_expected_artifacts():
    """
    train_cafv_classifier returns all expected keys.
    n_trials=0 skips Optuna and uses hardcoded fallback params so the test
    completes in < 2s without any mocking.
    """
    rows = []
    for target in [0, 1, 2]:
        for i in range(20):  # 20 per class → 60 total; enough for all splits
            rows.append(
                {
                    "Electric Range": (100 + i * 5) if target != 2 else 0,
                    "Model Year": 2018 + (i % 5),
                    "is_bev": i % 2,
                    "vehicle_age": 6 - (i % 5),
                    "is_tesla": int(i % 3 == 0),
                    "Make": "TESLA" if i % 3 == 0 else "NISSAN",
                    "cafv_code": target,
                }
            )
    df = pd.DataFrame(rows)

    result = models.train_cafv_classifier(df, random_state=42, n_trials=0)

    expected_keys = {
        "model",
        "agg_transformer",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "cv_scores",
        "report",
        "study",
    }
    assert expected_keys.issubset(result)
    assert "macro avg" in result["report"]
    assert result["study"] is None  # n_trials=0 → no Optuna study created
    assert len(result["cv_scores"]) == 5  # 5-fold CV


def test_train_cafv_classifier_test_set_not_in_cv():
    """X_train and X_test row indices must be disjoint (no leakage)."""
    rows = []
    for target in [0, 1, 2]:
        for i in range(20):
            rows.append(
                {
                    "Electric Range": (100 + i) if target != 2 else 0,
                    "Model Year": 2018 + i % 5,
                    "is_bev": i % 2,
                    "vehicle_age": 6 - i % 5,
                    "is_tesla": int(i % 3 == 0),
                    "Make": "TESLA" if i % 3 == 0 else "FORD",
                    "cafv_code": target,
                }
            )
    df = pd.DataFrame(rows)

    result = models.train_cafv_classifier(df, random_state=0, n_trials=0)

    train_idx = set(result["X_train"].index)
    test_idx = set(result["X_test"].index)
    assert train_idx.isdisjoint(test_idx), "X_train and X_test share row indices"


# ──────────────────────────────────────────────────────────────────────────────
# train_range_regressor
# ──────────────────────────────────────────────────────────────────────────────


def test_train_range_regressor_excludes_zero_ranges():
    """Zero-range rows must be excluded; y_train and y_test must be all > 0."""
    rows = []
    # 5 zero-range rows — must be filtered out
    for i in range(5):
        rows.append(
            {
                "Electric Range": 0,
                "Model Year": 2020,
                "is_bev": 1,
                "vehicle_age": 4,
                "is_tesla": 0,
                "Make": "FORD",
            }
        )
    # 30 non-zero rows — used for training/testing
    for i in range(30):
        rows.append(
            {
                "Electric Range": 80 + i * 5,
                "Model Year": 2018 + (i % 5),
                "is_bev": i % 2,
                "vehicle_age": 6 - (i % 5),
                "is_tesla": int(i % 3 == 0),
                "Make": "TESLA" if i % 3 == 0 else "NISSAN",
            }
        )
    df = pd.DataFrame(rows)

    result = models.train_range_regressor(df, random_state=42, n_trials=0)

    assert (result["y_train"] > 0).all(), "Zero-range rows leaked into y_train"
    assert (result["y_test"] > 0).all(), "Zero-range rows leaked into y_test"
    assert set(result["metrics"]) == {"mae", "rmse", "r2", "baseline_mae"}
    assert len(result["y_pred"]) == len(result["y_test"])


def test_train_range_regressor_metrics_keys():
    """Return dict must contain all expected top-level keys including study."""
    rows = []
    for i in range(30):
        rows.append(
            {
                "Electric Range": 100 + i * 5,
                "Model Year": 2019 + (i % 4),
                "is_bev": i % 2,
                "vehicle_age": 5 - (i % 4),
                "is_tesla": 0,
                "Make": "NISSAN",
            }
        )
    df = pd.DataFrame(rows)

    result = models.train_range_regressor(df, random_state=0, n_trials=0)

    expected_keys = {
        "model",
        "agg_transformer",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "y_pred",
        "metrics",
        "study",
    }
    assert expected_keys.issubset(result)
    assert result["study"] is None  # n_trials=0 → no Optuna study created


# ──────────────────────────────────────────────────────────────────────────────
# save_model / load_model
# ──────────────────────────────────────────────────────────────────────────────


def test_save_and_load_model_round_trip(tmp_path):
    model = {"name": "small-test-model", "version": 1}

    path = models.save_model(model, tmp_path, "example")
    loaded = models.load_model(path)

    assert path == tmp_path / "example.joblib"
    assert loaded == model
