"""
Unit tests for src/models.py.

The training tests monkeypatch expensive estimator work so they exercise the
module's data splitting, metric plumbing, and return shapes without fitting
real XGBoost models.

Run with: pytest tests/test_models.py -v
(requires `pip install -e .` from the project root so `src` is importable)
"""

import numpy as np
import pandas as pd

from src import models


class DummyClassifier:
    """
    Minimal XGBClassifier stand-in.

    best_iteration and n_estimators must be present: train_cafv_classifier
    logs both via model.best_iteration and model.n_estimators after fit().
    """

    best_iteration: int = 50
    n_estimators: int = 400

    def fit(self, X_train, y_train, eval_set=None, verbose=False):
        self.classes_ = sorted(y_train.unique())
        return self

    def predict(self, X_test):
        return np.resize(np.array(self.classes_), len(X_test))


class DummyRegressor:
    """
    Minimal XGBRegressor stand-in.

    best_iteration and n_estimators must be present: train_range_regressor
    logs both after fit().
    """

    best_iteration: int = 60
    n_estimators: int = 500

    def fit(self, X_train, y_train, eval_set=None, verbose=False):
        self.mean_ = float(y_train.mean())
        return self

    def predict(self, X_test):
        return np.full(len(X_test), self.mean_)


# ──────────────────────────────────────────────────────────────────────────────
# Model factory configuration
# ──────────────────────────────────────────────────────────────────────────────


def test_build_cafv_classifier_configuration():
    model = models.build_cafv_classifier(random_state=7)

    assert model.n_estimators == 400
    assert model.max_depth == 6
    assert model.random_state == 7
    assert model.eval_metric == "mlogloss"


def test_build_range_regressor_configuration():
    model = models.build_range_regressor(random_state=11)

    assert model.n_estimators == 500
    assert model.max_depth == 6
    assert model.random_state == 11
    assert model.eval_metric == "rmse"


def test_build_cafv_classifier_early_stopping_rounds_default_is_none():
    """Default must be None so sklearn CV can call .fit() without eval_set."""
    model = models.build_cafv_classifier(random_state=42)
    assert model.early_stopping_rounds is None


def test_build_range_regressor_early_stopping_rounds_default_is_none():
    model = models.build_range_regressor(random_state=42)
    assert model.early_stopping_rounds is None


# ──────────────────────────────────────────────────────────────────────────────
# train_cafv_classifier
# ──────────────────────────────────────────────────────────────────────────────


def test_train_cafv_classifier_returns_expected_artifacts(monkeypatch):
    # build_cafv_classifier is called with (random_state, early_stopping_rounds=...)
    # — the lambda must accept **kwargs to avoid TypeError.
    monkeypatch.setattr(
        models,
        "build_cafv_classifier",
        lambda random_state, **kwargs: DummyClassifier(),
    )
    monkeypatch.setattr(models, "cross_val_score", lambda *args, **kwargs: np.array([0.7, 0.8]))

    rows = []
    for target in [0, 1, 2]:
        for i in range(6):
            rows.append(
                {
                    "Electric Range": 100 + i,
                    "Model Year": 2020 + (i % 3),
                    "is_bev": i % 2,
                    "vehicle_age": 4 - (i % 3),
                    "make_market_share": 0.25,
                    "median_range_by_make": 120,
                    "is_tesla": int(i % 2 == 0),
                    "Make": "TESLA" if i % 2 == 0 else "NISSAN",
                    "cafv_code": target,
                }
            )
    df = pd.DataFrame(rows)

    result = models.train_cafv_classifier(df, random_state=42)

    assert isinstance(result["model"], DummyClassifier)
    assert result["cv_scores"].tolist() == [0.7, 0.8]
    assert {"X_train", "X_test", "y_train", "y_test", "report"}.issubset(result)
    assert "macro avg" in result["report"]


def test_train_cafv_classifier_test_set_not_in_cv(monkeypatch):
    """
    The test set must never appear inside a CV fold.
    With 80/20 primary split, X_train rows must not overlap X_test rows.
    """
    monkeypatch.setattr(
        models,
        "build_cafv_classifier",
        lambda random_state, **kwargs: DummyClassifier(),
    )
    monkeypatch.setattr(models, "cross_val_score", lambda *args, **kwargs: np.array([0.75]))

    rows = []
    for target in [0, 1, 2]:
        for i in range(10):
            rows.append(
                {
                    "Electric Range": 100 + i,
                    "Model Year": 2018 + i % 5,
                    "is_bev": i % 2,
                    "vehicle_age": 6 - i % 5,
                    "make_market_share": 0.2,
                    "median_range_by_make": 130,
                    "is_tesla": int(i % 3 == 0),
                    "Make": "TESLA" if i % 3 == 0 else "FORD",
                    "cafv_code": target,
                }
            )
    df = pd.DataFrame(rows)
    result = models.train_cafv_classifier(df, random_state=0)

    train_idx = set(result["X_train"].index)
    test_idx = set(result["X_test"].index)
    assert train_idx.isdisjoint(test_idx), "X_train and X_test share row indices"


# ──────────────────────────────────────────────────────────────────────────────
# train_range_regressor
# ──────────────────────────────────────────────────────────────────────────────


def test_train_range_regressor_excludes_zero_ranges(monkeypatch):
    # build_range_regressor is called with (random_state, early_stopping_rounds=20)
    # — the lambda must accept **kwargs.
    monkeypatch.setattr(
        models,
        "build_range_regressor",
        lambda random_state, **kwargs: DummyRegressor(),
    )
    df = pd.DataFrame(
        {
            "Electric Range": [0, 80, 90, 100, 120, 130, 150, 160],
            "Model Year": [2020, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "is_bev": [1, 0, 0, 1, 1, 0, 1, 0],
            "vehicle_age": [4, 6, 5, 4, 3, 2, 1, 0],
            "make_market_share": [0.2] * 8,
            "is_tesla": [0, 0, 1, 1, 0, 1, 0, 1],
            "Make": [
                "FORD",
                "NISSAN",
                "TESLA",
                "TESLA",
                "FORD",
                "TESLA",
                "FORD",
                "NISSAN",
            ],
        }
    )

    result = models.train_range_regressor(df, random_state=42)

    assert isinstance(result["model"], DummyRegressor)
    assert (result["y_train"] > 0).all(), "Zero-range rows leaked into y_train"
    assert (result["y_test"] > 0).all(), "Zero-range rows leaked into y_test"
    assert set(result["metrics"]) == {"mae", "rmse", "r2", "baseline_mae"}
    assert len(result["y_pred"]) == len(result["y_test"])


def test_train_range_regressor_metrics_keys(monkeypatch):
    """Return dict must contain all expected top-level keys."""
    monkeypatch.setattr(
        models,
        "build_range_regressor",
        lambda random_state, **kwargs: DummyRegressor(),
    )
    df = pd.DataFrame(
        {
            "Electric Range": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "Model Year": [2019] * 10,
            "is_bev": [1, 0] * 5,
            "vehicle_age": [5] * 10,
            "make_market_share": [0.3] * 10,
            "is_tesla": [0] * 10,
            "Make": ["NISSAN"] * 10,
        }
    )
    result = models.train_range_regressor(df, random_state=0)

    expected_keys = {
        "model",
        "agg_transformer",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "y_pred",
        "metrics",
    }
    assert expected_keys.issubset(result)


# ──────────────────────────────────────────────────────────────────────────────
# save_model / load_model
# ──────────────────────────────────────────────────────────────────────────────


def test_save_and_load_model_round_trip(tmp_path):
    model = {"name": "small-test-model", "version": 1}

    path = models.save_model(model, tmp_path, "example")
    loaded = models.load_model(path)

    assert path == tmp_path / "example.joblib"
    assert loaded == model
