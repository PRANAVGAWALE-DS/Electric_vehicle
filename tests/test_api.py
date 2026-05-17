"""
Tests for api/main.py — FastAPI inference endpoints.

Strategy: use unittest.mock to patch joblib.load so the test suite runs without
the actual .joblib artefacts on disk.  Each test injects a minimal stub that
implements the same interface (predict, predict_proba) that the API uses.

Run from project root:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Pre-import so patch("api.main.joblib.load") can resolve via getattr(api, 'main').
# Without this the patch context raises AttributeError before the fixture body runs.
import api.main as _api_module  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Model stubs — minimal, interface-compliant fakes
# ──────────────────────────────────────────────────────────────────────────────


class _FakeCAFVModel:
    """Stub XGBClassifier that always predicts class 0 (eligible)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([0] * len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # (n_samples, 3) — one hot on class 0
        return np.array([[0.85, 0.10, 0.05]] * len(X))


class _FakeRangeModel:
    """Stub XGBRegressor that always returns 220.0 miles."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([220.0] * len(X))


class _FakeAggTransformer:
    """
    Stub AggregateFeatureTransformer.

    transform() adds make_market_share = 0.45 (mocking Tesla dominance)
    and does NOT add median_range_by_make (not in CAFV_FEATURES / RANGE_FEATURES).
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["make_market_share"] = 0.45
        return X


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """
    TestClient with all four .joblib loads patched to return stubs.
    Patches "joblib.load" globally (not "api.main.joblib.load") so the patch
    survives importlib.reload(), which re-imports joblib into api.main fresh.
    """

    def _fake_load(path):
        name = str(path)
        if "cafv_classifier_agg_transformer" in name:
            return _FakeAggTransformer()
        if "cafv_classifier" in name:
            return _FakeCAFVModel()
        if "range_regressor_agg_transformer" in name:
            return _FakeAggTransformer()
        if "range_regressor" in name:
            return _FakeRangeModel()
        raise FileNotFoundError(f"Unexpected artefact path in test: {path}")

    with patch("joblib.load", side_effect=_fake_load):
        importlib.reload(_api_module)
        with TestClient(_api_module.app) as c:
            yield c


# ──────────────────────────────────────────────────────────────────────────────
# /health
# ──────────────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_all_models_loaded(self, client):
        body = client.get("/health").json()
        assert all(body["models_loaded"].values()), (
            f"Expected all models loaded, got: {body['models_loaded']}"
        )

    def test_reference_year_is_int(self, client):
        body = client.get("/health").json()
        assert isinstance(body["reference_year"], int)
        assert body["reference_year"] >= 2024

    def test_api_version_present(self, client):
        body = client.get("/health").json()
        assert "api_version" in body


# ──────────────────────────────────────────────────────────────────────────────
# /predict/cafv
# ──────────────────────────────────────────────────────────────────────────────


class TestCAFVPredict:
    _VALID = {"make": "TESLA", "model_year": 2022, "ev_type": "BEV"}

    def test_200_on_valid_input(self, client):
        r = client.post("/predict/cafv", json=self._VALID)
        assert r.status_code == 200

    def test_prediction_is_eligible(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        assert body["prediction"] == "eligible"
        assert body["predicted_code"] == 0

    def test_probabilities_sum_to_one(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        p = body["probabilities"]
        total = p["eligible"] + p["not_eligible"] + p["unknown"]
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_input_features_returned(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        features = body["input_features"]
        # All CAFV_FEATURES must be present
        for col in [
            "Model Year",
            "is_bev",
            "vehicle_age",
            "is_tesla",
            "make_market_share",
        ]:
            assert col in features, f"Missing feature '{col}' in input_features"

    def test_is_tesla_flag_set(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        assert body["input_features"]["is_tesla"] == 1

    def test_is_bev_flag_set(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        assert body["input_features"]["is_bev"] == 1

    def test_phev_sets_is_bev_zero(self, client):
        payload = {**self._VALID, "ev_type": "PHEV"}
        body = client.post("/predict/cafv", json=payload).json()
        assert body["input_features"]["is_bev"] == 0

    def test_make_normalised_to_uppercase(self, client):
        payload = {**self._VALID, "make": "tesla"}
        r = client.post("/predict/cafv", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["input_features"]["is_tesla"] == 1

    def test_non_tesla_sets_is_tesla_zero(self, client):
        payload = {**self._VALID, "make": "NISSAN"}
        body = client.post("/predict/cafv", json=payload).json()
        assert body["input_features"]["is_tesla"] == 0

    def test_vehicle_age_computed_correctly(self, client):
        import datetime

        ref = datetime.date.today().year
        body = client.post("/predict/cafv", json=self._VALID).json()
        expected_age = ref - self._VALID["model_year"]
        assert body["input_features"]["vehicle_age"] == expected_age

    def test_model_version_present(self, client):
        body = client.post("/predict/cafv", json=self._VALID).json()
        assert "model_version" in body

    # ── Validation error cases ─────────────────────────────────────────────

    def test_422_on_missing_make(self, client):
        r = client.post("/predict/cafv", json={"model_year": 2022, "ev_type": "BEV"})
        assert r.status_code == 422

    def test_422_on_invalid_ev_type(self, client):
        r = client.post(
            "/predict/cafv",
            json={"make": "TESLA", "model_year": 2022, "ev_type": "DIESEL"},
        )
        assert r.status_code == 422

    def test_422_on_year_too_old(self, client):
        r = client.post(
            "/predict/cafv",
            json={"make": "TESLA", "model_year": 1889, "ev_type": "BEV"},
        )
        assert r.status_code == 422

    def test_422_on_year_too_far_future(self, client):
        import datetime

        future_year = datetime.date.today().year + 5
        r = client.post(
            "/predict/cafv",
            json={"make": "TESLA", "model_year": future_year, "ev_type": "BEV"},
        )
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# /predict/range
# ──────────────────────────────────────────────────────────────────────────────


class TestRangePredict:
    _VALID = {"make": "TESLA", "model_year": 2022, "ev_type": "BEV"}

    def test_200_on_valid_input(self, client):
        r = client.post("/predict/range", json=self._VALID)
        assert r.status_code == 200

    def test_range_is_float(self, client):
        body = client.post("/predict/range", json=self._VALID).json()
        assert isinstance(body["predicted_range_miles"], float)

    def test_range_matches_stub(self, client):
        body = client.post("/predict/range", json=self._VALID).json()
        assert body["predicted_range_miles"] == 220.0

    def test_range_non_negative(self, client):
        body = client.post("/predict/range", json=self._VALID).json()
        assert body["predicted_range_miles"] >= 0.0

    def test_input_features_returned(self, client):
        body = client.post("/predict/range", json=self._VALID).json()
        for col in [
            "Model Year",
            "is_bev",
            "vehicle_age",
            "is_tesla",
            "make_market_share",
        ]:
            assert col in body["input_features"], f"Missing feature '{col}'"

    def test_note_field_present(self, client):
        body = client.post("/predict/range", json=self._VALID).json()
        assert "note" in body and len(body["note"]) > 0

    def test_422_on_missing_model_year(self, client):
        r = client.post("/predict/range", json={"make": "TESLA", "ev_type": "BEV"})
        assert r.status_code == 422

    def test_422_on_invalid_ev_type(self, client):
        r = client.post(
            "/predict/range",
            json={"make": "NISSAN", "model_year": 2020, "ev_type": "GAS"},
        )
        assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Cross-endpoint: same input → consistent derived features
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureConsistency:
    """
    The same vehicle input sent to both endpoints must produce identical
    derived features (is_bev, is_tesla, vehicle_age, Model Year).
    """

    _VALID = {"make": "CHEVROLET", "model_year": 2019, "ev_type": "PHEV"}

    def test_shared_features_match_across_endpoints(self, client):
        cafv_features = client.post("/predict/cafv", json=self._VALID).json()["input_features"]
        range_features = client.post("/predict/range", json=self._VALID).json()["input_features"]

        for col in ["Model Year", "is_bev", "vehicle_age", "is_tesla"]:
            assert cafv_features[col] == range_features[col], (
                f"Feature '{col}' differs: CAFV={cafv_features[col]}, Range={range_features[col]}"
            )
