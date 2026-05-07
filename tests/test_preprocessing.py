"""
Unit tests for src/features.py

Tests are designed to run without the real dataset — all use synthetic
DataFrames that match the schema of the WA State EV dataset.

Run with: pytest tests/test_preprocessing.py -v
(requires `pip install -e .` from the project root so `src` is importable)
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CAFV_CODE_MAP,
    CAFV_COL,
    CAFV_LABEL_MAP,
    CITY_COL,
    COUNTY_COL,
    EV_TYPE_COL,
    MAKE_COL,
    MODEL_COL,
    RANGE_COL,
    RAW_COLS_TO_DROP,
    YEAR_COL,
    AggregateFeatureTransformer,
    drop_irrelevant_columns,
    encode_cafv,
    encode_ev_type,
    engineer_features,
    flag_zero_range,
    full_preprocessing_pipeline,
    handle_duplicates,
    handle_missing_values,
    load_raw,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal valid DataFrame matching the post-drop schema."""
    return pd.DataFrame(
        {
            COUNTY_COL: ["King", "Pierce", "Snohomish"],
            CITY_COL: ["Seattle", "Tacoma", "Everett"],
            "State": ["WA", "WA", "WA"],
            "Postal Code": [98101, 98402, 98201],
            YEAR_COL: [2022, 2019, 2021],
            MAKE_COL: ["TESLA", "NISSAN", "CHEVROLET"],
            MODEL_COL: ["MODEL 3", "LEAF", "BOLT EV"],
            EV_TYPE_COL: [
                "Battery Electric Vehicle (BEV)",
                "Battery Electric Vehicle (BEV)",
                "Battery Electric Vehicle (BEV)",
            ],
            CAFV_COL: [
                "Clean Alternative Fuel Vehicle Eligible",
                "Not eligible due to low battery range",
                "Eligibility unknown as battery range has not been researched",
            ],
            RANGE_COL: [358, 149, 0],
            "Electric Utility": [
                "PUGET SOUND ENERGY INC",
                "CITY OF TACOMA - (WA)",
                "PUGET SOUND ENERGY INC",
            ],
        }
    )


@pytest.fixture
def raw_df(minimal_df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame that includes columns to be dropped."""
    df = minimal_df.copy()
    for col in RAW_COLS_TO_DROP:
        df[col] = "dummy_value"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# drop_irrelevant_columns
# ──────────────────────────────────────────────────────────────────────────────


class TestDropIrrelevantColumns:
    def test_drops_all_listed_columns(self, raw_df):
        result = drop_irrelevant_columns(raw_df)
        for col in RAW_COLS_TO_DROP:
            assert col not in result.columns, f"Column '{col}' should have been dropped"

    def test_retains_expected_columns(self, raw_df, minimal_df):
        result = drop_irrelevant_columns(raw_df)
        for col in minimal_df.columns:
            assert col in result.columns, f"Column '{col}' was unexpectedly dropped"

    def test_idempotent_on_already_clean_df(self, minimal_df):
        """Calling on a df without the drop-columns should not raise."""
        result = drop_irrelevant_columns(minimal_df)
        assert set(result.columns) == set(minimal_df.columns)


# ──────────────────────────────────────────────────────────────────────────────
# handle_duplicates
# ──────────────────────────────────────────────────────────────────────────────


class TestHandleDuplicates:
    def test_removes_exact_duplicates(self, minimal_df):
        df_with_dup = pd.concat([minimal_df, minimal_df.iloc[[0]]], ignore_index=True)
        assert len(df_with_dup) == 4
        result = handle_duplicates(df_with_dup)
        assert len(result) == 3

    def test_no_duplicates_unchanged(self, minimal_df):
        result = handle_duplicates(minimal_df)
        assert len(result) == len(minimal_df)

    def test_index_is_reset(self, minimal_df):
        df_with_dup = pd.concat([minimal_df, minimal_df.iloc[[0]]], ignore_index=True)
        result = handle_duplicates(df_with_dup)
        assert list(result.index) == list(range(len(result)))


# ──────────────────────────────────────────────────────────────────────────────
# handle_missing_values
# ──────────────────────────────────────────────────────────────────────────────


class TestHandleMissingValues:
    def test_county_nan_filled(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, COUNTY_COL] = np.nan
        result = handle_missing_values(df)
        assert result.loc[0, COUNTY_COL] == "Unknown"

    def test_city_nan_filled(self, minimal_df):
        df = minimal_df.copy()
        df.loc[1, CITY_COL] = np.nan
        result = handle_missing_values(df)
        assert result.loc[1, CITY_COL] == "Unknown"

    def test_model_nan_filled_from_make(self, minimal_df):
        df = minimal_df.copy()
        df.loc[2, MODEL_COL] = np.nan
        result = handle_missing_values(df)
        assert df.loc[2, MAKE_COL].title() in result.loc[2, MODEL_COL]

    def test_no_nans_unchanged(self, minimal_df):
        result = handle_missing_values(minimal_df)
        assert result[COUNTY_COL].isna().sum() == 0
        assert result[CITY_COL].isna().sum() == 0
        assert result[MODEL_COL].isna().sum() == 0

    def test_original_not_mutated(self, minimal_df):
        df = minimal_df.copy()
        df.loc[0, COUNTY_COL] = np.nan
        handle_missing_values(df)
        assert pd.isna(df.loc[0, COUNTY_COL])  # original unchanged


# ──────────────────────────────────────────────────────────────────────────────
# flag_zero_range
# ──────────────────────────────────────────────────────────────────────────────


class TestFlagZeroRange:
    def test_adds_column(self, minimal_df):
        result = flag_zero_range(minimal_df)
        assert "range_is_zero" in result.columns

    def test_flags_correctly(self, minimal_df):
        result = flag_zero_range(minimal_df)
        assert result.loc[2, "range_is_zero"]
        assert not result.loc[0, "range_is_zero"]

    def test_count_matches(self, minimal_df):
        result = flag_zero_range(minimal_df)
        expected = (minimal_df[RANGE_COL] == 0).sum()
        assert result["range_is_zero"].sum() == expected

    def test_does_not_drop_zero_range_rows(self, minimal_df):
        result = flag_zero_range(minimal_df)
        assert len(result) == len(minimal_df)

    def test_original_not_mutated(self, minimal_df):
        df = minimal_df.copy()
        flag_zero_range(df)
        assert "range_is_zero" not in df.columns


# ──────────────────────────────────────────────────────────────────────────────
# encode_cafv
# ──────────────────────────────────────────────────────────────────────────────


class TestEncodeCafv:
    def test_adds_label_and_code_columns(self, minimal_df):
        result = encode_cafv(minimal_df)
        assert "cafv_label" in result.columns
        assert "cafv_code" in result.columns

    def test_label_values_correct(self, minimal_df):
        result = encode_cafv(minimal_df)
        assert result.loc[0, "cafv_label"] == "eligible"
        assert result.loc[1, "cafv_label"] == "not_eligible"
        assert result.loc[2, "cafv_label"] == "unknown"

    def test_code_values_correct(self, minimal_df):
        result = encode_cafv(minimal_df)
        assert result.loc[0, "cafv_code"] == CAFV_CODE_MAP["eligible"]
        assert result.loc[1, "cafv_code"] == CAFV_CODE_MAP["not_eligible"]
        assert result.loc[2, "cafv_code"] == CAFV_CODE_MAP["unknown"]

    def test_all_label_map_values_covered(self):
        """Every CAFV_LABEL_MAP key must map to a CAFV_CODE_MAP key."""
        for label_key, label_val in CAFV_LABEL_MAP.items():
            assert label_val in CAFV_CODE_MAP, (
                f"CAFV label '{label_val}' (from '{label_key}') has no entry in CAFV_CODE_MAP"
            )

    def test_unknown_cafv_string_defaults_to_unknown_not_nan(self, minimal_df):
        """
        An unrecognised CAFV string (e.g. from a DOL dataset update) must
        default to cafv_label='unknown' and cafv_code=2, not propagate NaN
        (which would crash px.sunburst and XGBClassifier.fit).
        """
        df = minimal_df.copy()
        df.loc[0, CAFV_COL] = "Some Future Eligibility Category"
        result = encode_cafv(df)
        assert result.loc[0, "cafv_label"] == "unknown"
        assert result.loc[0, "cafv_code"] == CAFV_CODE_MAP["unknown"]

    def test_no_nan_in_cafv_label_after_encode(self, minimal_df):
        """cafv_label must never be NaN after encoding, even with unexpected input."""
        df = minimal_df.copy()
        df.loc[1, CAFV_COL] = None
        result = encode_cafv(df)
        assert result["cafv_label"].isna().sum() == 0
        assert result["cafv_code"].isna().sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# encode_ev_type
# ──────────────────────────────────────────────────────────────────────────────


class TestEncodeEvType:
    def test_adds_is_bev_column(self, minimal_df):
        result = encode_ev_type(minimal_df)
        assert "is_bev" in result.columns

    def test_bev_is_1(self, minimal_df):
        result = encode_ev_type(minimal_df)
        bev_mask = minimal_df[EV_TYPE_COL] == "Battery Electric Vehicle (BEV)"
        assert (result.loc[bev_mask, "is_bev"] == 1).all()

    def test_phev_is_0(self):
        df = pd.DataFrame(
            {
                EV_TYPE_COL: ["Plug-in Hybrid Electric Vehicle (PHEV)"],
            }
        )
        result = encode_ev_type(df)
        assert result.loc[0, "is_bev"] == 0

    def test_dtype_is_int(self, minimal_df):
        result = encode_ev_type(minimal_df)
        assert result["is_bev"].dtype in [int, np.int64, np.int32]


# ──────────────────────────────────────────────────────────────────────────────
# engineer_features
# NOTE: engineer_features() produces ONLY vehicle_age and is_tesla.
# make_market_share / median_range_by_make are produced by
# AggregateFeatureTransformer (tested separately below).
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineerFeatures:
    @pytest.fixture
    def prepared_df(self, minimal_df) -> pd.DataFrame:
        """DataFrame after the encoding steps that engineer_features depends on."""
        return encode_ev_type(flag_zero_range(minimal_df))

    def test_adds_vehicle_age(self, prepared_df):
        result = engineer_features(prepared_df)
        assert "vehicle_age" in result.columns
        from src.features import REFERENCE_YEAR

        assert result.loc[0, "vehicle_age"] == REFERENCE_YEAR - prepared_df.loc[0, YEAR_COL]

    def test_vehicle_age_nonnegative_for_valid_years(self, prepared_df):
        result = engineer_features(prepared_df)
        assert (result["vehicle_age"] >= 0).all()

    def test_adds_is_tesla(self, prepared_df):
        result = engineer_features(prepared_df)
        assert "is_tesla" in result.columns
        assert result.loc[0, "is_tesla"] == 1  # TESLA
        assert result.loc[1, "is_tesla"] == 0  # NISSAN

    def test_does_not_add_aggregate_columns(self, prepared_df):
        """
        make_market_share and median_range_by_make must NOT appear — those
        require AggregateFeatureTransformer.fit_transform() on the training
        split only. Computing them here would leak test-set statistics.
        """
        result = engineer_features(prepared_df)
        assert "make_market_share" not in result.columns
        assert "median_range_by_make" not in result.columns

    def test_original_not_mutated(self, prepared_df):
        cols_before = set(prepared_df.columns)
        engineer_features(prepared_df)
        assert set(prepared_df.columns) == cols_before


# ──────────────────────────────────────────────────────────────────────────────
# AggregateFeatureTransformer
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def agg_df() -> pd.DataFrame:
    """
    DataFrame with Make and Electric Range suitable for AggregateFeatureTransformer.
    Includes 3 makes: TESLA (×3), NISSAN (×2), FORD (×1).
    """
    return pd.DataFrame(
        {
            MAKE_COL: ["TESLA", "TESLA", "TESLA", "NISSAN", "NISSAN", "FORD"],
            RANGE_COL: [300, 310, 320, 150, 160, 0],
        }
    )


class TestAggregateFeatureTransformer:
    def test_fit_transform_adds_market_share(self, agg_df):
        agg = AggregateFeatureTransformer()
        result = agg.fit_transform(agg_df)
        assert "make_market_share" in result.columns

    def test_market_share_values_correct(self, agg_df):
        agg = AggregateFeatureTransformer()
        result = agg.fit_transform(agg_df)
        # TESLA appears 3/6=0.5, NISSAN 2/6≈0.333, FORD 1/6≈0.167
        tesla_share = result.loc[result[MAKE_COL] == "TESLA", "make_market_share"].iloc[0]
        nissan_share = result.loc[result[MAKE_COL] == "NISSAN", "make_market_share"].iloc[0]
        assert pytest.approx(tesla_share, rel=1e-6) == 3 / 6
        assert pytest.approx(nissan_share, rel=1e-6) == 2 / 6

    def test_fit_transform_adds_median_range_by_make(self, agg_df):
        agg = AggregateFeatureTransformer()
        result = agg.fit_transform(agg_df)
        assert "median_range_by_make" in result.columns

    def test_median_range_excludes_zero_range_rows(self, agg_df):
        """
        FORD has only one record with range=0, which is excluded from the nonzero
        median calculation. FORD should fall back to the global median.
        """
        agg = AggregateFeatureTransformer()
        result = agg.fit_transform(agg_df)
        global_median = agg._global_median_range
        ford_median = result.loc[result[MAKE_COL] == "FORD", "median_range_by_make"].iloc[0]
        assert ford_median == global_median

    def test_transform_on_test_uses_train_statistics(self, agg_df):
        """
        Test set rows with known makes must receive the training-set market share,
        not statistics recomputed from the test set.
        """
        train = agg_df.iloc[:4].copy()  # TESLA×3, NISSAN×1
        test = agg_df.iloc[4:].copy()  # NISSAN×1, FORD×1

        agg = AggregateFeatureTransformer()
        agg.fit(train)
        result_test = agg.transform(test)

        # Train has TESLA=3/4=0.75, NISSAN=1/4=0.25
        nissan_share_in_test = result_test.loc[
            result_test[MAKE_COL] == "NISSAN", "make_market_share"
        ].iloc[0]
        assert pytest.approx(nissan_share_in_test, rel=1e-6) == 1 / 4

    def test_unseen_make_in_test_gets_zero_market_share(self, agg_df):
        """
        A make that was not in the training set must receive market_share=0,
        not NaN (which would crash XGBoost).
        """
        train = agg_df[agg_df[MAKE_COL] != "FORD"].copy()
        test = pd.DataFrame({MAKE_COL: ["FORD", "RIVIAN"], RANGE_COL: [0, 400]})

        agg = AggregateFeatureTransformer()
        agg.fit(train)
        result = agg.transform(test)

        assert (result["make_market_share"] == 0.0).all()

    def test_transform_before_fit_raises(self):
        agg = AggregateFeatureTransformer()
        with pytest.raises(RuntimeError, match="fit()"):
            agg.transform(pd.DataFrame({MAKE_COL: ["TESLA"], RANGE_COL: [300]}))

    def test_feature_names_out_before_fit_raises(self):
        agg = AggregateFeatureTransformer()
        with pytest.raises(RuntimeError, match="fit()"):
            _ = agg.feature_names_out

    def test_feature_names_out_after_fit(self, agg_df):
        agg = AggregateFeatureTransformer()
        agg.fit(agg_df)
        names = agg.feature_names_out
        assert "make_market_share" in names
        assert "median_range_by_make" in names

    def test_fit_requires_make_column(self):
        bad_df = pd.DataFrame({RANGE_COL: [100, 200]})
        agg = AggregateFeatureTransformer()
        with pytest.raises(ValueError, match="Make"):
            agg.fit(bad_df)

    def test_original_not_mutated(self, agg_df):
        original_cols = set(agg_df.columns)
        agg = AggregateFeatureTransformer()
        agg.fit_transform(agg_df)
        assert set(agg_df.columns) == original_cols


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline shape sanity
# ──────────────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    def test_load_raw_reads_csv(self, tmp_path):
        path = tmp_path / "ev.csv"
        pd.DataFrame({MAKE_COL: ["TESLA"], YEAR_COL: [2022]}).to_csv(path, index=False)
        result = load_raw(path)
        assert result.to_dict("records") == [{MAKE_COL: "TESLA", YEAR_COL: 2022}]

    def test_full_preprocessing_pipeline_uses_dol_id_before_drop(self, raw_df, tmp_path):
        path = tmp_path / "ev.csv"
        raw_df = raw_df.copy()
        raw_df["DOL Vehicle ID"] = [101, 101, 102]
        raw_df.to_csv(path, index=False)

        result = full_preprocessing_pipeline(path)

        assert len(result) == 2
        assert "DOL Vehicle ID" not in result.columns
        assert "vehicle_age" in result.columns

    def test_base_columns_present_after_pipeline_steps(self, minimal_df):
        """
        After the per-row encoding steps, the base ML features must exist.
        make_market_share / median_range_by_make are NOT included here —
        they require AggregateFeatureTransformer after the train/test split.
        """
        df = flag_zero_range(minimal_df)
        df = encode_cafv(df)
        df = encode_ev_type(df)
        df = engineer_features(df)

        required = [
            "range_is_zero",
            "cafv_label",
            "cafv_code",
            "is_bev",
            "vehicle_age",
            "is_tesla",
        ]
        for col in required:
            assert col in df.columns, f"Expected column '{col}' not found after pipeline"

    def test_aggregate_columns_present_after_agg_transformer(self, minimal_df):
        """
        make_market_share and median_range_by_make appear only after
        AggregateFeatureTransformer.fit_transform().
        """
        df = flag_zero_range(minimal_df)
        df = encode_cafv(df)
        df = encode_ev_type(df)
        df = engineer_features(df)

        agg = AggregateFeatureTransformer()
        df = agg.fit_transform(df)

        assert "make_market_share" in df.columns
        assert "median_range_by_make" in df.columns

    def test_row_count_preserved(self, minimal_df):
        """No rows should be dropped by encoding/engineering steps."""
        df = flag_zero_range(minimal_df)
        df = encode_cafv(df)
        df = encode_ev_type(df)
        df = engineer_features(df)
        assert len(df) == len(minimal_df)
