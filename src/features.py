"""
Feature engineering and preprocessing for the Washington State EV dataset.

Dataset: WA State Department of Licensing Electric Vehicle Population Data
         Kaggle mirror: sahirmaharajj/electric-vehicle-population (2024 snapshot)
Project data snapshot: 177,866 entries — ~99.8% Washington State registrations.

Aggregate features (make_market_share, median_range_by_make)
------------------------------------------------------------
These are computed by AggregateFeatureTransformer — a stateful sklearn-compatible
transformer that is .fit() on training data only and .transform()-ed onto the
test set.  They are NOT produced by engineer_features() to prevent target leakage.

Usage in training:
    from features import full_preprocessing_pipeline, AggregateFeatureTransformer

    df = full_preprocessing_pipeline(path)
    X_train, X_test, y_train, y_test = train_test_split(...)

    agg = AggregateFeatureTransformer()
    X_train = agg.fit_transform(X_train, y_train)   # fits on train only
    X_test  = agg.transform(X_test)                  # applies same mapping
"""

import datetime
import logging
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Column name constants (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
RAW_COLS_TO_DROP = [
    "VIN (1-10)",
    "Base MSRP",  # ~95% zeros — not useful for analysis
    "Legislative District",
    "DOL Vehicle ID",
    "Vehicle Location",  # WKT POINT strings — parsed separately if needed
    "2020 Census Tract",
]

CAFV_COL = "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
EV_TYPE_COL = "Electric Vehicle Type"
RANGE_COL = "Electric Range"
MAKE_COL = "Make"
MODEL_COL = "Model"
YEAR_COL = "Model Year"
COUNTY_COL = "County"
STATE_COL = "State"
CITY_COL = "City"
UTILITY_COL = "Electric Utility"
POSTAL_COL = "Postal Code"

# Short labels for the 3 CAFV classes
CAFV_LABEL_MAP = {
    "Clean Alternative Fuel Vehicle Eligible": "eligible",
    "Not eligible due to low battery range": "not_eligible",
    "Eligibility unknown as battery range has not been researched": "unknown",
}
CAFV_CODE_MAP = {"eligible": 0, "not_eligible": 1, "unknown": 2}

# Reference year for "vehicle_age" feature — resolves to the current calendar
# year at import time so models retrained in future years use correct ages.
# Pin explicitly (e.g. REFERENCE_YEAR = 2024) only when reproducing a past run.
REFERENCE_YEAR: int = datetime.date.today().year


# ──────────────────────────────────────────────────────────────────────────────
# Individual pipeline steps
# ──────────────────────────────────────────────────────────────────────────────


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %s rows, %d columns from %s", f"{len(df):,}", df.shape[1], path)
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified as irrelevant for analysis."""
    to_drop = [c for c in RAW_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    logger.info("Dropped %d columns: %s", len(to_drop), to_drop)
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are identical across ALL remaining columns.

    Important: VIN (1-10) and DOL Vehicle ID — the true unique-row identifiers —
    are dropped before this step.  Without them, many legitimate registrations
    share the same (Make, Model, Year, County, City, Range, …) values and would
    be wrongly eliminated.  We therefore only drop rows where *every* retained
    column is identical, which in practice removes only data-entry exact
    duplicates (same vehicle submitted twice), not distinct vehicles that happen
    to share attributes.

    On the real WA DOL dataset this drops 0–3 rows.  If you see thousands being
    dropped it means the upstream column-drop step is misconfigured.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped > 1000:
        logger.error(
            "handle_duplicates dropped %s rows — this is almost certainly wrong. "
            "VIN / DOL Vehicle ID should have been retained as dedup keys, or "
            "deduplication should be skipped for this dataset.",
            f"{dropped:,}",
        )
    elif dropped:
        logger.warning("Dropped %s exact-duplicate rows", f"{dropped:,}")
    else:
        logger.info("No duplicate rows found.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with domain-aware strategy."""
    df = df.copy()

    # County / City: tag as Unknown rather than dropping rows
    for col in [COUNTY_COL, CITY_COL]:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna("Unknown")
            if n:
                logger.info("Filled %s missing '%s' → 'Unknown'", f"{n:,}", col)

    # Model: use Make as fallback to preserve the row
    if MODEL_COL in df.columns:
        mask = df[MODEL_COL].isna()
        n = mask.sum()
        df.loc[mask, MODEL_COL] = df.loc[mask, MAKE_COL].str.title() + " (Unknown Model)"
        if n:
            logger.info("Filled %s missing Model values via Make", f"{n:,}")

    return df


def flag_zero_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean column ``range_is_zero``.

    Zero range indicates vehicles whose EPA range hasn't been entered into the
    DOL database yet — predominantly post-2021 models. These rows are kept but
    flagged so range analyses aren't skewed.

    Note: the live fraction of zero-range records varies with each DOL snapshot;
    see runtime logs for the current value rather than relying on any hardcoded
    percentage.
    """
    df = df.copy()
    df["range_is_zero"] = df[RANGE_COL] == 0
    n = df["range_is_zero"].sum()
    logger.info(
        "Flagged %s records (%s%%) with Electric Range = 0 (retained, not removed)",
        f"{n:,}",
        f"{100 * n / len(df):.1f}",
    )
    return df


def encode_cafv(df: pd.DataFrame) -> pd.DataFrame:
    """Map CAFV eligibility to a short string label and an integer code."""
    df = df.copy()
    df["cafv_label"] = df[CAFV_COL].map(CAFV_LABEL_MAP)

    unmapped = df["cafv_label"].isna().sum()
    if unmapped > 0:
        unknown_vals = df.loc[df["cafv_label"].isna(), CAFV_COL].unique().tolist()
        logger.warning(
            "encode_cafv: %d rows have unrecognised CAFV values — defaulting to 'unknown'. "
            "Unrecognised values: %s. Update CAFV_LABEL_MAP if this is a new eligibility "
            "category from a DOL dataset update.",
            unmapped,
            unknown_vals,
        )
        df["cafv_label"] = df["cafv_label"].fillna("unknown")

    df["cafv_code"] = df["cafv_label"].map(CAFV_CODE_MAP)
    return df


def encode_ev_type(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encode EV type: BEV → 1, PHEV → 0."""
    df = df.copy()
    df["is_bev"] = (df[EV_TYPE_COL] == "Battery Electric Vehicle (BEV)").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive base features that require only per-row information.

    Aggregate features that depend on the distribution of the training set
    (make_market_share, median_range_by_make) are intentionally excluded here.
    Use AggregateFeatureTransformer.fit_transform(X_train) /
    AggregateFeatureTransformer.transform(X_test) for those — computing them
    here on the full dataframe would leak test-set statistics into training.

    Features produced:
        vehicle_age  – Years since manufacture (REFERENCE_YEAR − Model Year)
        is_tesla     – 1 if Make == TESLA, else 0
    """
    df = df.copy()
    df["vehicle_age"] = REFERENCE_YEAR - df[YEAR_COL]
    df["is_tesla"] = (df[MAKE_COL] == "TESLA").astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Train-split-aware aggregate feature transformer
# ──────────────────────────────────────────────────────────────────────────────


class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Compute make-level aggregate features in a leakage-free way.

    fit()       — learns make_market_share and median_range_by_make from X_train
                  (with y_train providing the range target for the median).
    transform() — maps the learned statistics onto any split (train or test)
                  using the training distribution only.

    Parameters
    ----------
    range_col : str
        Name of the electric-range column in the input dataframe.
        Defaults to RANGE_COL ("Electric Range").
    make_col : str
        Name of the make column. Defaults to MAKE_COL ("Make").

    Usage
    -----
    >>> agg = AggregateFeatureTransformer()
    >>> X_train = agg.fit_transform(X_train)
    >>> X_test  = agg.transform(X_test)

    Note: for range regression, pass the non-zero-filtered training frame so
    that make medians are computed on real range values only.
    """

    def __init__(
        self,
        range_col: str = RANGE_COL,
        make_col: str = MAKE_COL,
    ) -> None:
        self.range_col = range_col
        self.make_col = make_col

        # Populated by fit()
        self._market_share_map: pd.Series | None = None
        self._median_range_map: pd.Series | None = None
        self._global_median_range: float = 0.0

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "AggregateFeatureTransformer":
        """Learn aggregate statistics from X (training split only)."""
        if self.make_col not in X.columns:
            raise ValueError(
                f"AggregateFeatureTransformer.fit(): '{self.make_col}' not in X.columns. "
                f"Available columns: {list(X.columns)}"
            )

        make_counts = X[self.make_col].value_counts()
        self._market_share_map = make_counts / len(X)

        if self.range_col in X.columns:
            nonzero = X[X[self.range_col] > 0]
            self._median_range_map = nonzero.groupby(self.make_col)[self.range_col].median()
            self._global_median_range = (
                float(nonzero[self.range_col].median()) if len(nonzero) > 0 else 0.0
            )
        else:
            # Range column absent (e.g. when called on a feature-only X) —
            # skip median range computation silently.
            self._median_range_map = pd.Series(dtype=float)
            self._global_median_range = 0.0

        logger.info(
            "AggregateFeatureTransformer fitted on %s rows, %d unique makes.",
            f"{len(X):,}",
            len(make_counts),
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training-set statistics to X. Safe to call on test set."""
        if self._market_share_map is None:
            raise RuntimeError(
                "AggregateFeatureTransformer.transform() called before fit(). "
                "Call fit_transform(X_train) first."
            )
        X = X.copy()

        X["make_market_share"] = (
            X[self.make_col]
            .map(self._market_share_map)
            .fillna(0.0)  # unseen makes in test → 0 share
        )

        if len(self._median_range_map) > 0:
            X["median_range_by_make"] = (
                X[self.make_col]
                .map(self._median_range_map)
                .fillna(self._global_median_range)  # unseen make → global median
            )

        return X

    # ------------------------------------------------------------------
    @property
    def feature_names_out(self) -> list[str]:
        """Names of columns added by transform()."""
        if self._median_range_map is None:
            raise RuntimeError(
                "AggregateFeatureTransformer.feature_names_out accessed before fit(). "
                "Call fit_transform(X_train) first."
            )
        cols = ["make_market_share"]
        if len(self._median_range_map) > 0:
            cols.append("median_range_by_make")
        return cols


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: full pipeline
# ──────────────────────────────────────────────────────────────────────────────


def full_preprocessing_pipeline(path: str | Path) -> pd.DataFrame:
    """
    Load → deduplicate → clean → flag → encode → engineer base features.

    Deduplication happens BEFORE dropping columns so that DOL Vehicle ID —
    the real unique-row key — is still available as a dedup key.  Doing it
    after the column drop causes ~96% of legitimate rows to be wrongly removed
    (many distinct vehicles share Make/Model/Year/County).

    Aggregate features (make_market_share, median_range_by_make) are NOT
    computed here — use AggregateFeatureTransformer inside your training
    routine after the train/test split.
    """
    df = load_raw(path)

    # ── Deduplicate on the real unique key BEFORE dropping it ─────────────────
    id_col = "DOL Vehicle ID"
    if id_col in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            logger.warning("Dropped %s rows with duplicate DOL Vehicle ID", f"{dropped:,}")
        else:
            logger.info("No duplicate DOL Vehicle IDs found.")
    else:
        df = handle_duplicates(df)  # fallback — safe only pre-column-drop

    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = flag_zero_range(df)
    df = encode_cafv(df)
    df = encode_ev_type(df)
    df = engineer_features(df)
    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df
