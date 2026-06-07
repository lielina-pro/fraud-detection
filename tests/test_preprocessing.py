"""
tests/test_preprocessing.py

Unit tests for data_preprocessing and feature_engineering modules.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import clean_fraud_data, ip_to_int, merge_ip_country
from src.feature_engineering import (
    add_time_features,
    add_velocity_features,
    encode_categoricals,
)


# ---------------------------------------------------------------------------
# Fixtures – small synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fraud_df():
    """Minimal Fraud_Data-like DataFrame for testing."""
    return pd.DataFrame({
        "user_id": [1, 2, 2, 3],
        "signup_time": pd.to_datetime([
            "2024-01-01 08:00:00",
            "2024-01-02 09:00:00",
            "2024-01-02 09:00:00",   # duplicate row
            "2024-01-03 10:00:00",
        ]),
        "purchase_time": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-02 11:00:00",
            "2024-01-02 11:00:00",   # duplicate row
            "2024-01-04 12:00:00",
        ]),
        "purchase_value": [50.0, 120.0, 120.0, 30.0],
        "device_id": ["d1", "d2", "d2", "d3"],
        "source": ["SEO", "Ads", "Ads", "SEO"],
        "browser": ["Chrome", "Firefox", "Firefox", "Safari"],
        "sex": ["M", "F", "F", "M"],
        "age": [25, 32, 32, 28],
        "ip_address": [
            "192.168.1.1",
            "10.0.0.1",
            "10.0.0.1",
            "172.16.0.5",
        ],
        "class": [0, 1, 1, 0],
    })


@pytest.fixture
def sample_ip_df():
    """Minimal IpAddress_to_Country-like DataFrame."""
    return pd.DataFrame({
        "lower_bound_ip_address": ["192.168.0.0", "10.0.0.0"],
        "upper_bound_ip_address": ["192.168.255.255", "10.255.255.255"],
        "country": ["CountryA", "CountryB"],
    })


# ---------------------------------------------------------------------------
# Tests: data_preprocessing
# ---------------------------------------------------------------------------

class TestCleanFraudData:
    def test_removes_duplicates(self, sample_fraud_df):
        cleaned = clean_fraud_data(sample_fraud_df)
        assert len(cleaned) == 3  # 4 rows - 1 duplicate

    def test_class_column_is_int(self, sample_fraud_df):
        cleaned = clean_fraud_data(sample_fraud_df)
        assert cleaned["class"].dtype == int

    def test_no_missing_class(self, sample_fraud_df):
        df = sample_fraud_df.copy()
        df.loc[0, "class"] = np.nan
        cleaned = clean_fraud_data(df)
        assert cleaned["class"].isna().sum() == 0


class TestIpToInt:
    def test_known_conversion(self):
        series = pd.Series(["0.0.0.0", "255.255.255.255", "192.168.1.1"])
        result = ip_to_int(series)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 4294967295
        assert result.iloc[2] == 3232235777

    def test_invalid_ip_returns_nan(self):
        # "not_an_ip" has no dots -> invalid
        # "999.0.0.1" has octet 999 > 255 -> invalid
        # "1.2.3" has only 3 parts -> invalid
        series = pd.Series(["not_an_ip", "999.0.0.1", "1.2.3"])
        result = ip_to_int(series)
        assert result.isna().all()


class TestMergeIpCountry:
    def test_country_column_exists(self, sample_fraud_df, sample_ip_df):
        merged = merge_ip_country(sample_fraud_df, sample_ip_df)
        assert "country" in merged.columns

    def test_matched_rows_have_country(self, sample_fraud_df, sample_ip_df):
        merged = merge_ip_country(sample_fraud_df, sample_ip_df)
        # 192.168.1.1 → CountryA; 10.0.0.1 → CountryB
        matched = merged.dropna(subset=["country"])
        assert len(matched) >= 1


# ---------------------------------------------------------------------------
# Tests: feature_engineering
# ---------------------------------------------------------------------------

class TestAddTimeFeatures:
    def test_columns_created(self, sample_fraud_df):
        result = add_time_features(sample_fraud_df)
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "time_since_signup" in result.columns

    def test_hour_of_day_range(self, sample_fraud_df):
        result = add_time_features(sample_fraud_df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_time_since_signup_positive(self, sample_fraud_df):
        result = add_time_features(sample_fraud_df)
        # Our sample data has purchase after signup
        assert (result["time_since_signup"] >= 0).all()


class TestAddVelocityFeatures:
    def test_columns_created(self, sample_fraud_df):
        result = add_velocity_features(sample_fraud_df)
        assert "user_tx_count" in result.columns
        assert "user_tx_per_day" in result.columns

    def test_user_with_two_transactions(self, sample_fraud_df):
        result = add_velocity_features(sample_fraud_df)
        # user_id=2 appears twice → count should be 2
        user2_count = result[result["user_id"] == 2]["user_tx_count"].iloc[0]
        assert user2_count == 2


class TestEncodeCategoricals:
    def test_original_cols_removed(self, sample_fraud_df):
        result = encode_categoricals(sample_fraud_df)
        assert "source" not in result.columns
        assert "browser" not in result.columns
        assert "sex" not in result.columns

    def test_dummy_cols_created(self, sample_fraud_df):
        result = encode_categoricals(sample_fraud_df)
        # At least one dummy column per original cat col
        dummy_cols = [c for c in result.columns if "_" in c]
        assert len(dummy_cols) > 0
