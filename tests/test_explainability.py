"""
tests/test_explainability.py

Unit tests for src/explainability.py
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.explainability import (
    compute_shap_values,
    mean_absolute_shap,
    top_fraud_drivers,
)
from src.model_trainer import get_lightgbm, resample_smote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_lgb_and_data():
    """
    Train a small LightGBM model on synthetic data.
    scope='module' so we train only once for all tests.
    """
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        weights=[0.85, 0.15],
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y_s = pd.Series(y, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=42, stratify=y_s
    )
    X_res, y_res = resample_smote(X_train, y_train)

    model = get_lightgbm()
    model.fit(X_res, y_res)

    return model, X_test, y_test, [f"feat_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Tests: compute_shap_values
# ---------------------------------------------------------------------------

class TestComputeShapValues:
    def test_returns_tuple(self, trained_lgb_and_data):
        model, X_test, _, _ = trained_lgb_and_data
        result = compute_shap_values(model, X_test)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shap_values_shape(self, trained_lgb_and_data):
        model, X_test, _, _ = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        assert shap_vals.shape == X_test.shape

    def test_shap_values_are_numeric(self, trained_lgb_and_data):
        model, X_test, _, _ = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        assert not np.isnan(shap_vals).any()

    def test_explainer_has_expected_value(self, trained_lgb_and_data):
        model, X_test, _, _ = trained_lgb_and_data
        _, explainer = compute_shap_values(model, X_test)
        assert hasattr(explainer, "expected_value")


# ---------------------------------------------------------------------------
# Tests: mean_absolute_shap
# ---------------------------------------------------------------------------

class TestMeanAbsoluteShap:
    def test_returns_series(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = mean_absolute_shap(shap_vals, feature_cols)
        assert isinstance(result, pd.Series)

    def test_length_matches_features(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = mean_absolute_shap(shap_vals, feature_cols)
        assert len(result) == len(feature_cols)

    def test_values_are_non_negative(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = mean_absolute_shap(shap_vals, feature_cols)
        assert (result >= 0).all()

    def test_sorted_descending(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = mean_absolute_shap(shap_vals, feature_cols)
        assert list(result.values) == sorted(result.values, reverse=True)


# ---------------------------------------------------------------------------
# Tests: top_fraud_drivers
# ---------------------------------------------------------------------------

class TestTopFraudDrivers:
    def test_returns_dataframe(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = top_fraud_drivers(shap_vals, feature_cols, top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = top_fraud_drivers(shap_vals, feature_cols, top_n=3)
        assert len(result) == 3

    def test_required_columns_present(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = top_fraud_drivers(shap_vals, feature_cols, top_n=5)
        for col in ["Feature", "Mean |SHAP|",
                    "Mean SHAP (fraud direction)",
                    "% samples pushing to fraud"]:
            assert col in result.columns

    def test_sorted_by_mean_shap(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = top_fraud_drivers(shap_vals, feature_cols, top_n=5)
        shap_vals_col = result["Mean |SHAP|"].values
        assert list(shap_vals_col) == sorted(shap_vals_col, reverse=True)

    def test_pct_column_between_0_and_100(self, trained_lgb_and_data):
        model, X_test, _, feature_cols = trained_lgb_and_data
        shap_vals, _ = compute_shap_values(model, X_test)
        result = top_fraud_drivers(shap_vals, feature_cols, top_n=5)
        assert (result["% samples pushing to fraud"] >= 0).all()
        assert (result["% samples pushing to fraud"] <= 100).all()
