"""
tests/test_model_trainer.py

Unit tests for src/model_trainer.py
Run with: python -m pytest tests/ -v
"""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model_trainer import (
    compare_models,
    evaluate_model,
    get_lightgbm,
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
    resample_smote,
    resample_undersample,
    scale_numeric,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imbalanced_dataset():
    """Small imbalanced dataset (90/10 split) for fast testing."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_s = pd.Series(y, name="label")
    return X_df, y_s


@pytest.fixture
def split_data(imbalanced_dataset):
    """Pre-split train/test data."""
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Tests: data preparation
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_shapes(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.2)
        assert len(X_tr) + len(X_te) == len(X)
        assert len(y_tr) + len(y_te) == len(y)

    def test_fraud_rate_preserved(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        _, _, y_tr, y_te = stratified_split(X, y, test_size=0.2)
        # Both splits should be close to overall fraud rate (10%)
        assert abs(y_tr.mean() - y.mean()) < 0.03
        assert abs(y_te.mean() - y.mean()) < 0.03


class TestScaleNumeric:
    def test_columns_scaled(self, split_data):
        X_tr, X_te, _, _ = split_data
        cols = ["f0", "f1", "f2"]
        X_tr_s, X_te_s, scaler = scale_numeric(X_tr, X_te, cols)
        # Scaled train columns should have near-zero mean
        assert abs(X_tr_s["f0"].mean()) < 0.01

    def test_no_leakage(self, split_data):
        """Scaler should be fit on train, applied to test — no test data in fit."""
        X_tr, X_te, _, _ = split_data
        cols = ["f0", "f1"]
        X_tr_s, X_te_s, scaler = scale_numeric(X_tr, X_te, cols)
        # scaler mean should match X_train mean, not X_test mean
        assert abs(scaler.mean_[0] - X_tr["f0"].mean()) < 0.01

    def test_returns_dataframes(self, split_data):
        X_tr, X_te, _, _ = split_data
        X_tr_s, X_te_s, _ = scale_numeric(X_tr, X_te, ["f0"])
        assert isinstance(X_tr_s, pd.DataFrame)
        assert isinstance(X_te_s, pd.DataFrame)


class TestResampleSmote:
    def test_minority_increased(self, split_data):
        X_tr, _, y_tr, _ = split_data
        before = y_tr.value_counts()[1]
        X_res, y_res = resample_smote(X_tr, y_tr)
        after = pd.Series(y_res).value_counts()[1]
        assert after > before

    def test_classes_balanced(self, split_data):
        X_tr, _, y_tr, _ = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        counts = pd.Series(y_res).value_counts()
        assert counts[0] == counts[1]

    def test_returns_dataframe(self, split_data):
        X_tr, _, y_tr, _ = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)


class TestResampleUndersample:
    def test_majority_reduced(self, split_data):
        X_tr, _, y_tr, _ = split_data
        before = y_tr.value_counts()[0]
        X_res, y_res = resample_undersample(X_tr, y_tr, sampling_strategy=0.5)
        after = pd.Series(y_res).value_counts()[0]
        assert after < before

    def test_returns_dataframe(self, split_data):
        X_tr, _, y_tr, _ = split_data
        X_res, y_res = resample_undersample(X_tr, y_tr)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)


# ---------------------------------------------------------------------------
# Tests: model definitions
# ---------------------------------------------------------------------------

class TestModelGetters:
    def test_logistic_regression_has_fit(self):
        model = get_logistic_regression()
        assert hasattr(model, "fit")

    def test_random_forest_has_fit(self):
        model = get_random_forest()
        assert hasattr(model, "fit")

    def test_xgboost_has_fit(self):
        model = get_xgboost()
        assert hasattr(model, "fit")

    def test_lightgbm_has_fit(self):
        model = get_lightgbm()
        assert hasattr(model, "fit")


# ---------------------------------------------------------------------------
# Tests: evaluate_model and compare_models
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_required_keys(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        result = evaluate_model(
            "LR", get_logistic_regression(),
            X_res, y_res, X_te, y_te,
            verbose=False,
        )
        for key in ["name", "f1", "auc_pr", "roc_auc", "tp", "fp", "fn", "tn"]:
            assert key in result

    def test_metrics_in_range(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        result = evaluate_model(
            "LR", get_logistic_regression(),
            X_res, y_res, X_te, y_te,
            verbose=False,
        )
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["auc_pr"] <= 1.0
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_cm_sums_to_test_size(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        result = evaluate_model(
            "LR", get_logistic_regression(),
            X_res, y_res, X_te, y_te,
            verbose=False,
        )
        total = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        assert total == len(y_te)


class TestCompareModels:
    def test_returns_dataframe(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        results = [
            evaluate_model("LR", get_logistic_regression(),
                           X_res, y_res, X_te, y_te, verbose=False),
        ]
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "AUC-PR" in df.columns

    def test_sorted_by_auc_pr(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        X_res, y_res = resample_smote(X_tr, y_tr)
        results = [
            evaluate_model("LR", get_logistic_regression(),
                           X_res, y_res, X_te, y_te, verbose=False),
            evaluate_model("RF", get_random_forest(),
                           X_res, y_res, X_te, y_te, verbose=False),
        ]
        df = compare_models(results)
        assert df["AUC-PR"].iloc[0] >= df["AUC-PR"].iloc[1]
