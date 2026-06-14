"""
src/model_trainer.py

Model building, training, evaluation, and cross-validation utilities.
All functions are flake8-compliant (max line length 88).
"""

import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# 1. Data preparation helpers
# ---------------------------------------------------------------------------

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train/test split preserving class distribution.

    Returns (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list,
) -> tuple:
    """
    Fit StandardScaler on X_train[num_cols], transform both splits.

    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler


def resample_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple:
    """Apply SMOTE to training data. Returns (X_resampled, y_resampled)."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_train.name),
    )


def resample_undersample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """
    Apply random undersampling to training data.
    Returns (X_resampled, y_resampled).
    """
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_train.name),
    )


# ---------------------------------------------------------------------------
# 2. Model definitions
# ---------------------------------------------------------------------------

def get_logistic_regression() -> LogisticRegression:
    """Return a configured Logistic Regression baseline."""
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
    )


def get_random_forest() -> RandomForestClassifier:
    """Return a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def get_xgboost() -> XGBClassifier:
    """Return a configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )


def get_lightgbm() -> LGBMClassifier:
    """Return a configured LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Fit model and evaluate on test set.

    Returns dict with keys: name, f1, auc_pr, roc_auc, tp, fp, fn, tn, model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  {name}")
        print(f"  F1={f1:.4f}  AUC-PR={auc_pr:.4f}  ROC-AUC={roc_auc:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(classification_report(
            y_test, y_pred, target_names=["Legit", "Fraud"]
        ))

    return {
        "name": name,
        "f1": round(f1, 4),
        "auc_pr": round(auc_pr, 4),
        "roc_auc": round(roc_auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "model": model,
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Return a sorted summary DataFrame from a list of evaluate_model dicts.
    Sorted by AUC-PR descending.
    """
    summary = pd.DataFrame([
        {
            "Model": r["name"],
            "F1": r["f1"],
            "AUC-PR": r["auc_pr"],
            "ROC-AUC": r["roc_auc"],
            "TP": r["tp"],
            "FP": r["fp"],
            "FN": r["fn"],
            "TN": r["tn"],
        }
        for r in results
    ])
    return summary.sort_values("AUC-PR", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Stratified K-Fold cross-validation
# ---------------------------------------------------------------------------

FRAUD_NUM_COLS = [
    "purchase_value", "age", "time_since_signup", "hour_of_day",
    "day_of_week", "user_tx_count", "user_tx_per_day", "device_user_count",
]

CREDIT_NUM_COLS = ["Time", "Amount"]


def cross_validate_fraud(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified K-Fold CV for Fraud_Data with SMOTE inside each fold.

    model_fn : callable that returns a fresh unfitted model each call.
    Returns dict with f1_scores, pr_scores, mean/std for each.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    f1_scores, pr_scores = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        # Scale inside fold to prevent leakage
        sc = StandardScaler()
        existing = [c for c in FRAUD_NUM_COLS if c in X_tr.columns]
        X_tr[existing] = sc.fit_transform(X_tr[existing])
        X_val[existing] = sc.transform(X_val[existing])

        # SMOTE inside fold
        smote = SMOTE(random_state=random_state)
        X_tr_r, y_tr_r = smote.fit_resample(X_tr, y_tr)

        model = model_fn()
        model.fit(X_tr_r, y_tr_r)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        f1 = round(f1_score(y_val, y_pred), 4)
        pr = round(average_precision_score(y_val, y_prob), 4)
        f1_scores.append(f1)
        pr_scores.append(pr)
        print(f"  Fold {fold + 1}: F1={f1:.4f}  AUC-PR={pr:.4f}")

    return {
        "f1_scores": f1_scores,
        "pr_scores": pr_scores,
        "f1_mean": round(float(np.mean(f1_scores)), 4),
        "f1_std": round(float(np.std(f1_scores)), 4),
        "pr_mean": round(float(np.mean(pr_scores)), 4),
        "pr_std": round(float(np.std(pr_scores)), 4),
    }


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

def save_model(path: str, model, scaler, feature_cols: list, name: str) -> None:
    """Pickle the model artefact bundle to disk."""
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "name": name,
            },
            f,
        )
    print(f"Model saved → {path}")


def load_model(path: str) -> dict:
    """Load a pickled model artefact bundle from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
