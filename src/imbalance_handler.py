"""
src/imbalance_handler.py

Resampling strategies for handling severe class imbalance.
All functions are flake8-compliant (max line length 88).
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def report_class_balance(y: pd.Series, label: str = "") -> None:
    """Print class distribution and fraud rate."""
    counts = y.value_counts().sort_index()
    total = len(y)
    fraud_rate = counts.get(1, 0) / total * 100
    tag = f" [{label}]" if label else ""
    print(f"Class distribution{tag}:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:,} ({cnt / total * 100:.2f}%)")
    print(f"  Fraud rate: {fraud_rate:.2f}%\n")


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to training data.

    Why SMOTE over random undersampling here?
    - Fraud_Data has ~9% fraud rate — undersampling would discard too many
      legitimate examples, losing generalisation signal.
    - SMOTE generates synthetic fraud samples in feature space, preserving
      the full legitimate dataset while balancing classes.

    IMPORTANT: Only ever call this on TRAINING data, never on test data.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print("After SMOTE:")
    report_class_balance(y_res, label="train")
    return X_res, y_res


def apply_undersample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """
    Apply random undersampling to training data.

    Use this for creditcard.csv where the dataset is large (284k rows) and
    undersampling the majority class still leaves sufficient training data.

    sampling_strategy=0.5 → minority class will be 1/3 of the resulting set.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = rus.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print("After undersampling:")
    report_class_balance(y_res, label="train")
    return X_res, y_res
