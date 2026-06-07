"""
src/feature_engineering.py

Feature engineering for the Fraud_Data (e-commerce) dataset.
All functions are flake8-compliant (max line length 88).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------------
# 1. Temporal features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour_of_day, day_of_week, and time_since_signup (seconds).

    Requires: purchase_time (datetime), signup_time (datetime).
    """
    df = df.copy()

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0 = Monday

    delta = df["purchase_time"] - df["signup_time"]
    df["time_since_signup"] = delta.dt.total_seconds()

    # Flag suspicious sign: purchase before signup (data issue)
    n_negative = (df["time_since_signup"] < 0).sum()
    if n_negative > 0:
        print(f"  Warning: {n_negative} rows have purchase_time < signup_time.")

    return df


# ---------------------------------------------------------------------------
# 2. Transaction velocity features
# ---------------------------------------------------------------------------

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction velocity features per user_id:
    - user_tx_count     : total number of transactions for that user in the dataset
    - user_tx_per_day   : average transactions per day for that user

    These are computed over the full dataset (not per time window) because
    the dataset represents a bounded observation window.
    """
    df = df.copy()

    tx_count = df.groupby("user_id")["purchase_time"].transform("count")
    df["user_tx_count"] = tx_count

    # Days active = range between first and last purchase (min 1 to avoid /0)
    user_first = df.groupby("user_id")["purchase_time"].transform("min")
    user_last = df.groupby("user_id")["purchase_time"].transform("max")
    days_active = ((user_last - user_first).dt.total_seconds() / 86400).clip(lower=1)
    df["user_tx_per_day"] = df["user_tx_count"] / days_active

    return df


def add_device_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many unique users share each device_id.
    A device used by many users is a strong fraud signal.
    """
    df = df.copy()
    device_users = (
        df.groupby("device_id")["user_id"]
        .transform("nunique")
    )
    df["device_user_count"] = device_users
    return df


# ---------------------------------------------------------------------------
# 3. Encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode: source, browser, sex.
    drop_first=True to avoid the dummy variable trap.
    """
    df = df.copy()
    cat_cols = ["source", "browser", "sex"]
    # Only encode columns that are present
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True, dtype=int)
    return df


# ---------------------------------------------------------------------------
# 4. Scaling
# ---------------------------------------------------------------------------

NUMERIC_COLS_FRAUD = [
    "purchase_value",
    "age",
    "time_since_signup",
    "user_tx_count",
    "user_tx_per_day",
    "device_user_count",
    "hour_of_day",
    "day_of_week",
]

NUMERIC_COLS_CREDIT = ["Time", "Amount"]


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list,
    method: str = "standard",
) -> tuple:
    """
    Fit scaler on train_df[cols] and transform both train and test.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    cols     : list of column names to scale
    method   : 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)

    Returns
    -------
    (train_df_scaled, test_df_scaled, fitted_scaler)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    # Only scale columns that are present
    existing = [c for c in cols if c in train_df.columns]

    train_df[existing] = scaler.fit_transform(train_df[existing])
    test_df[existing] = scaler.transform(test_df[existing])

    return train_df, test_df, scaler


# ---------------------------------------------------------------------------
# 5. Full pipeline for Fraud_Data
# ---------------------------------------------------------------------------

def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline on Fraud_Data.
    Call this BEFORE train/test split (velocity features need the full dataset).
    Encoding and scaling are done AFTER split to prevent leakage.
    """
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = add_device_count(df)
    return df


def get_feature_columns(df: pd.DataFrame, target: str = "class") -> list:
    """
    Return the list of model input columns, excluding identifiers and target.
    """
    drop_cols = [
        target,
        "user_id",
        "device_id",
        "ip_address",
        "signup_time",
        "purchase_time",
        "ip_int",
        "lower_int",
        "upper_int",
        "country",          # we may use country-based dummies separately
    ]
    return [c for c in df.columns if c not in drop_cols]


def add_country_dummies(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    One-hot encode the top_n most frequent countries.
    All other countries → 'country_Other'.
    """
    df = df.copy()
    if "country" not in df.columns:
        return df

    top_countries = df["country"].value_counts().nlargest(top_n).index
    df["country_grouped"] = np.where(
        df["country"].isin(top_countries),
        df["country"],
        "Other",
    )
    df = pd.get_dummies(
        df, columns=["country_grouped"], drop_first=False, dtype=int
    )
    return df
