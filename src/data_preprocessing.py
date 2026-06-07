"""
src/data_preprocessing.py

Utility functions for loading, cleaning, and merging the fraud datasets.
All functions are flake8-compliant (max line length 88).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_fraud_data(path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv and parse datetime columns."""
    df = pd.read_csv(path)
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    return df


def load_ip_country(path: str) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv."""
    return pd.read_csv(path)


def load_creditcard(path: str) -> pd.DataFrame:
    """Load creditcard.csv (may be inside a zip)."""
    if path.endswith(".zip"):
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def report_missing(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Return a DataFrame summarising missing values."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": missing, "pct": pct})
    report = report[report["missing"] > 0].sort_values("pct", ascending=False)
    label = f" [{name}]" if name else ""
    print(f"Missing values{label}:\n{report}\n")
    return report


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fraud_Data:
    - Drop duplicate rows
    - Drop duplicate device IDs (same device used by multiple users is a fraud
      signal we keep as a feature, so we only remove exact full-row duplicates)
    - Ensure correct dtypes
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows from Fraud_Data.")

    # Ensure numeric types
    df["purchase_value"] = pd.to_numeric(df["purchase_value"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Drop rows where target is missing
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)
    return df.reset_index(drop=True)


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean creditcard data:
    - Drop full duplicates
    - Ensure Class is integer
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows from creditcard.")

    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. IP → Country geolocation merge
# ---------------------------------------------------------------------------

def ip_to_int(ip_series: pd.Series) -> pd.Series:
    """
    Convert a Series of dotted-decimal IP strings to integer.
    E.g. '192.168.1.1' → 3232235777
    Non-parseable IPs become NaN.
    """
    def _convert(ip):
        try:
            parts = str(ip).split(".")
            if len(parts) != 4:
                return np.nan
            octets = [int(p) for p in parts]
            if any(o < 0 or o > 255 for o in octets):
                return np.nan
            return sum(o << (8 * (3 - i)) for i, o in enumerate(octets))
        except (ValueError, AttributeError):
            return np.nan

    return ip_series.apply(_convert)


def merge_ip_country(
    fraud_df: pd.DataFrame,
    ip_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Fraud_Data with IpAddress_to_Country using range-based lookup.

    Strategy:
    - Convert IP strings to integers in both dataframes.
    - Sort ip_df by lower_bound.
    - Use pd.merge_asof to find the matching range for each transaction IP.
    - Keep only rows where ip_int <= upper_bound_ip_int (valid match).
    """
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # Convert IPs to integers
    fraud_df["ip_int"] = ip_to_int(fraud_df["ip_address"])
    ip_df["lower_int"] = ip_to_int(ip_df["lower_bound_ip_address"])
    ip_df["upper_int"] = ip_to_int(ip_df["upper_bound_ip_address"])

    # Drop rows with unparseable IPs
    fraud_df = fraud_df.dropna(subset=["ip_int"])
    fraud_df["ip_int"] = fraud_df["ip_int"].astype(np.int64)

    ip_df = ip_df.dropna(subset=["lower_int", "upper_int"])
    ip_df["lower_int"] = ip_df["lower_int"].astype(np.int64)
    ip_df["upper_int"] = ip_df["upper_int"].astype(np.int64)

    ip_df_sorted = ip_df.sort_values("lower_int").reset_index(drop=True)
    fraud_sorted = fraud_df.sort_values("ip_int").reset_index(drop=True)

    merged = pd.merge_asof(
        fraud_sorted,
        ip_df_sorted[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward",
    )

    # Invalidate rows where ip_int exceeds the matched range's upper bound
    merged.loc[merged["ip_int"] > merged["upper_int"], "country"] = np.nan

    unmatched = merged["country"].isna().sum()
    pct = unmatched / len(merged) * 100
    print(f"IP-to-country: {unmatched} IPs could not be matched ({pct:.1f}%).")

    return merged
