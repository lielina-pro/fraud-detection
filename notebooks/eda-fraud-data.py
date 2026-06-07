# %% [markdown]
# # EDA – Fraud_Data.csv
# **Project:** Improved Detection of Fraud Cases for E-commerce Transactions
# **Week 5 & 6 | 10 Academy KAIM**
#
# This notebook covers:
# 1. Data loading and initial inspection
# 2. Missing value analysis
# 3. Data cleaning
# 4. Univariate distributions
# 5. Bivariate analysis (features vs. fraud target)
# 6. Class imbalance analysis
# 7. IP-to-country geolocation merge
# 8. Feature engineering
# 9. Resampling (SMOTE)
# 10. Save processed data

# %% [markdown]
# ## 0. Imports

# %%
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add project root to path so we can import src/
sys.path.insert(0, os.path.abspath(".."))

from src.data_preprocessing import (  # noqa: E402
    clean_fraud_data,
    load_fraud_data,
    load_ip_country,
    merge_ip_country,
    report_missing,
)
from src.feature_engineering import (  # noqa: E402
    add_country_dummies,
    build_fraud_features,
    encode_categoricals,
    get_feature_columns,
    scale_features,
    NUMERIC_COLS_FRAUD,
)
from src.imbalance_handler import (  # noqa: E402
    apply_smote,
    report_class_balance,
)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100

# %%
# ---------------------------------------------------------------------------
# EDIT THESE PATHS to point to your downloaded data files
# ---------------------------------------------------------------------------
FRAUD_PATH = "../data/raw/Fraud_Data.csv"
IP_COUNTRY_PATH = "../data/raw/IpAddress_to_Country.csv"
PROCESSED_DIR = "../data/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Data

# %%
fraud_raw = load_fraud_data(FRAUD_PATH)
ip_country = load_ip_country(IP_COUNTRY_PATH)

print("Fraud_Data shape:", fraud_raw.shape)
print("IpAddress_to_Country shape:", ip_country.shape)
fraud_raw.head()

# %% [markdown]
# ## 2. Initial Inspection

# %%
print("=== Data Types ===")
print(fraud_raw.dtypes)
print("\n=== Basic Statistics ===")
fraud_raw.describe()

# %%
# Missing values
report_missing(fraud_raw, name="Fraud_Data")
report_missing(ip_country, name="IpAddress_to_Country")

# %% [markdown]
# ## 3. Data Cleaning

# %%
fraud_clean = clean_fraud_data(fraud_raw)
print("Shape after cleaning:", fraud_clean.shape)

# %% [markdown]
# ## 4. Class Imbalance Analysis

# %%
report_class_balance(fraud_clean["class"], label="Fraud_Data raw")

fig, ax = plt.subplots(figsize=(6, 4))
counts = fraud_clean["class"].value_counts()
ax.bar(
    ["Legitimate (0)", "Fraud (1)"],
    counts.values,
    color=["steelblue", "tomato"],
    edgecolor="white",
)
ax.set_title("Class Distribution – Fraud_Data", fontsize=13)
ax.set_ylabel("Number of Transactions")
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "class_distribution_fraud.png"))
plt.show()

# %% [markdown]
# ## 5. Univariate Analysis

# %%
# Numeric columns
num_cols = ["purchase_value", "age"]
fig, axes = plt.subplots(1, len(num_cols), figsize=(12, 4))
for ax, col in zip(axes, num_cols):
    sns.histplot(fraud_clean[col], ax=ax, bins=40, kde=True, color="steelblue")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
plt.suptitle("Univariate – Numeric Features", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "univariate_numeric.png"))
plt.show()

# %%
# Categorical columns
cat_cols = ["source", "browser", "sex"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(14, 4))
for ax, col in zip(axes, cat_cols):
    order = fraud_clean[col].value_counts().index
    sns.countplot(data=fraud_clean, x=col, ax=ax, order=order, palette="muted")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Univariate – Categorical Features", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "univariate_categorical.png"))
plt.show()

# %% [markdown]
# ## 6. Bivariate Analysis (Feature vs. Fraud Target)

# %%
# Purchase value by class
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=fraud_clean,
    x="class",
    y="purchase_value",
    palette={0: "steelblue", 1: "tomato"},
    ax=ax,
)
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
ax.set_title("Purchase Value by Class")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "purchase_value_by_class.png"))
plt.show()

# %%
# Age by class
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(
    data=fraud_clean,
    x="age",
    hue="class",
    fill=True,
    palette={0: "steelblue", 1: "tomato"},
    ax=ax,
)
ax.set_title("Age Distribution by Class")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "age_by_class.png"))
plt.show()

# %%
# Fraud rate by source and browser
for col in ["source", "browser", "sex"]:
    fraud_rate = (
        fraud_clean.groupby(col)["class"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fraud_rate.columns = [col, "fraud_rate"]

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(
        data=fraud_rate,
        x=col,
        y="fraud_rate",
        palette="muted",
        ax=ax,
    )
    ax.set_title(f"Fraud Rate by {col}")
    ax.set_ylabel("Fraud Rate")
    ax.set_ylim(0, fraud_rate["fraud_rate"].max() * 1.3)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2%}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f"fraud_rate_{col}.png"))
    plt.show()

# %% [markdown]
# ## 7. Geolocation – IP to Country Merge

# %%
fraud_geo = merge_ip_country(fraud_clean, ip_country)
print("Shape after geo merge:", fraud_geo.shape)
print("Sample country values:")
print(fraud_geo["country"].value_counts().head(10))

# %%
# Top 15 countries by fraud count
fraud_by_country = (
    fraud_geo[fraud_geo["class"] == 1]["country"]
    .value_counts()
    .nlargest(15)
    .reset_index()
)
fraud_by_country.columns = ["country", "fraud_count"]

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=fraud_by_country,
    x="fraud_count",
    y="country",
    palette="Reds_r",
    ax=ax,
)
ax.set_title("Top 15 Countries by Fraud Count")
ax.set_xlabel("Number of Fraudulent Transactions")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_by_country.png"))
plt.show()

# %%
# Fraud rate per country (countries with ≥100 transactions)
country_stats = (
    fraud_geo.groupby("country")["class"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "fraud", "count": "total"})
)
country_stats["fraud_rate"] = country_stats["fraud"] / country_stats["total"]
top_rate = (
    country_stats[country_stats["total"] >= 100]
    .sort_values("fraud_rate", ascending=False)
    .head(15)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=top_rate,
    x="fraud_rate",
    y="country",
    palette="Oranges_r",
    ax=ax,
)
ax.set_title("Top 15 Countries by Fraud Rate (≥100 transactions)")
ax.set_xlabel("Fraud Rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_rate_by_country.png"))
plt.show()

# %% [markdown]
# ## 8. Feature Engineering

# %%
fraud_featured = build_fraud_features(fraud_geo)
fraud_featured = add_country_dummies(fraud_featured, top_n=10)

print("Columns after feature engineering:")
print(fraud_featured.columns.tolist())

# %%
# Visualise time_since_signup by class
fig, ax = plt.subplots(figsize=(8, 4))
# Cap at 99th percentile to reduce outlier distortion
cap = fraud_featured["time_since_signup"].quantile(0.99)
plot_df = fraud_featured[fraud_featured["time_since_signup"] <= cap]
sns.kdeplot(
    data=plot_df,
    x="time_since_signup",
    hue="class",
    fill=True,
    palette={0: "steelblue", 1: "tomato"},
    ax=ax,
)
ax.set_title("Time Since Signup by Class (capped at 99th pct)")
ax.set_xlabel("Seconds since signup")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "time_since_signup_by_class.png"))
plt.show()

# %%
# Hour of day fraud heatmap
hour_fraud = (
    fraud_featured.groupby("hour_of_day")["class"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "fraud", "count": "total"})
)
hour_fraud["fraud_rate"] = hour_fraud["fraud"] / hour_fraud["total"]

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(hour_fraud.index, hour_fraud["fraud_rate"], color="tomato", edgecolor="white")
ax.set_title("Fraud Rate by Hour of Day")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud Rate")
ax.set_xticks(range(24))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_rate_by_hour.png"))
plt.show()

# %% [markdown]
# ## 9. Train/Test Split & Resampling

# %%
feature_cols = get_feature_columns(fraud_featured, target="class")
X = fraud_featured[feature_cols]
y = fraud_featured["class"]

print("Feature matrix shape:", X.shape)
print("Features:", feature_cols)

# Stratified split — preserves fraud rate in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

# %%
# Scale numeric features (fit on train only)
X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train, X_test, cols=NUMERIC_COLS_FRAUD, method="standard"
)

# %%
# Class balance before resampling
report_class_balance(y_train, label="BEFORE SMOTE")

# Apply SMOTE on training set only
X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

# %% [markdown]
# ## 10. Save Processed Data

# %%
# Save the full feature-engineered dataset (pre-split) for reproducibility
fraud_featured.to_csv(
    os.path.join(PROCESSED_DIR, "fraud_data_featured.csv"), index=False
)

# Save train/test splits (scaled + resampled train)
X_train_res.to_csv(
    os.path.join(PROCESSED_DIR, "X_train_fraud.csv"), index=False
)
y_train_res.to_csv(
    os.path.join(PROCESSED_DIR, "y_train_fraud.csv"), index=False
)
X_test_scaled.to_csv(
    os.path.join(PROCESSED_DIR, "X_test_fraud.csv"), index=False
)
y_test.to_csv(
    os.path.join(PROCESSED_DIR, "y_test_fraud.csv"), index=False
)

print("All processed files saved to:", PROCESSED_DIR)
