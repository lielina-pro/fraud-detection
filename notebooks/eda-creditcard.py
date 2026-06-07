# %% [markdown]
# # EDA – creditcard.csv
# **Project:** Improved Detection of Fraud Cases – Bank Credit Card Transactions
# **Week 5 & 6 | 10 Academy KAIM**

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

sys.path.insert(0, os.path.abspath(".."))

from src.data_preprocessing import clean_creditcard, load_creditcard, report_missing  # noqa: E402
from src.feature_engineering import scale_features, NUMERIC_COLS_CREDIT  # noqa: E402
from src.imbalance_handler import apply_undersample, report_class_balance  # noqa: E402

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100

# %%
# ---------------------------------------------------------------------------
# EDIT THIS PATH to point to your creditcard file
# ---------------------------------------------------------------------------
CREDIT_PATH = "../data/raw/creditcard.csv"          # or creditcard.csv.zip
PROCESSED_DIR = "../data/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Data

# %%
credit_raw = load_creditcard(CREDIT_PATH)
print("creditcard shape:", credit_raw.shape)
credit_raw.head()

# %% [markdown]
# ## 2. Initial Inspection

# %%
print("=== Data Types ===")
print(credit_raw.dtypes)
print("\n=== Basic Statistics ===")
credit_raw.describe()

# %%
report_missing(credit_raw, name="creditcard")

# %% [markdown]
# ## 3. Data Cleaning

# %%
credit_clean = clean_creditcard(credit_raw)
print("Shape after cleaning:", credit_clean.shape)

# %% [markdown]
# ## 4. Class Imbalance

# %%
report_class_balance(credit_clean["Class"], label="creditcard raw")

fig, ax = plt.subplots(figsize=(6, 4))
counts = credit_clean["Class"].value_counts()
ax.bar(
    ["Legitimate (0)", "Fraud (1)"],
    counts.values,
    color=["steelblue", "tomato"],
    edgecolor="white",
)
ax.set_title("Class Distribution – creditcard.csv", fontsize=13)
ax.set_ylabel("Number of Transactions")
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "class_distribution_credit.png"))
plt.show()

# %% [markdown]
# ## 5. Univariate Analysis

# %%
# Distribution of Amount
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(credit_clean["Amount"], ax=axes[0], bins=60, kde=True, color="steelblue")
axes[0].set_title("Distribution of Amount")

# Log-transform Amount for better visualisation
log_amount = np.log1p(credit_clean["Amount"])
sns.histplot(log_amount, ax=axes[1], bins=60, kde=True, color="steelblue")
axes[1].set_title("Distribution of log(Amount + 1)")
plt.suptitle("Transaction Amount", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "amount_distribution.png"))
plt.show()

# %%
# Distribution of Time
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(credit_clean["Time"], ax=ax, bins=80, kde=True, color="slateblue")
ax.set_title("Distribution of Time (seconds since first transaction)")
ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "time_distribution.png"))
plt.show()

# %% [markdown]
# ## 6. Bivariate Analysis

# %%
# Amount by class
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(
    data=credit_clean,
    x="Class",
    y="Amount",
    palette={0: "steelblue", 1: "tomato"},
    ax=ax,
)
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
ax.set_title("Transaction Amount by Class")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "amount_by_class.png"))
plt.show()

# %%
# Correlation heatmap – all V features + Amount + Class
fig, ax = plt.subplots(figsize=(14, 10))
corr = credit_clean.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    linewidths=0.3,
    ax=ax,
    annot=False,
)
ax.set_title("Correlation Matrix – creditcard.csv")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "credit_correlation.png"))
plt.show()

# %%
# Top correlated features with Class
corr_with_target = (
    corr["Class"]
    .drop("Class")
    .abs()
    .sort_values(ascending=False)
    .head(10)
)
print("Top 10 features correlated with Class:")
print(corr_with_target)

fig, ax = plt.subplots(figsize=(8, 4))
corr_with_target.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 10 Features Correlated with Fraud (Class)")
ax.set_ylabel("|Correlation|")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "top_correlations_credit.png"))
plt.show()

# %% [markdown]
# ## 7. Train/Test Split & Resampling

# %%
X = credit_clean.drop(columns=["Class"])
y = credit_clean["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# %%
# Scale Time and Amount (V1-V28 are already PCA-scaled)
X_train_scaled, X_test_scaled, scaler_credit = scale_features(
    X_train, X_test, cols=NUMERIC_COLS_CREDIT, method="standard"
)

# %%
# Class balance before resampling
report_class_balance(y_train, label="BEFORE undersample")

# Use undersampling for creditcard (284k rows – still huge after sampling)
X_train_res, y_train_res = apply_undersample(
    X_train_scaled, y_train, sampling_strategy=0.5
)

# %% [markdown]
# ## 8. Save Processed Data

# %%
X_train_res.to_csv(
    os.path.join(PROCESSED_DIR, "X_train_credit.csv"), index=False
)
y_train_res.to_csv(
    os.path.join(PROCESSED_DIR, "y_train_credit.csv"), index=False
)
X_test_scaled.to_csv(
    os.path.join(PROCESSED_DIR, "X_test_credit.csv"), index=False
)
y_test.to_csv(
    os.path.join(PROCESSED_DIR, "y_test_credit.csv"), index=False
)

print("All processed files saved to:", PROCESSED_DIR)
