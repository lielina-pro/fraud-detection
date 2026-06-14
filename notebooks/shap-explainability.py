# %% [markdown]
# # Task 3 – Model Explainability with SHAP
# **Project:** Improved Detection of Fraud Cases | 10 Academy KAIM Week 5 & 6
#
# This notebook:
# 1. Loads trained models from Task 2
# 2. Computes SHAP values for Fraud_Data (LightGBM) and creditcard (XGBoost)
# 3. Produces SHAP summary plots and bar importance plots
# 4. Explains individual predictions: True Positive, False Positive, False Negative
# 5. Generates SHAP dependence plots for top features
# 6. Identifies the top 5 fraud drivers
# 7. Delivers 3+ actionable business recommendations

# %% [markdown]
# ## 0. Imports

# %%
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.abspath(".."))
warnings.filterwarnings("ignore")

from src.explainability import (  # noqa: E402
    compute_shap_values,
    mean_absolute_shap,
    plot_dependence,
    plot_force_matplotlib,
    plot_shap_bar,
    plot_shap_summary,
    top_fraud_drivers,
)
from src.data_preprocessing import (  # noqa: E402
    clean_fraud_data,
    load_fraud_data,
    load_ip_country,
    merge_ip_country,
)
from src.feature_engineering import (  # noqa: E402
    add_country_dummies,
    add_device_count,
    add_time_features,
    add_velocity_features,
    encode_categoricals,
    get_feature_columns,
    scale_features,
    FRAUD_NUM_COLS,
)
from src.model_trainer import (  # noqa: E402
    get_lightgbm,
    get_xgboost,
    load_model,
    resample_smote,
    resample_undersample,
    scale_numeric,
    stratified_split,
    CREDIT_NUM_COLS,
)

plt.rcParams["figure.dpi"] = 100

MODELS_DIR = "../models/"
PROCESSED_DIR = "../data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# %% [markdown]
# ---
# # PART A — Fraud_Data.csv (LightGBM)

# %% [markdown]
# ## 1. Load Data and Retrain Best Model

# %%
# Load and preprocess
fraud_raw = load_fraud_data("../data/raw/Fraud_Data.csv")
ip_country = load_ip_country("../data/raw/IpAddress_to_Country.csv")

fraud_clean = clean_fraud_data(fraud_raw)
fraud_geo = merge_ip_country(fraud_clean, ip_country)
fraud_featured = add_time_features(fraud_geo)
fraud_featured = add_velocity_features(fraud_featured)
fraud_featured = add_device_count(fraud_featured)
fraud_featured = add_country_dummies(fraud_featured, top_n=10)
fraud_featured = encode_categoricals(fraud_featured)

feature_cols = get_feature_columns(fraud_featured, target="class")
X = fraud_featured[feature_cols]
y = fraud_featured["class"]

X_train, X_test, y_train, y_test = stratified_split(X, y)
X_train_s, X_test_s, scaler = scale_numeric(X_train, X_test, FRAUD_NUM_COLS)
X_train_res, y_train_res = resample_smote(X_train_s, y_train)

lgb_model = get_lightgbm()
lgb_model.fit(X_train_res, y_train_res)
print("LightGBM trained on Fraud_Data.")
print(f"Test set: {len(X_test_s):,} samples  |  Fraud: {y_test.sum():,}")

# %% [markdown]
# ## 2. Compute SHAP Values

# %%
# Sample 2000 test examples for speed
X_shap = X_test_s.sample(2000, random_state=42)
y_shap = y_test.loc[X_shap.index]

print("Computing SHAP values (TreeExplainer)...")
shap_vals, explainer = compute_shap_values(lgb_model, X_shap)
print(f"SHAP values shape: {shap_vals.shape}")

# %% [markdown]
# ## 3. Global Feature Importance — Summary Plot

# %%
# Beeswarm: every dot = one sample, color = feature value
plot_shap_summary(
    shap_vals, X_shap,
    title="SHAP Summary – Fraud_Data (LightGBM)",
    max_display=15,
    save_path=os.path.join(PROCESSED_DIR, "shap_summary_fraud.png"),
)

# %%
# Bar chart: cleaner view for reporting
plot_shap_bar(
    shap_vals, feature_cols,
    title="Top 10 Fraud Drivers – Fraud_Data (LightGBM)",
    top_n=10,
    color="steelblue",
    save_path=os.path.join(PROCESSED_DIR, "shap_bar_fraud.png"),
)

# %% [markdown]
# ## 4. Top 5 Fraud Drivers — Fraud_Data

# %%
drivers_fraud = top_fraud_drivers(shap_vals, feature_cols, top_n=5)
print("Top 5 Fraud Drivers – Fraud_Data:")
print(drivers_fraud.to_string())

# %% [markdown]
# ## 5. Individual Explanations — Force Plots

# %%
# Identify TP, FP, FN examples
y_pred_shap = lgb_model.predict(X_shap)
y_prob_shap = lgb_model.predict_proba(X_shap)[:, 1]

results_df = pd.DataFrame({
    "y_true": y_shap.values,
    "y_pred": y_pred_shap,
    "y_prob": y_prob_shap,
}, index=X_shap.index)

tp_idx = (results_df
          .loc[(results_df.y_true == 1) & (results_df.y_pred == 1)]
          .sort_values("y_prob", ascending=False)
          .index[0])
fp_idx = (results_df
          .loc[(results_df.y_true == 0) & (results_df.y_pred == 1)]
          .sort_values("y_prob", ascending=False)
          .index[0])
fn_idx = (results_df
          .loc[(results_df.y_true == 1) & (results_df.y_pred == 0)]
          .sort_values("y_prob")
          .index[0])

sample_list = list(X_shap.index)
tp_pos = sample_list.index(tp_idx)
fp_pos = sample_list.index(fp_idx)
fn_pos = sample_list.index(fn_idx)

tp_prob = results_df.loc[tp_idx, "y_prob"]
fp_prob = results_df.loc[fp_idx, "y_prob"]
fn_prob = results_df.loc[fn_idx, "y_prob"]

print(f"TP example — fraud probability: {tp_prob:.4f}")
print(f"FP example — fraud probability: {fp_prob:.4f}")
print(f"FN example — fraud probability: {fn_prob:.4f}")

# %%
# TRUE POSITIVE — correctly flagged fraud
plot_force_matplotlib(
    explainer, shap_vals, X_shap, tp_pos,
    title=f"TRUE POSITIVE: Correct Fraud Alert  (P(fraud)={tp_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_tp_fraud.png"),
)

# %%
# FALSE POSITIVE — legitimate transaction wrongly flagged
plot_force_matplotlib(
    explainer, shap_vals, X_shap, fp_pos,
    title=f"FALSE POSITIVE: Wrongly Flagged Legitimate  (P(fraud)={fp_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_fp_fraud.png"),
)

# %%
# FALSE NEGATIVE — missed fraud
plot_force_matplotlib(
    explainer, shap_vals, X_shap, fn_pos,
    title=f"FALSE NEGATIVE: Missed Fraud  (P(fraud)={fn_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_fn_fraud.png"),
)

# %% [markdown]
# ## 6. Dependence Plots — Top Features

# %%
# time_since_signup vs hour_of_day interaction
plot_dependence(
    shap_vals, X_shap,
    feature="time_since_signup",
    interaction_feature="hour_of_day",
    title="SHAP Dependence: time_since_signup (colored by hour_of_day)",
    save_path=os.path.join(PROCESSED_DIR, "dependence_time_signup.png"),
)

# %%
# device_user_count dependence
plot_dependence(
    shap_vals, X_shap,
    feature="device_user_count",
    interaction_feature="auto",
    title="SHAP Dependence: device_user_count",
    save_path=os.path.join(PROCESSED_DIR, "dependence_device_count.png"),
)

# %% [markdown]
# ---
# # PART B — creditcard.csv (XGBoost)

# %% [markdown]
# ## 7. Load and Retrain creditcard Model

# %%
import pandas as pd  # noqa: F811

credit = pd.read_csv("../data/raw/creditcard.csv")
credit = credit.drop_duplicates()
credit["Class"] = credit["Class"].astype(int)

X_c = credit.drop(columns=["Class"])
y_c = credit["Class"]

X_train_c, X_test_c, y_train_c, y_test_c = stratified_split(X_c, y_c)
X_train_cs, X_test_cs, scaler_c = scale_numeric(X_train_c, X_test_c, CREDIT_NUM_COLS)
X_train_cr, y_train_cr = resample_undersample(X_train_cs, y_train_c)

xgb_model = get_xgboost()
xgb_model.fit(X_train_cr, y_train_cr)
print("XGBoost trained on creditcard.")
print(f"Test set: {len(X_test_cs):,} samples  |  Fraud: {y_test_c.sum():,}")

# %% [markdown]
# ## 8. Compute SHAP Values — creditcard

# %%
X_shap_c = X_test_cs.sample(2000, random_state=42)
y_shap_c = y_test_c.loc[X_shap_c.index]

print("Computing SHAP values (TreeExplainer - XGBoost)...")
shap_vals_c, explainer_c = compute_shap_values(xgb_model, X_shap_c)
print(f"SHAP values shape: {shap_vals_c.shape}")

# %%
plot_shap_summary(
    shap_vals_c, X_shap_c,
    title="SHAP Summary – creditcard (XGBoost)",
    max_display=15,
    save_path=os.path.join(PROCESSED_DIR, "shap_summary_credit.png"),
)

# %%
plot_shap_bar(
    shap_vals_c, list(X_c.columns),
    title="Top 10 Fraud Drivers – creditcard (XGBoost)",
    top_n=10,
    color="tomato",
    save_path=os.path.join(PROCESSED_DIR, "shap_bar_credit.png"),
)

# %% [markdown]
# ## 9. Top 5 Fraud Drivers — creditcard

# %%
drivers_credit = top_fraud_drivers(shap_vals_c, list(X_c.columns), top_n=5)
print("Top 5 Fraud Drivers – creditcard:")
print(drivers_credit.to_string())

# %% [markdown]
# ## 10. Individual Explanations — creditcard

# %%
y_pred_c = xgb_model.predict(X_shap_c)
y_prob_c = xgb_model.predict_proba(X_shap_c)[:, 1]

results_c = pd.DataFrame({
    "y_true": y_shap_c.values,
    "y_pred": y_pred_c,
    "y_prob": y_prob_c,
}, index=X_shap_c.index)

tp_c = (results_c
        .loc[(results_c.y_true == 1) & (results_c.y_pred == 1)]
        .sort_values("y_prob", ascending=False)
        .index[0])
fp_c = (results_c
        .loc[(results_c.y_true == 0) & (results_c.y_pred == 1)]
        .sort_values("y_prob", ascending=False)
        .index[0])
fn_c = (results_c
        .loc[(results_c.y_true == 1) & (results_c.y_pred == 0)]
        .sort_values("y_prob")
        .index[0])

sample_list_c = list(X_shap_c.index)

tp_c_prob = results_c.loc[tp_c, "y_prob"]
fp_c_prob = results_c.loc[fp_c, "y_prob"]
fn_c_prob = results_c.loc[fn_c, "y_prob"]

plot_force_matplotlib(
    explainer_c, shap_vals_c, X_shap_c,
    sample_list_c.index(tp_c),
    title=f"TRUE POSITIVE – creditcard  (P(fraud)={tp_c_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_tp_credit.png"),
)

plot_force_matplotlib(
    explainer_c, shap_vals_c, X_shap_c,
    sample_list_c.index(fp_c),
    title=f"FALSE POSITIVE – creditcard  (P(fraud)={fp_c_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_fp_credit.png"),
)

plot_force_matplotlib(
    explainer_c, shap_vals_c, X_shap_c,
    sample_list_c.index(fn_c),
    title=f"FALSE NEGATIVE – creditcard  (P(fraud)={fn_c_prob:.4f})",
    save_path=os.path.join(PROCESSED_DIR, "force_fn_credit.png"),
)

# %% [markdown]
# ---
# # PART C — Business Recommendations

# %% [markdown]
# ## 11. Consolidated Top 5 Fraud Drivers

# %%
print("=" * 60)
print("FRAUD_DATA — Top 5 Fraud Drivers (LightGBM + SHAP)")
print("=" * 60)
print(drivers_fraud.to_string())

print("\n")
print("=" * 60)
print("CREDITCARD — Top 5 Fraud Drivers (XGBoost + SHAP)")
print("=" * 60)
print(drivers_credit.to_string())

# %% [markdown]
# ## 12. Actionable Business Recommendations
#
# Based on SHAP analysis of both models, the following recommendations
# are derived directly from the top fraud drivers:
#
# ---
#
# ### Recommendation 1 — Immediate Block on Near-Zero time_since_signup
# **Driver:** `time_since_signup` (Mean |SHAP| = 2.12 — by far the strongest signal)
#
# **Finding:** 99.52% of transactions occurring within 1 hour of account creation
# are fraudulent. The SHAP dependence plot shows a sharp cliff: values below
# ~3,600 seconds produce extreme positive SHAP values.
#
# **Action:** Apply a hard rule: any transaction within 60 minutes of signup
# triggers mandatory step-up authentication (OTP / 3D Secure). Transactions
# within 5 minutes should be auto-held for manual review. This rule alone
# would intercept the majority of fraud cases at near-zero cost.
#
# ---
#
# ### Recommendation 2 — Device Fingerprint Sharing Monitoring
# **Driver:** `device_user_count` (Mean |SHAP| = 1.49 — second strongest)
#
# **Finding:** Devices linked to more than one user account carry a 52.5% fraud
# rate vs 3.0% for single-user devices. The SHAP value increases sharply once
# device_user_count exceeds 2.
#
# **Action:** Flag any device associated with ≥2 user accounts for enhanced
# review on every subsequent transaction. Devices with ≥5 accounts should be
# temporarily blocked pending identity verification. Implement device
# fingerprinting (browser, OS, screen resolution, timezone) to improve
# device identity robustness beyond IP-based tracking.
#
# ---
#
# ### Recommendation 3 — Geographic Risk Scoring
# **Driver:** `country` features (Norway 13.0%, Mexico 12.8%, Sweden 12.0%
# fraud rates from EDA; confirmed by positive SHAP for these country dummies)
#
# **Finding:** Country of origin contributes meaningfully to fraud probability.
# 14.5% of IPs could not be mapped to any country — unresolved IPs themselves
# correlate with elevated fraud risk (anonymised VPNs, Tor exit nodes).
#
# **Action:** Introduce a real-time geographic risk score: tier countries by
# historical fraud rate. Transactions from high-risk countries (rate > 10%)
# or from unresolvable IPs should trigger friction (CAPTCHA, OTP). Consider
# integrating a commercial IP intelligence feed to reduce the 14.5% unmapped
# rate.
#
# ---
#
# ### Recommendation 4 — Real-Time Velocity Monitoring
# **Driver:** `user_tx_count`, `user_tx_per_day` (combined SHAP contribution
# in top 10 for Fraud_Data)
#
# **Finding:** High transaction velocity per user is a consistent fraud signal.
# Fraudsters exploit stolen credentials to transact rapidly before detection.
#
# **Action:** Implement a sliding-window velocity counter (transactions per
# user in the last 1 hour, 6 hours, 24 hours). Trigger review if a user
# exceeds 3 transactions within 1 hour or 10 within 24 hours. This pairs
# well with Recommendation 1 — new accounts with high velocity are almost
# certainly fraudulent.
#
# ---
#
# ### Recommendation 5 — PCA Feature Monitoring for Bank Transactions
# **Driver:** `V14`, `V4`, `V12`, `V10` (top SHAP features in creditcard)
#
# **Finding:** Even though V-features are anonymised, V14 alone (Mean |SHAP|
# = 1.79) dominates the XGBoost model's fraud decisions. The SHAP summary
# shows V14 values below zero strongly predict fraud.
#
# **Action:** Work with the bank's data engineering team to identify which
# original transaction attributes map to V14 and V4 post-PCA. Re-training
# on raw (non-anonymised) features would likely improve AUC-PR beyond 0.75
# and enable interpretable rules for compliance reporting. Additionally,
# monitor model drift on V14 and V4 distributions monthly, as these are the
# highest-leverage features and concept drift here would degrade the model
# most rapidly.
