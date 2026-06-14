# %% [markdown]
# # Task 2 – Model Building and Training
# **Project:** Improved Detection of Fraud Cases | 10 Academy KAIM Week 5 & 6
#
# This notebook:
# 1. Prepares data (split, scale, resample) for both datasets
# 2. Trains Logistic Regression (baseline) + Random Forest, XGBoost, LightGBM
# 3. Evaluates using F1, AUC-PR, ROC-AUC, and Confusion Matrix
# 4. Runs Stratified K-Fold (k=5) cross-validation on the best model
# 5. Compares all models and selects the best with justification

# %% [markdown]
# ## 0. Imports

# %%
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.abspath(".."))
warnings.filterwarnings("ignore")

from src.model_trainer import (  # noqa: E402
    CREDIT_NUM_COLS,
    FRAUD_NUM_COLS,
    compare_models,
    cross_validate_fraud,
    evaluate_model,
    get_lightgbm,
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
    load_model,
    resample_smote,
    resample_undersample,
    save_model,
    scale_numeric,
    stratified_split,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100

PROCESSED_DIR = "../data/processed/"
MODELS_DIR = "../models/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# %% [markdown]
# ---
# # PART A — Fraud_Data.csv (E-commerce)

# %% [markdown]
# ## 1. Load Feature-Engineered Data

# %%
fraud = pd.read_csv(os.path.join(PROCESSED_DIR, "fraud_data_featured.csv"))
feature_cols = [c for c in fraud.columns if c not in [
    "class", "user_id", "device_id", "ip_address", "signup_time",
    "purchase_time", "ip_int", "lower_bound_ip_address",
    "upper_bound_ip_address", "country",
]]
X_fraud = fraud[feature_cols]
y_fraud = fraud["class"]

print(f"Features: {len(feature_cols)}")
print(f"Shape: {X_fraud.shape}")
print(f"Fraud rate: {y_fraud.mean():.4f} ({y_fraud.sum():,} fraud / {len(y_fraud):,} total)")

# %% [markdown]
# ## 2. Stratified Train/Test Split

# %%
X_train, X_test, y_train, y_test = stratified_split(X_fraud, y_fraud)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate:  {y_test.mean():.4f}")

# %% [markdown]
# ## 3. Scale and Resample (SMOTE — training only)

# %%
X_train_s, X_test_s, scaler_fraud = scale_numeric(X_train, X_test, FRAUD_NUM_COLS)

print("Before SMOTE:", y_train.value_counts().to_dict())
X_train_res, y_train_res = resample_smote(X_train_s, y_train)
print("After SMOTE: ", pd.Series(y_train_res).value_counts().to_dict())

# %% [markdown]
# ## 4. Train and Evaluate All Models

# %%
fraud_results = []

# 4a. Logistic Regression — baseline
lr = get_logistic_regression()
fraud_results.append(evaluate_model(
    "Logistic Regression", lr,
    X_train_res, y_train_res, X_test_s, y_test,
))

# %%
# 4b. Random Forest
rf = get_random_forest()
fraud_results.append(evaluate_model(
    "Random Forest", rf,
    X_train_res, y_train_res, X_test_s, y_test,
))

# %%
# 4c. XGBoost
xgb_model = get_xgboost()
fraud_results.append(evaluate_model(
    "XGBoost", xgb_model,
    X_train_res, y_train_res, X_test_s, y_test,
))

# %%
# 4d. LightGBM
lgb_model = get_lightgbm()
fraud_results.append(evaluate_model(
    "LightGBM", lgb_model,
    X_train_res, y_train_res, X_test_s, y_test,
))

# %% [markdown]
# ## 5. Model Comparison – Fraud_Data

# %%
fraud_summary = compare_models(fraud_results)
print(fraud_summary.to_string(index=False))

# %%
# Bar chart: F1 and AUC-PR side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
models = fraud_summary["Model"]
x = np.arange(len(models))
w = 0.35

for ax, metric, title in zip(
    axes,
    ["F1", "AUC-PR"],
    ["F1-Score by Model", "AUC-PR by Model"],
):
    bars = ax.bar(x, fraud_summary[metric], color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

plt.suptitle("Fraud_Data – Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_model_comparison.png"))
plt.show()

# %% [markdown]
# ## 6. Confusion Matrices – Fraud_Data

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, r in zip(axes, fraud_results):
    cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
    )
    ax.set_title(r["name"], fontsize=10)
plt.suptitle("Confusion Matrices – Fraud_Data", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_confusion_matrices.png"))
plt.show()

# %% [markdown]
# ## 7. Stratified K-Fold Cross-Validation – LightGBM

# %%
print("Running 5-Fold CV on LightGBM (Fraud_Data)...")
cv_results = cross_validate_fraud(X_fraud, y_fraud, model_fn=get_lightgbm)

print(f"\nCV F1:     {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
print(f"CV AUC-PR: {cv_results['pr_mean']:.4f} ± {cv_results['pr_std']:.4f}")

# %%
# Visualise CV scores
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
folds = [f"Fold {i+1}" for i in range(5)]
for ax, scores, metric in zip(
    axes,
    [cv_results["f1_scores"], cv_results["pr_scores"]],
    ["F1-Score", "AUC-PR"],
):
    ax.bar(folds, scores, color="steelblue", edgecolor="white")
    ax.axhline(np.mean(scores), color="tomato", linestyle="--",
               label=f"Mean={np.mean(scores):.4f}")
    ax.set_title(f"5-Fold CV {metric} – LightGBM")
    ax.set_ylabel(metric)
    ax.set_ylim(0.5, 0.9)
    ax.legend()
plt.suptitle("Cross-Validation – LightGBM (Fraud_Data)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "fraud_cv_scores.png"))
plt.show()

# %% [markdown]
# ## 8. Save Best Model – Fraud_Data

# %%
# Best model by AUC-PR
best_fraud = sorted(fraud_results, key=lambda x: x["auc_pr"], reverse=True)[0]
print(f"Best model: {best_fraud['name']} (AUC-PR={best_fraud['auc_pr']})")

save_model(
    path=os.path.join(MODELS_DIR, "best_fraud_model.pkl"),
    model=best_fraud["model"],
    scaler=scaler_fraud,
    feature_cols=feature_cols,
    name=best_fraud["name"],
)

# %% [markdown]
# ---
# # PART B — creditcard.csv (Bank Transactions)

# %% [markdown]
# ## 9. Load and Prepare creditcard Data

# %%
credit = pd.read_csv("../data/raw/creditcard.csv")
credit = credit.drop_duplicates()
credit["Class"] = credit["Class"].astype(int)

print(f"Shape: {credit.shape}")
print(f"Fraud rate: {credit['Class'].mean():.4f}")
print(f"Fraud count: {credit['Class'].sum():,} / {len(credit):,}")

X_credit = credit.drop(columns=["Class"])
y_credit = credit["Class"]

X_train_c, X_test_c, y_train_c, y_test_c = stratified_split(X_credit, y_credit)
print(f"\nTrain: {len(X_train_c):,}  |  Test: {len(X_test_c):,}")

# %%
X_train_cs, X_test_cs, scaler_credit = scale_numeric(
    X_train_c, X_test_c, CREDIT_NUM_COLS
)

print("Before undersampling:", y_train_c.value_counts().to_dict())
X_train_cr, y_train_cr = resample_undersample(X_train_cs, y_train_c)
print("After undersampling: ", pd.Series(y_train_cr).value_counts().to_dict())

# %% [markdown]
# ## 10. Train and Evaluate – creditcard

# %%
credit_results = []

lr_c = get_logistic_regression()
credit_results.append(evaluate_model(
    "Logistic Regression", lr_c,
    X_train_cr, y_train_cr, X_test_cs, y_test_c,
))

# %%
rf_c = get_random_forest()
credit_results.append(evaluate_model(
    "Random Forest", rf_c,
    X_train_cr, y_train_cr, X_test_cs, y_test_c,
))

# %%
xgb_c = get_xgboost()
credit_results.append(evaluate_model(
    "XGBoost", xgb_c,
    X_train_cr, y_train_cr, X_test_cs, y_test_c,
))

# %%
lgb_c = get_lightgbm()
credit_results.append(evaluate_model(
    "LightGBM", lgb_c,
    X_train_cr, y_train_cr, X_test_cs, y_test_c,
))

# %% [markdown]
# ## 11. Model Comparison – creditcard

# %%
credit_summary = compare_models(credit_results)
print(credit_summary.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
models_c = credit_summary["Model"]
x_c = np.arange(len(models_c))

for ax, metric, title in zip(
    axes,
    ["F1", "AUC-PR"],
    ["F1-Score by Model", "AUC-PR by Model"],
):
    bars = ax.bar(x_c, credit_summary[metric], color="tomato", edgecolor="white")
    ax.set_xticks(x_c)
    ax.set_xticklabels(models_c, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

plt.suptitle("creditcard – Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "credit_model_comparison.png"))
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, r in zip(axes, credit_results):
    cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
    )
    ax.set_title(r["name"], fontsize=10)
plt.suptitle("Confusion Matrices – creditcard", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "credit_confusion_matrices.png"))
plt.show()

# %% [markdown]
# ## 12. Save Best Model – creditcard

# %%
best_credit = sorted(credit_results, key=lambda x: x["auc_pr"], reverse=True)[0]
print(f"Best model: {best_credit['name']} (AUC-PR={best_credit['auc_pr']})")

save_model(
    path=os.path.join(MODELS_DIR, "best_credit_model.pkl"),
    model=best_credit["model"],
    scaler=scaler_credit,
    feature_cols=list(X_credit.columns),
    name=best_credit["name"],
)

# %% [markdown]
# ---
# # PART C — Final Model Selection & Justification

# %%
print("=" * 60)
print("FRAUD_DATA – Best models by AUC-PR:")
print(fraud_summary[["Model", "F1", "AUC-PR", "ROC-AUC"]].to_string(index=False))

print("\nCREDITCARD – Best models by AUC-PR:")
print(credit_summary[["Model", "F1", "AUC-PR", "ROC-AUC"]].to_string(index=False))

print("\n" + "=" * 60)
print("SELECTION RATIONALE")
print("-" * 60)
print(f"Fraud_Data → {best_fraud['name']}")
print("  AUC-PR is the primary metric for imbalanced data.")
print("  LightGBM and Random Forest lead; LightGBM is faster")
print("  and confirmed by 5-fold CV (AUC-PR 0.7162 ± 0.0073).")
print()
print(f"creditcard → {best_credit['name']}")
print("  XGBoost achieves highest AUC-PR (0.7545).")
print("  Its ranking-based objective is well-suited to the")
print("  extreme 0.17% fraud rate in this dataset.")
