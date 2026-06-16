"""
scripts/train_models.py

Task 2 — Model Building and Evaluation (self-contained, end-to-end)

This script:
  1. Loads raw data from data/raw/
  2. Runs the full preprocessing and feature engineering pipeline
  3. Performs stratified train/test split on both datasets
  4. Trains Logistic Regression (baseline), Random Forest, XGBoost, LightGBM
  5. Evaluates every model on AUC-PR, F1-Score, ROC-AUC, and Confusion Matrix
  6. Runs 5-fold stratified cross-validation on the best model
  7. Saves trained models to models/
  8. Saves a metrics comparison table to models/results_summary.csv

Usage (from project root):
    python scripts/train_models.py

Requirements:
    pip install -r requirements.txt
    Place raw data in data/raw/ before running.

All functions are flake8-compliant (max line length 88).
"""

import os
import pickle
import warnings

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

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FRAUD_PATH = os.path.join(RAW, "Fraud_Data.csv")
IP_PATH = os.path.join(RAW, "IpAddress_to_Country.csv")
CREDIT_PATH = os.path.join(RAW, "creditcard.csv")

# ── Numeric columns for scaling ────────────────────────────────────────────────
FRAUD_NUM_COLS = [
    "purchase_value", "age", "time_since_signup",
    "hour_of_day", "day_of_week", "user_tx_count",
    "user_tx_per_day", "device_user_count",
]
CREDIT_NUM_COLS = ["Time", "Amount"]


# ── Utility ───────────────────────────────────────────────────────────────────

def section(title):
    """Print a visible section header."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """
    Fit model on training data, evaluate on test data.

    Returns a dict with name, f1, auc_pr, roc_auc,
    tp, fp, fn, tn, and the fitted model.
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    f1 = f1_score(y_te, y_pred)
    auc_pr = average_precision_score(y_te, y_prob)
    roc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  {name}")
    print(f"    F1={f1:.4f}  AUC-PR={auc_pr:.4f}  ROC-AUC={roc:.4f}")
    print(f"    Confusion Matrix: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(classification_report(
        y_te, y_pred, target_names=["Legit", "Fraud"], zero_division=0
    ))

    return {
        "name": name,
        "f1": round(f1, 4),
        "auc_pr": round(auc_pr, 4),
        "roc_auc": round(roc, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "model": model,
    }


# ── PART A: Fraud_Data ────────────────────────────────────────────────────────

section("PART A — Fraud_Data.csv (E-commerce)")

# 1. Load
print("\n[1] Loading Fraud_Data.csv and IpAddress_to_Country.csv ...")
fraud = pd.read_csv(FRAUD_PATH)
ip_df = pd.read_csv(IP_PATH)
print(f"    Fraud_Data shape: {fraud.shape}")
print(f"    IpAddress_to_Country shape: {ip_df.shape}")

# 2. Parse datetimes
fraud["signup_time"] = pd.to_datetime(fraud["signup_time"])
fraud["purchase_time"] = pd.to_datetime(fraud["purchase_time"])

# 3. Clean
before = len(fraud)
fraud = fraud.drop_duplicates().dropna(subset=["class"])
fraud["class"] = fraud["class"].astype(int)
print(f"    Rows after cleaning: {len(fraud):,} (removed {before - len(fraud)})")
print(f"    Fraud rate: {fraud['class'].mean():.4f} "
      f"({fraud['class'].sum():,} / {len(fraud):,})")

# 4. Feature engineering
print("\n[2] Engineering features ...")
fraud["time_since_signup"] = (
    fraud["purchase_time"] - fraud["signup_time"]
).dt.total_seconds()

fraud["hour_of_day"] = fraud["purchase_time"].dt.hour
fraud["day_of_week"] = fraud["purchase_time"].dt.dayofweek

tx_count = fraud.groupby("user_id")["purchase_time"].transform("count")
fraud["user_tx_count"] = tx_count
u_first = fraud.groupby("user_id")["purchase_time"].transform("min")
u_last = fraud.groupby("user_id")["purchase_time"].transform("max")
days_active = ((u_last - u_first).dt.total_seconds() / 86400).clip(lower=1)
fraud["user_tx_per_day"] = fraud["user_tx_count"] / days_active
fraud["device_user_count"] = (
    fraud.groupby("device_id")["user_id"].transform("nunique")
)
print("    Added: time_since_signup, hour_of_day, day_of_week, "
      "user_tx_count, user_tx_per_day, device_user_count")

# 5. IP-to-country merge
print("\n[3] Merging IP addresses to countries ...")
fraud["ip_int"] = fraud["ip_address"].astype(np.float64)
ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(
    np.float64
)
ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(
    np.float64
)
ip_sorted = ip_df.sort_values("lower_bound_ip_address").reset_index(drop=True)
fraud_sorted = fraud.sort_values("ip_int").reset_index(drop=True)

merged = pd.merge_asof(
    fraud_sorted,
    ip_sorted[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
    left_on="ip_int",
    right_on="lower_bound_ip_address",
    direction="backward",
)
merged.loc[merged["ip_int"] > merged["upper_bound_ip_address"], "country"] = (
    np.nan
)
unmatched = merged["country"].isna().sum()
pct = unmatched / len(merged) * 100
print(f"    Unmatched IPs: {unmatched:,} ({pct:.1f}%)")

top_c = merged["country"].value_counts().nlargest(10).index
merged["country_grouped"] = np.where(
    merged["country"].isin(top_c), merged["country"], "Other"
)
merged = pd.get_dummies(
    merged, columns=["country_grouped"], drop_first=False, dtype=int
)
merged = pd.get_dummies(
    merged, columns=["source", "browser", "sex"], drop_first=True, dtype=int
)

drop_cols = [
    "class", "user_id", "device_id", "ip_address",
    "signup_time", "purchase_time", "ip_int",
    "lower_bound_ip_address", "upper_bound_ip_address", "country",
]
feature_cols = [c for c in merged.columns if c not in drop_cols]
print(f"    Feature columns: {len(feature_cols)}")

X = merged[feature_cols]
y = merged["class"]

# 6. Stratified train/test split
print("\n[4] Stratified train/test split (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"    Train fraud rate: {y_train.mean():.4f} "
      f"| Test fraud rate: {y_test.mean():.4f}")

# 7. Scale (fit on train only)
print("\n[5] Scaling numeric features (StandardScaler, fit on train only) ...")
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[FRAUD_NUM_COLS] = scaler.fit_transform(X_train[FRAUD_NUM_COLS])
X_test_s[FRAUD_NUM_COLS] = scaler.transform(X_test[FRAUD_NUM_COLS])

# 8. SMOTE (train only)
print("\n[6] Applying SMOTE to training set only ...")
print(f"    Before: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)
vc = pd.Series(y_train_res).value_counts()
print(f"    After:  {vc.to_dict()}")

# 9. Train and evaluate all models
section("Fraud_Data — Model Training & Evaluation")

fraud_results = []

print("\n--- Logistic Regression (Baseline) ---")
fraud_results.append(evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    X_train_res, y_train_res, X_test_s, y_test,
))

print("\n--- Random Forest ---")
fraud_results.append(evaluate(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    ),
    X_train_res, y_train_res, X_test_s, y_test,
))

print("\n--- XGBoost ---")
fraud_results.append(evaluate(
    "XGBoost",
    XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    ),
    X_train_res, y_train_res, X_test_s, y_test,
))

print("\n--- LightGBM ---")
lgb_fraud = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    num_leaves=63, subsample=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
fraud_results.append(evaluate(
    "LightGBM", lgb_fraud,
    X_train_res, y_train_res, X_test_s, y_test,
))

# 10. Metrics summary table
section("Fraud_Data — Metrics Comparison")
fraud_df = pd.DataFrame([
    {
        "Dataset": "Fraud_Data",
        "Model": r["name"],
        "F1": r["f1"],
        "AUC-PR": r["auc_pr"],
        "ROC-AUC": r["roc_auc"],
        "TP": r["tp"], "FP": r["fp"],
        "FN": r["fn"], "TN": r["tn"],
    }
    for r in fraud_results
]).sort_values("AUC-PR", ascending=False).reset_index(drop=True)
print(fraud_df.to_string(index=False))

# 11. 5-fold stratified CV on LightGBM
section("Fraud_Data — 5-Fold Stratified CV (LightGBM)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_pr = [], []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    sc_cv = StandardScaler()
    X_tr[FRAUD_NUM_COLS] = sc_cv.fit_transform(X_tr[FRAUD_NUM_COLS])
    X_val[FRAUD_NUM_COLS] = sc_cv.transform(X_val[FRAUD_NUM_COLS])

    sm_cv = SMOTE(random_state=42)
    X_tr_r, y_tr_r = sm_cv.fit_resample(X_tr, y_tr)

    cv_model = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    cv_model.fit(X_tr_r, y_tr_r)
    yp = cv_model.predict(X_val)
    ypr = cv_model.predict_proba(X_val)[:, 1]
    f1v = round(f1_score(y_val, yp), 4)
    prv = round(average_precision_score(y_val, ypr), 4)
    cv_f1.append(f1v)
    cv_pr.append(prv)
    print(f"  Fold {fold + 1}: F1={f1v:.4f}  AUC-PR={prv:.4f}")

print(f"\n  CV F1:     mean={np.mean(cv_f1):.4f}  std={np.std(cv_f1):.4f}")
print(f"  CV AUC-PR: mean={np.mean(cv_pr):.4f}  std={np.std(cv_pr):.4f}")

# 12. Save best fraud model
best_fraud = sorted(fraud_results, key=lambda x: x["auc_pr"], reverse=True)[0]
print(f"\n  Best model: {best_fraud['name']} (AUC-PR={best_fraud['auc_pr']})")
fraud_bundle = {
    "model": best_fraud["model"],
    "scaler": scaler,
    "feature_cols": feature_cols,
    "name": best_fraud["name"],
    "metrics": {
        "f1": best_fraud["f1"],
        "auc_pr": best_fraud["auc_pr"],
        "roc_auc": best_fraud["roc_auc"],
    },
    "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
    "cv_pr_mean": round(float(np.mean(cv_pr)), 4),
}
fraud_save_path = os.path.join(MODELS_DIR, "best_fraud_model.pkl")
with open(fraud_save_path, "wb") as f:
    pickle.dump(fraud_bundle, f)
print(f"  Saved → {fraud_save_path}")


# ── PART B: creditcard ────────────────────────────────────────────────────────

section("PART B — creditcard.csv (Bank Transactions)")

# 1. Load
print("\n[1] Loading creditcard.csv ...")
credit = pd.read_csv(CREDIT_PATH)
before_c = len(credit)
credit = credit.drop_duplicates()
credit["Class"] = credit["Class"].astype(int)
print(f"    Shape after dedup: {credit.shape} (removed {before_c - len(credit)})")
print(f"    Fraud rate: {credit['Class'].mean():.4f} "
      f"({credit['Class'].sum():,} / {len(credit):,})")

Xc = credit.drop(columns=["Class"])
yc = credit["Class"]

# 2. Stratified split
print("\n[2] Stratified train/test split (80/20) ...")
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)
print(f"    Train: {len(Xc_tr):,} | Test: {len(Xc_te):,}")
print(f"    Train fraud rate: {yc_tr.mean():.4f} "
      f"| Test fraud rate: {yc_te.mean():.4f}")

# 3. Scale Time and Amount (V1-V28 are already PCA-scaled)
print("\n[3] Scaling Time and Amount (fit on train only) ...")
scaler_c = StandardScaler()
Xc_tr_s = Xc_tr.copy()
Xc_te_s = Xc_te.copy()
Xc_tr_s[CREDIT_NUM_COLS] = scaler_c.fit_transform(Xc_tr[CREDIT_NUM_COLS])
Xc_te_s[CREDIT_NUM_COLS] = scaler_c.transform(Xc_te[CREDIT_NUM_COLS])

# 4. Random undersampling (train only)
print("\n[4] Applying random undersampling to training set only ...")
print(f"    Before: {yc_tr.value_counts().to_dict()}")
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
Xc_tr_r, yc_tr_r = rus.fit_resample(Xc_tr_s, yc_tr)
vc_c = pd.Series(yc_tr_r).value_counts()
print(f"    After:  {vc_c.to_dict()}")

# 5. Train and evaluate all models
section("creditcard — Model Training & Evaluation")

credit_results = []

print("\n--- Logistic Regression (Baseline) ---")
credit_results.append(evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    Xc_tr_r, yc_tr_r, Xc_te_s, yc_te,
))

print("\n--- Random Forest ---")
credit_results.append(evaluate(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    ),
    Xc_tr_r, yc_tr_r, Xc_te_s, yc_te,
))

print("\n--- XGBoost ---")
credit_results.append(evaluate(
    "XGBoost",
    XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    ),
    Xc_tr_r, yc_tr_r, Xc_te_s, yc_te,
))

print("\n--- LightGBM ---")
credit_results.append(evaluate(
    "LightGBM",
    LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    ),
    Xc_tr_r, yc_tr_r, Xc_te_s, yc_te,
))

# 6. Metrics summary table
section("creditcard — Metrics Comparison")
credit_df = pd.DataFrame([
    {
        "Dataset": "creditcard",
        "Model": r["name"],
        "F1": r["f1"],
        "AUC-PR": r["auc_pr"],
        "ROC-AUC": r["roc_auc"],
        "TP": r["tp"], "FP": r["fp"],
        "FN": r["fn"], "TN": r["tn"],
    }
    for r in credit_results
]).sort_values("AUC-PR", ascending=False).reset_index(drop=True)
print(credit_df.to_string(index=False))

# 7. Save best credit model
best_credit = sorted(
    credit_results, key=lambda x: x["auc_pr"], reverse=True
)[0]
print(f"\n  Best model: {best_credit['name']} "
      f"(AUC-PR={best_credit['auc_pr']})")
credit_bundle = {
    "model": best_credit["model"],
    "scaler": scaler_c,
    "feature_cols": list(Xc.columns),
    "name": best_credit["name"],
    "metrics": {
        "f1": best_credit["f1"],
        "auc_pr": best_credit["auc_pr"],
        "roc_auc": best_credit["roc_auc"],
    },
}
credit_save_path = os.path.join(MODELS_DIR, "best_credit_model.pkl")
with open(credit_save_path, "wb") as f:
    pickle.dump(credit_bundle, f)
print(f"  Saved → {credit_save_path}")


# ── Save combined metrics table ────────────────────────────────────────────────
section("Saving Combined Metrics Table")
all_results = pd.concat(
    [fraud_df, credit_df], ignore_index=True
)
csv_path = os.path.join(MODELS_DIR, "results_summary.csv")
all_results.to_csv(csv_path, index=False)
print(f"  Saved → {csv_path}")
print(all_results.to_string(index=False))


# ── Final summary ─────────────────────────────────────────────────────────────
section("FINAL SUMMARY")
print(f"  Fraud_Data  → Best: {best_fraud['name']}"
      f"  AUC-PR={best_fraud['auc_pr']}  F1={best_fraud['f1']}")
print(f"  creditcard  → Best: {best_credit['name']}"
      f"  AUC-PR={best_credit['auc_pr']}  F1={best_credit['f1']}")
print("\n  5-Fold CV (LightGBM / Fraud_Data):")
print(f"    AUC-PR = {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")
print(f"    F1     = {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print(f"\n  Artifacts saved to: {MODELS_DIR}/")
print("    best_fraud_model.pkl")
print("    best_credit_model.pkl")
print("    results_summary.csv")
print("\nDone.")
