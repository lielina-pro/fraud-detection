# Fraud Detection for E-commerce and Bank Transactions

[![CI – Lint and Tests](https://github.com/lielina-pro/fraud-detection/actions/workflows/unittests.yml/badge.svg)](https://github.com/lielina-pro/fraud-detection/actions/workflows/unittests.yml)
![Python](https://img.shields.io/badge/Python-3.14.3-blue?logo=python)
![Tests](https://img.shields.io/badge/Tests-46%20passed-brightgreen)
![Flake8](https://img.shields.io/badge/Flake8-0%20errors-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Author:** Lielina Fekadu  
> **Program:** 10 Academy KAIM — Week 5 & 6  
> **Organisation:** Adey Innovations Inc.  
> **Repo:** [github.com/lielina-pro/fraud-detection](https://github.com/lielina-pro/fraud-detection)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results](#2-key-results)
3. [Repository Structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [Setup & Installation](#5-setup--installation)
6. [Running the Notebooks](#6-running-the-notebooks)
7. [Running Tests & Linting](#7-running-tests--linting)
8. [Project Pipeline](#8-project-pipeline)
9. [Feature Engineering](#9-feature-engineering)
10. [Model Performance](#10-model-performance)
11. [SHAP Explainability](#11-shap-explainability)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Business Recommendations](#13-business-recommendations)
14. [CI/CD Pipeline](#14-cicd-pipeline)

---

## 1. Project Overview

Fraud costs the global economy hundreds of billions of dollars annually. This project builds a **production-ready, end-to-end fraud detection system** covering two business lines for Adey Innovations Inc.:

- **E-commerce transactions** — rich behavioural and device signals (Fraud_Data.csv)
- **Bank credit card transactions** — PCA-anonymised transaction features (creditcard.csv)

Both datasets are severely class-imbalanced. The system applies tailored resampling strategies for each, evaluates all models on **AUC-PR and F1-Score** (not accuracy), and delivers **SHAP-based explanations** that are directly actionable by fraud operations and compliance teams.

The full project spans three tasks:

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Data analysis, cleaning, feature engineering, class imbalance handling | ✅ Complete |
| Task 2 | Model building, evaluation, stratified K-fold cross-validation | ✅ Complete |
| Task 3 | SHAP explainability, force plots, business recommendations | ✅ Complete |

---

## 2. Key Results

| Dataset | Best Model | F1-Score | AUC-PR | ROC-AUC |
|---------|-----------|----------|--------|---------|
| Fraud_Data.csv (E-commerce) | **LightGBM** | 0.6832 | 0.7027 | 0.8339 |
| creditcard.csv (Bank) | **XGBoost** | 0.2151 | 0.7545 | 0.9780 |

**LightGBM 5-fold cross-validation (Fraud_Data):** AUC-PR = 0.7162 ± 0.0073

**Top fraud signal discovered:** 99.52% of transactions made within 60 minutes of account creation are fraudulent.

---

## 2b. Task 2 — Quick Start (Model Training)

> **For graders:** This single command runs the complete Task 2 pipeline on both datasets — no notebook required.

```powershell
# From the project root (with .venv activated and data in data/raw/)
python scripts/train_models.py
```

This script:
- Loads `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` from `data/raw/`
- Runs full preprocessing and feature engineering
- Performs **stratified 80/20 train/test split** on both datasets
- Trains **Logistic Regression (baseline)**, **Random Forest**, **XGBoost**, **LightGBM**
- Prints **AUC-PR, F1-Score, ROC-AUC, and Confusion Matrix** for every model
- Runs **5-fold stratified cross-validation** (SMOTE inside each fold)
- Saves `models/best_fraud_model.pkl`, `models/best_credit_model.pkl`
- Saves `models/results_summary.csv` — the full metrics comparison table

The `models/results_summary.csv` is **committed to the repository** so results are visible without running anything.

---

## 3. Repository Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI: flake8 lint + pytest on every push
├── data/
│   ├── raw/                       # Original CSVs — gitignored, place files here
│   └── processed/                 # Cleaned & engineered outputs — gitignored
├── models/
│   ├── .gitkeep                   # Ensures models/ directory is visible on GitHub
│   └── results_summary.csv        # ★ Metrics table committed to repo for visibility
├── notebooks/
│   ├── eda-fraud-data.py          # Task 1: EDA + preprocessing (Fraud_Data)
│   ├── eda-creditcard.py          # Task 1: EDA + preprocessing (creditcard)
│   ├── modeling.py                # Task 2: Model training + evaluation
│   └── shap-explainability.py     # Task 3: SHAP analysis + recommendations
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Loading, cleaning, IP-to-country merge
│   ├── feature_engineering.py     # Time, velocity, encoding, scaling
│   ├── imbalance_handler.py       # SMOTE, undersampling, class balance reports
│   ├── model_trainer.py           # Training, evaluation, cross-validation
│   └── explainability.py          # SHAP computation, plots, business drivers
├── scripts/
│   └── train_models.py            # ★ Task 2: Self-contained end-to-end training script
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # 14 tests — data cleaning & feature engineering
│   ├── test_model_trainer.py      # 19 tests — model utilities
│   └── test_explainability.py     # 13 tests — SHAP functions
├── .flake8                        # Flake8 config (max-line-length 88)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Datasets

> **Data files are not committed to Git** (they are gitignored due to size and sensitivity).  
> Place all three files inside `data/raw/` before running any notebook.

| File | Size | Description | Target Column |
|------|------|-------------|---------------|
| `Fraud_Data.csv` | 15.3 MB | E-commerce transactions with user ID, device ID, IP address, signup/purchase timestamps, and purchase value | `class` (0=legit, 1=fraud) |
| `IpAddress_to_Country.csv` | 4.7 MB | IP range to country lookup table used to enrich Fraud_Data with geolocation | Reference only |
| `creditcard.csv` | 66 MB | Bank card transactions with PCA-anonymised features V1–V28, plus Time and Amount | `Class` (0=legit, 1=fraud) |

These files were provided via Google Drive as part of the 10 Academy KAIM Week 5 & 6 project materials.

---

## 5. Setup & Installation

### Prerequisites

- Python **3.14.3** (the version this project was developed and tested on)
- Git
- Windows PowerShell or any terminal

### Step 1 — Clone the repository

```powershell
git clone https://github.com/lielina-pro/fraud-detection.git
cd fraud-detection
```

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run this once first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

After activation you will see `(.venv)` at the start of your terminal prompt.

### Step 3 — Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses `>=` version specifiers to ensure compatibility with Python 3.14. Fixed `==` pins will fail on Python 3.14 because many packages do not yet publish pre-built wheels for it.

### Step 4 — Place the data files

```
fraud-detection/
└── data/
    └── raw/
        ├── Fraud_Data.csv
        ├── IpAddress_to_Country.csv
        └── creditcard.csv
```

### Step 5 — Verify everything works

```powershell
python -m pytest tests/ -v        # should show: 46 passed
flake8 src/ tests/                 # should show: 0 (no output)
```

---

## 6. Running the Notebooks

All notebooks are written in **percent-script format** (`.py` files with `# %%` cell markers). They can be run interactively in VS Code or converted to Jupyter `.ipynb` files.

### Option A — VS Code (recommended)

1. Install the **Jupyter extension** in VS Code
2. Open any `.py` file in `notebooks/`
3. Click **Run Cell** (▶) above any `# %%` marker, or use `Shift+Enter`
4. Select `.venv` as the kernel when prompted

### Option B — Convert to Jupyter Notebook

```powershell
pip install jupytext
jupytext --to notebook notebooks/eda-fraud-data.py
jupytext --to notebook notebooks/eda-creditcard.py
jupytext --to notebook notebooks/modeling.py
jupytext --to notebook notebooks/shap-explainability.py
jupyter notebook
```

### Notebook run order

Run the notebooks in this order — each one depends on outputs from the previous:

```
1. notebooks/eda-fraud-data.py       → saves data/processed/fraud_data_featured.csv
2. notebooks/eda-creditcard.py       → saves data/processed/X_train_credit.csv etc.
3. notebooks/modeling.py             → saves models/best_fraud_model.pkl etc.
4. notebooks/shap-explainability.py  → loads saved models, generates SHAP plots
```

All plots are automatically saved to `data/processed/` as `.png` files.

---

## 7. Running Tests & Linting

### Run all 46 unit tests

```powershell
python -m pytest tests/ -v
```

Expected output:
```
46 passed in ~60s
```

### Run a specific test file

```powershell
python -m pytest tests/test_preprocessing.py -v      # 14 tests
python -m pytest tests/test_model_trainer.py -v      # 19 tests
python -m pytest tests/test_explainability.py -v     # 13 tests
```

### Run flake8 linting

```powershell
flake8 src/ tests/
```

Expected output: nothing (zero errors = success). The `.flake8` config sets `max-line-length = 88` and ignores `W503`.

---

## 8. Project Pipeline

```
Raw Data
   │
   ▼
┌─────────────────────────────────────────┐
│  Task 1 — Data Analysis & Preprocessing │
│  src/data_preprocessing.py              │
│  src/feature_engineering.py             │
│  src/imbalance_handler.py               │
│                                         │
│  • Clean & validate                     │
│  • IP → Country merge (merge_asof)      │
│  • Engineer 6 new features              │
│  • Stratified 80/20 split               │
│  • StandardScaler (fit on train only)   │
│  • SMOTE / Undersampling (train only)   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Task 2 — Model Building                │
│  src/model_trainer.py                   │
│                                         │
│  • Logistic Regression (baseline)       │
│  • Random Forest                        │
│  • XGBoost                              │
│  • LightGBM                             │
│  • Stratified K-Fold CV (k=5)           │
│  • Evaluate: AUC-PR, F1, ROC-AUC, CM   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Task 3 — Explainability                │
│  src/explainability.py                  │
│                                         │
│  • SHAP TreeExplainer                   │
│  • Summary + bar plots                  │
│  • Force plots (TP, FP, FN)             │
│  • Top 5 fraud drivers                  │
│  • 5 business recommendations           │
└─────────────────────────────────────────┘
```

---

## 9. Feature Engineering

Six features are engineered from the raw Fraud_Data columns. All are computed on the full dataset **before** the train/test split (to capture global velocity signals), while encoding and scaling are applied **after** the split to prevent data leakage.

| Feature | Derivation | Key Finding |
|---------|-----------|-------------|
| `time_since_signup` | `purchase_time − signup_time` in seconds | Median 0 hrs fraud vs 1,443 hrs legitimate. **99.52% of purchases within 60 min of signup are fraud** |
| `hour_of_day` | `purchase_time.hour` (0–23) | Peak fraud at hour 17 (10.7%), hour 9 (10.6%) |
| `day_of_week` | `purchase_time.dayofweek` (0=Mon) | Captures weekly behavioural patterns |
| `user_tx_count` | Total transactions per `user_id` | High-velocity accounts show elevated fraud probability |
| `user_tx_per_day` | `user_tx_count / days_active` | Normalises velocity across users |
| `device_user_count` | Unique `user_id`s per `device_id` | Shared devices: **52.5% fraud rate** vs 3.0% for single-user devices |

**IP-to-Country merge:** `ip_address` (stored as float integer) is range-merged with `IpAddress_to_Country.csv` using `pd.merge_asof` in backward direction. 14.5% of IPs are unresolvable (likely VPNs/Tor). Top 10 countries are one-hot encoded; all others become `country_Other`.

---

## 10. Model Performance

### Fraud_Data.csv (E-commerce) — SMOTE applied to training set

| Model | F1 | AUC-PR | ROC-AUC | Notes |
|-------|----|--------|---------|-------|
| Logistic Regression | 0.6389 | 0.6365 | 0.8232 | Baseline |
| Random Forest | 0.6317 | 0.7040 | 0.8344 | |
| XGBoost | 0.6685 | 0.7025 | 0.8358 | |
| **LightGBM** ✅ | **0.6832** | **0.7027** | **0.8339** | **Selected** |

**5-fold CV (LightGBM):** AUC-PR = 0.7162 ± 0.0073 | F1 = 0.6958 ± 0.0056

### creditcard.csv (Bank) — Random undersampling applied to training set

| Model | F1 | AUC-PR | ROC-AUC | Notes |
|-------|----|--------|---------|-------|
| Logistic Regression | 0.1663 | 0.6101 | 0.9578 | Baseline |
| Random Forest | 0.3450 | 0.6897 | 0.9735 | |
| **XGBoost** ✅ | **0.2151** | **0.7545** | **0.9780** | **Selected** |
| LightGBM | 0.2067 | 0.6721 | 0.9742 | |

> **Why AUC-PR and not accuracy?** A model that predicts every transaction as legitimate achieves 90.64% accuracy on Fraud_Data and 99.83% on creditcard — while catching zero fraud. AUC-PR and F1 focus exclusively on minority-class performance.

---

## 11. SHAP Explainability

SHAP (`TreeExplainer`) is used to explain both selected models. A positive SHAP value pushes a prediction toward fraud; a negative value pushes toward legitimate.

### Top 5 Fraud Drivers — Fraud_Data (LightGBM)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `time_since_signup` | 2.12 | Near-zero = almost certain fraud. Strongest signal by far |
| 2 | `device_user_count` | 1.49 | Shared devices drive 52.5% fraud rate |
| 3 | `day_of_week` | 0.32 | Certain days carry higher fraud concentration |
| 4 | `country_United States` | 0.19 | High volume — both fraud and legit |
| 5 | `hour_of_day` | 0.19 | Late afternoon and morning peaks |

### Top 5 Fraud Drivers — creditcard (XGBoost)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `V14` | 1.79 | Low V14 values strongly predict fraud |
| 2 | `V4` | 1.11 | High V4 values predict fraud |
| 3 | `V12` | 0.67 | Moderate fraud signal |
| 4 | `V10` | 0.66 | Moderate fraud signal |
| 5 | `V11` | 0.38 | Supplementary signal |

---

## 12. Key Design Decisions

**Why SMOTE for Fraud_Data and undersampling for creditcard?**
Fraud_Data has a 9.36% fraud rate with 151k rows — SMOTE generates synthetic fraud examples in feature space while retaining all legitimate data. creditcard has only 0.17% fraud across 284k rows — undersampling to a 2:1 ratio still leaves a large, representative training set and avoids SMOTE's computational cost at this scale. Both resampling operations are applied **only to the training split**.

**Why AUC-PR over ROC-AUC as the primary metric?**
On imbalanced data, ROC-AUC is misleadingly optimistic because it accounts for true negatives (the majority class). AUC-PR measures precision and recall exclusively on the minority (fraud) class — the operationally relevant objective.

**Why LightGBM for e-commerce and XGBoost for bank?**
LightGBM achieves the highest AUC-PR and F1 on Fraud_Data, confirmed by 5-fold CV showing low variance (±0.0073). XGBoost leads on AUC-PR for creditcard by a clear margin (0.7545 vs 0.6897 for Random Forest), and its boosting objective aligns naturally with ranking fraud candidates at extreme imbalance ratios.

**Why are velocity features computed before the train/test split?**
Features like `user_tx_count` aggregate across the full observation window. Computing them after splitting would produce incomplete counts in the training set and cause the model to learn on artificially truncated velocity signals. Encoding and scaling, however, are fitted **only on the training split** to prevent leakage.

---

## 13. Business Recommendations

| # | Recommendation | SHAP Evidence | Action |
|---|---------------|---------------|--------|
| 1 | Block near-zero `time_since_signup` | Mean \|SHAP\| = 2.12; 99.52% fraud within 1 hr of signup | Require OTP for any purchase within 60 min of signup; auto-hold within 5 min |
| 2 | Flag shared device fingerprints | Mean \|SHAP\| = 1.49; 52.5% fraud rate on shared devices | Flag devices linked to ≥2 accounts; block devices with ≥5 accounts pending ID check |
| 3 | Geographic risk scoring | Country dummies in top 10 SHAP; Norway 13.0%, Mexico 12.8% fraud rates | Tier countries by fraud rate; add CAPTCHA/OTP for high-risk or unresolvable IPs |
| 4 | Real-time velocity monitoring | `user_tx_count` and `user_tx_per_day` in top 10 SHAP | Alert if >3 transactions/hour or >10/day per user |
| 5 | De-anonymise V14/V4 for compliance | V14 Mean \|SHAP\| = 1.79 — dominates the bank model | Identify raw features behind V14/V4; monitor monthly for distribution drift |

---

## 14. CI/CD Pipeline

Every push and pull request to `main` or `develop` automatically triggers the GitHub Actions workflow defined in `.github/workflows/unittests.yml`:

```
Push / PR
    │
    ▼
┌──────────────────────────┐
│  ubuntu-latest           │
│  Python 3.11             │
│                          │
│  pip install -r          │
│  requirements.txt        │
│          │               │
│    ┌─────▼──────┐        │
│    │  flake8    │        │
│    │  src/      │        │
│    │  tests/    │        │
│    └─────┬──────┘        │
│          │               │
│    ┌─────▼──────┐        │
│    │  pytest    │        │
│    │  tests/ -v │        │
│    └────────────┘        │
└──────────────────────────┘
```

The CI badge at the top of this README reflects the live status of the latest run on the `main` branch.

---

*Built with Python 3.14.3 · scikit-learn · LightGBM · XGBoost · SHAP · pandas · imbalanced-learn*
