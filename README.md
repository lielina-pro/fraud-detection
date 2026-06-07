# Fraud Detection for E-commerce and Bank Transactions

**10 Academy – KAIM Week 5 & 6**
**Adey Innovations Inc.**

---

## Project Overview

This project builds an end-to-end fraud detection system for two transaction types:

| Dataset | Description | Target |
|---|---|---|
| `Fraud_Data.csv` | E-commerce transactions with user/device/IP context | `class` (1=fraud) |
| `creditcard.csv` | Bank transactions with PCA-anonymized features V1–V28 | `Class` (1=fraud) |

Both datasets are **highly imbalanced** — fraud is a small minority. The system handles this with appropriate resampling (SMOTE for e-commerce, undersampling for credit card) and evaluates using **AUC-PR and F1-Score** rather than overall accuracy.

---

## Repository Structure

```
fraud-detection/
├── .github/workflows/unittests.yml   # CI: flake8 lint + pytest
├── data/
│   ├── raw/                          # Original CSVs (gitignored)
│   └── processed/                    # Cleaned, engineered data
├── notebooks/
│   ├── eda-fraud-data.py             # Task 1: EDA + preprocessing for Fraud_Data
│   ├── eda-creditcard.py             # Task 1: EDA + preprocessing for creditcard
│   ├── modeling.py                   # Task 2: Model building (coming in Interim-2)
│   └── shap-explainability.py        # Task 3: SHAP analysis (coming in Final)
├── src/
│   ├── data_preprocessing.py         # Loading, cleaning, IP-to-country merge
│   ├── feature_engineering.py        # Time, velocity, encoding, scaling
│   └── imbalance_handler.py          # SMOTE, undersampling, class reports
├── tests/
│   └── test_preprocessing.py         # Unit tests for src/ modules
├── models/                           # Saved model artifacts (gitignored)
├── requirements.txt
├── .flake8                           # Flake8 configuration
└── .gitignore
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd fraud-detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add data files

Place your downloaded CSVs in `data/raw/`:

```
data/raw/Fraud_Data.csv
data/raw/IpAddress_to_Country.csv
data/raw/creditcard.csv            # or creditcard.csv.zip
```

### 3. Run notebooks

The `.py` files in `notebooks/` are percent-script format — open them in VS Code
with the Jupyter extension, or convert to `.ipynb`:

```bash
jupytext --to notebook notebooks/eda-fraud-data.py
jupyter notebook notebooks/eda-fraud-data.ipynb
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Run linting

```bash
flake8 src/ scripts/ tests/
```

---

## Key Design Decisions

### Why SMOTE for Fraud_Data?
Fraud_Data has ~9% fraud rate with ~150k rows. Undersampling would discard most of the
legitimate signal. SMOTE generates synthetic fraud examples in feature space, retaining
all legitimate data while balancing classes. Applied on **training set only**.

### Why undersampling for creditcard.csv?
creditcard.csv has 284k rows with only 0.17% fraud. Even after 50% undersampling,
the training set remains large and representative. This avoids the computational cost
of SMOTE on a very large dataset.

### Why AUC-PR over ROC-AUC?
On severely imbalanced data, ROC-AUC can be misleadingly optimistic (a model that
predicts all legitimate still gets high ROC-AUC). AUC-PR focuses on the minority
class performance and is the correct metric here.

---

## Timeline

| Milestone | Date |
|---|---|
| Interim-1 (Task 1 complete) | Sun 07 Jun 2026 |
| Interim-2 (Task 2 complete) | Sun 14 Jun 2026 |
| Final Submission | Tue 16 Jun 2026 |
