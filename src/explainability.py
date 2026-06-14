"""
src/explainability.py

SHAP-based model explainability utilities.
All functions are flake8-compliant (max line length 88).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# 1. SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    model_type: str = "tree",
) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model      : fitted tree-based model (LightGBM, XGBoost, RandomForest)
    X_sample   : DataFrame of input features to explain
    model_type : currently only 'tree' is supported

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
                  For binary classification, returns the fraud-class (class 1)
                  SHAP values.
    explainer   : fitted shap.TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    # LightGBM returns [class0_values, class1_values] for binary
    if isinstance(raw, list) and len(raw) == 2:
        shap_values = raw[1]
    else:
        shap_values = raw

    return shap_values, explainer


def mean_absolute_shap(
    shap_values: np.ndarray,
    feature_cols: list,
) -> pd.Series:
    """
    Return mean absolute SHAP value per feature, sorted descending.
    This is the global feature importance measure.
    """
    return (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
        .sort_values(ascending=False)
    )


# ---------------------------------------------------------------------------
# 2. Summary plot
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    title: str = "SHAP Summary Plot",
    max_display: int = 15,
    save_path: str = None,
) -> None:
    """
    Beeswarm summary plot — shows feature impact direction and magnitude
    for all samples.

    Each dot = one sample. Color = feature value (red=high, blue=low).
    X-axis = SHAP value (positive = pushes toward fraud).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(title, fontsize=13, pad=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Bar importance plot
# ---------------------------------------------------------------------------

def plot_shap_bar(
    shap_values: np.ndarray,
    feature_cols: list,
    title: str = "Top Feature Importances (Mean |SHAP|)",
    top_n: int = 10,
    color: str = "steelblue",
    save_path: str = None,
) -> None:
    """
    Horizontal bar chart of mean absolute SHAP values.
    Easier to read than beeswarm for stakeholder presentations.
    """
    importance = mean_absolute_shap(shap_values, feature_cols).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 5))
    importance[::-1].plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)

    for bar in ax.patches:
        ax.text(
            bar.get_width() + importance.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Force plots (individual explanations)
# ---------------------------------------------------------------------------

def plot_force_matplotlib(
    explainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    sample_pos: int,
    title: str = "",
    save_path: str = None,
) -> None:
    """
    Matplotlib-compatible force plot for one sample.
    Shows which features push the prediction up (red) or down (blue).

    Parameters
    ----------
    explainer   : fitted shap.TreeExplainer
    shap_values : SHAP values array (n_samples, n_features)
    X_sample    : DataFrame used for explanation
    sample_pos  : integer position (iloc) of the sample to explain
    """
    fig, ax = plt.subplots(figsize=(14, 3))

    sv_row = shap_values[sample_pos]
    x_row = X_sample.iloc[sample_pos]
    feature_names = list(X_sample.columns)

    # Select top contributing features (by |shap|)
    top_n = 8
    top_idx = np.argsort(np.abs(sv_row))[::-1][:top_n]

    labels = [f"{feature_names[i]}\n={x_row.iloc[i]:.2f}" for i in top_idx]
    values = sv_row[top_idx]
    colors = ["tomato" if v > 0 else "steelblue" for v in values]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (positive = toward Fraud)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (0.005 if w >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.4f}",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Dependence plot
# ---------------------------------------------------------------------------

def plot_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    title: str = None,
    save_path: str = None,
) -> None:
    """
    SHAP dependence plot for a single feature.
    Shows how SHAP value changes with feature value, colored by an
    interaction feature.
    """
    feature_names = list(X_sample.columns)
    feat_idx = feature_names.index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_sample,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Business insight helper
# ---------------------------------------------------------------------------

def top_fraud_drivers(
    shap_values: np.ndarray,
    feature_cols: list,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top N fraud drivers with mean SHAP,
    mean positive SHAP (pushes toward fraud), and % of samples where
    the feature increases fraud probability.

    Designed for reporting and business recommendations.
    """
    rows = []
    for i, feat in enumerate(feature_cols):
        col_shap = shap_values[:, i]
        rows.append({
            "Feature": feat,
            "Mean |SHAP|": round(float(np.abs(col_shap).mean()), 4),
            "Mean SHAP (fraud direction)": round(
                float(col_shap[col_shap > 0].mean())
                if (col_shap > 0).any() else 0.0, 4
            ),
            "% samples pushing to fraud": round(
                float((col_shap > 0).mean() * 100), 1
            ),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("Mean |SHAP|", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df.index = df.index + 1  # 1-based ranking
    return df
