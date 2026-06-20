"""
SHAP-based explainability for the CreditLens XGBoost model.

Functions
---------
build_explainer(model, X_background)
    Create a shap.TreeExplainer and compute SHAP values for the background set.
    Caches the explainer to models/shap_explainer.joblib for reuse in the app.

global_importance(shap_values, X_background, feature_names, out_path)
    Generate beeswarm + mean-|SHAP| bar chart side by side; save to data/shap_global.png.

explain_one(explainer, row, feature_names)
    Compute SHAP values for a single applicant; return top drivers + adverse-action reasons.
    Used by the Phase-5 Streamlit app to render a per-applicant waterfall chart.
"""

import os
import tempfile
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_MODELS_DIR = os.path.join(_REPO_DIR, "models")
_DATA_DIR = os.path.join(_REPO_DIR, "data")
_EXPLAINER_PATH = os.path.join(_MODELS_DIR, "shap_explainer.joblib")
_GLOBAL_PNG = os.path.join(_DATA_DIR, "shap_global.png")

# Plain-English adverse-action reasons for regulated lending context.
# Under ECOA/FCRA, lenders must cite specific reasons for an adverse credit decision.
_REASON_MAP = {
    "PAY_0":             "Recent payment delinquency (most recent month)",
    "PAY_2":             "Payment delinquency 2 months prior",
    "PAY_3":             "Payment delinquency 3 months prior",
    "PAY_4":             "Payment delinquency 4 months prior",
    "PAY_5":             "Payment delinquency 5 months prior",
    "PAY_6":             "Payment delinquency 6 months prior",
    "months_delinquent": "High number of months with late payments",
    "utilization":       "High credit utilisation ratio",
    "avg_pay_ratio":     "Low payment-to-balance ratio (consistently underpaying)",
    "bill_trend":        "Rising outstanding balance trend",
    "LIMIT_BAL":         "Credit limit relative to peers",
    "AGE":               "Age of borrower",
    "BILL_AMT1":         "High current outstanding balance",
    "BILL_AMT2":         "High balance 2 months prior",
    "BILL_AMT3":         "High balance 3 months prior",
    "BILL_AMT4":         "High balance 4 months prior",
    "BILL_AMT5":         "High balance 5 months prior",
    "BILL_AMT6":         "High balance 6 months prior",
    "PAY_AMT1":          "Low payment amount (most recent month)",
    "PAY_AMT2":          "Low payment amount 2 months prior",
    "PAY_AMT3":          "Low payment amount 3 months prior",
    "PAY_AMT4":          "Low payment amount 4 months prior",
    "PAY_AMT5":          "Low payment amount 5 months prior",
    "PAY_AMT6":          "Low payment amount 6 months prior",
    "SEX":               "Gender of applicant",
    "EDUCATION":         "Education level of applicant",
    "MARRIAGE":          "Marital status of applicant",
}


def build_explainer(model, X_background: np.ndarray):
    """Build a shap.TreeExplainer over the trained XGBoost model.

    Uses the training data as the background distribution so SHAP values represent
    the deviation of each prediction from the average model output over training.
    The explainer is cached to disk so the Streamlit app can load it without
    rerunning the expensive shap_values() computation at every request.

    Parameters
    ----------
    model : xgboost.XGBClassifier
        Fitted XGBoost model (from train_xgb()).
    X_background : np.ndarray
        Training data matrix — used as the reference distribution for SHAP.

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values : np.ndarray
        SHAP values for each sample in X_background, shape (n_samples, n_features).
        Positive values push toward default; negative values push away from default.
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_background)

    # shap < 0.42 returns a list [neg_class, pos_class] for binary classifiers;
    # shap >= 0.42 returns just the positive-class array directly.
    if isinstance(raw, list):
        shap_values = raw[1]
    else:
        shap_values = raw

    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.makedirs(_DATA_DIR, exist_ok=True)
    joblib.dump(explainer, _EXPLAINER_PATH)
    print(f"Explainer cached     → {_EXPLAINER_PATH}")
    print(f"SHAP values shape    : {shap_values.shape}")

    return explainer, shap_values


def global_importance(
    shap_values: np.ndarray,
    X_background: np.ndarray,
    feature_names: list,
    out_path: str = _GLOBAL_PNG,
) -> str:
    """Generate a two-panel SHAP figure: beeswarm (left) + mean-|SHAP| bar (right).

    The beeswarm shows every borrower in the background set as a dot:
      - Vertical axis = feature (sorted by mean absolute SHAP impact)
      - Horizontal axis = SHAP value (positive → increases default probability)
      - Color = feature value relative to the population (red = high, blue = low)

    The bar chart shows the mean absolute SHAP value per feature — the standard
    global feature-importance ranking for SHAP write-ups.

    Both panels are rendered to temporary PNGs, then tiled side by side into
    a single output file so the README can reference one image.

    Parameters
    ----------
    shap_values : np.ndarray  shape (n, n_features)
    X_background : np.ndarray  shape (n, n_features)  feature matrix
    feature_names : list[str]  ordered column names
    out_path : str  path for the output PNG (default: data/shap_global.png)

    Returns
    -------
    str : path to the saved PNG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp_dir = tempfile.mkdtemp()
    beeswarm_path = os.path.join(tmp_dir, "beeswarm.png")
    bar_path = os.path.join(tmp_dir, "bar.png")

    # --- Beeswarm ---
    shap.summary_plot(
        shap_values, X_background,
        feature_names=feature_names,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close("all")

    # --- Bar (mean |SHAP|) ---
    shap.summary_plot(
        shap_values, X_background,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.gcf().set_size_inches(7, 8)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close("all")

    # --- Tile side by side ---
    img_bee = plt.imread(beeswarm_path)
    img_bar = plt.imread(bar_path)

    # Pad shorter image vertically so heights match before concatenation
    h1, h2 = img_bee.shape[0], img_bar.shape[0]
    if h1 != h2:
        pad = abs(h1 - h2)
        channels = img_bee.shape[2] if img_bee.ndim == 3 else 1
        if h1 < h2:
            padding = np.ones((pad, img_bee.shape[1], channels), dtype=img_bee.dtype)
            img_bee = np.concatenate([img_bee, padding], axis=0)
        else:
            padding = np.ones((pad, img_bar.shape[1], channels), dtype=img_bar.dtype)
            img_bar = np.concatenate([img_bar, padding], axis=0)

    combined = np.concatenate([img_bee, img_bar], axis=1)
    fig, ax = plt.subplots(
        figsize=(combined.shape[1] / 100, combined.shape[0] / 100)
    )
    ax.imshow(combined)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"SHAP global importance → {out_path}")
    return out_path


def explain_one(
    explainer,
    row: np.ndarray,
    feature_names: list,
) -> dict:
    """Compute a per-applicant SHAP explanation.

    Returns all data needed by the Phase-5 Streamlit app to render a waterfall
    chart and surface the top-3 adverse-action reasons for a rejected applicant.

    Adverse-action reasons are the features with the largest *positive* SHAP values
    (i.e., features that push the score toward default). Only features increasing risk
    are cited — features that lower risk are not relevant to adverse action.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Cached explainer from build_explainer().
    row : np.ndarray
        Single-applicant feature vector, shape (n_features,) or (1, n_features).
        Must be preprocessed (scaled/encoded) the same way as the training data.
    feature_names : list[str]
        Ordered column names matching the preprocessed feature matrix.

    Returns
    -------
    dict with:
      shap_values    : np.ndarray (n_features,)  per-feature SHAP values
      base_value     : float  model's average log-odds output over the training set
      feature_names  : list[str]
      feature_values : np.ndarray (n_features,)  raw preprocessed values
      top_drivers    : list[(feature_name, shap_value)]  all features by |shap| desc
      adverse_action : list[str]  top-3 plain-language reasons pushing score up
    """
    if row.ndim == 1:
        row = row.reshape(1, -1)

    raw = explainer.shap_values(row)
    if isinstance(raw, list):
        sv = raw[1][0]
    else:
        sv = raw[0]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[1])
    else:
        base = float(base)

    order = np.argsort(np.abs(sv))[::-1]
    top_drivers = [(feature_names[i], float(sv[i])) for i in order]

    # Only features pushing the score *up* (positive SHAP) are adverse-action reasons
    adverse = [
        _REASON_MAP.get(fname, fname)
        for fname, sv_val in top_drivers
        if sv_val > 0
    ][:3]

    return {
        "shap_values":    sv,
        "base_value":     base,
        "feature_names":  list(feature_names),
        "feature_values": row[0],
        "top_drivers":    top_drivers,
        "adverse_action": adverse,
    }


if __name__ == "__main__":
    import joblib as _joblib
    from src.data_loader import load_raw
    from src.preprocess import build_split, get_feature_names
    from src.train import train_xgb

    xgb_path = os.path.join(_MODELS_DIR, "xgb.joblib")

    df = load_raw()
    X_train, X_test, y_train, y_test, ct = build_split(df)
    feature_names = get_feature_names(ct)

    if os.path.exists(xgb_path):
        xgb = _joblib.load(xgb_path)
        print(f"Loaded XGBoost ← {xgb_path}")
    else:
        print("Training XGBoost (no persisted model found)...")
        xgb, _, _ = train_xgb()

    print("\n=== Building SHAP Explainer ===")
    explainer, shap_values = build_explainer(xgb, X_train)

    print("\n=== Global Feature Importance ===")
    out = global_importance(shap_values, X_train, feature_names)
    print(f"Plot saved → {out}")

    print("\n=== Top 10 Global Drivers (mean |SHAP|) ===")
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    for name, score in ranked[:10]:
        print(f"  {name:25s}  {score:.4f}")

    print("\n=== Sample Applicant Explanation ===")
    result = explain_one(explainer, X_test[0], feature_names)
    print(f"Base value (expected log-odds): {result['base_value']:.4f}")
    print("Top 5 drivers (by |SHAP|):")
    for fname, sv in result["top_drivers"][:5]:
        direction = "↑ risk" if sv > 0 else "↓ risk"
        print(f"  {fname:25s}  {sv:+.4f}  {direction}")
    print("Adverse-action reasons:")
    for i, reason in enumerate(result["adverse_action"], 1):
        print(f"  {i}. {reason}")
