"""
CreditLens — Streamlit Credit Default Risk Scorer

Borrower enters key attributes → XGBoost outputs a default probability
→ risk band (Low / Medium / High) using the Phase-3 recall-optimised threshold
→ SHAP waterfall showing which features drove the score → adverse-action reasons.

Usage:
    streamlit run src/app.py

Prerequisites:
    python -m src.train      # builds models/xgb.joblib + models/logreg.joblib
    python -m src.explain    # builds models/shap_explainer.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split

# Resolve repo root so imports work regardless of cwd
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from src.preprocess import clean, CONTINUOUS_COLS, PAY_COLS, NOMINAL_INT_COLS, get_feature_names
from src.features import add_features, ENGINEERED_COLS
from src.explain import load_explainer, explain_one, plot_waterfall
from src.evaluate import find_recall_threshold

_MODELS_DIR = os.path.join(_REPO_DIR, "models")
_DATA_DIR = os.path.join(_REPO_DIR, "data")
_XGB_PATH = os.path.join(_MODELS_DIR, "xgb.joblib")
_CT_PATH = os.path.join(_MODELS_DIR, "preprocess.joblib")

# PAY_* status codes and their plain-English labels for the UI form
_PAY_OPTIONS = {
    -2: "-2 · No consumption",
    -1: "-1 · Paid duly (on time)",
     0:  "0 · Revolving credit",
     1:  "1 · One month past due",
     2:  "2 · Two months past due",
     3:  "3 · Three months past due",
     4:  "4 · Four months past due",
     5:  "5 · Five months past due",
     6:  "6 · Six months past due",
}
_PAY_VALUES = list(_PAY_OPTIONS.keys())


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load preprocessor, XGBoost, SHAP explainer, and compute operating threshold.

    Returns (ct, xgb, explainer, threshold, error_msg). error_msg is None on success.
    All heavy objects are cached for the process lifetime via @st.cache_resource so
    concurrent users share a single loaded copy.
    """
    missing = [p for p in [_XGB_PATH, _CT_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, None, "\n".join(f"Missing artifact: {p}" for p in missing)

    ct = joblib.load(_CT_PATH)
    xgb = joblib.load(_XGB_PATH)

    try:
        explainer = load_explainer()
    except FileNotFoundError as exc:
        return None, None, None, None, str(exc)

    # Reconstruct the canonical test split (same seed as training) and derive
    # the operating threshold: highest threshold still meeting ≥60% recall.
    from src.data_loader import load_raw
    df = load_raw()
    df_clean = add_features(clean(df))
    X = df_clean.drop(columns=["default"])
    y = df_clean["default"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_t = ct.transform(X_test)
    threshold = find_recall_threshold(xgb, X_test_t, y_test, min_recall=0.60)

    return ct, xgb, explainer, threshold, None


def _preprocess_row(row_dict: dict, ct) -> np.ndarray:
    """Apply the training pipeline to a single applicant dict: clean → features → CT."""
    df = pd.DataFrame([row_dict])
    df = clean(df)
    df = add_features(df)
    return ct.transform(df)


def _risk_band(prob: float, threshold: float) -> tuple[str, str]:
    """Map a default probability to (band_label, hex_color)."""
    if prob >= threshold:
        return "HIGH RISK", "#e74c3c"
    elif prob >= threshold / 2:
        return "MEDIUM RISK", "#f39c12"
    else:
        return "LOW RISK", "#27ae60"


def _sidebar(threshold: float) -> None:
    """Render the diagnostics sidebar with global SHAP, ROC/PR, and calibration images."""
    st.sidebar.title("Model Diagnostics")
    st.sidebar.caption(
        f"Operating threshold: **{threshold:.3f}**\n\n"
        "*(highest threshold still achieving ≥60% recall on the held-out test set)*"
    )

    for title, fname in [
        ("Global SHAP Importance", "shap_global.png"),
        ("ROC / Precision-Recall Curves", "roc_pr.png"),
        ("Calibration (Reliability Diagram)", "calibration.png"),
    ]:
        path = os.path.join(_DATA_DIR, fname)
        if os.path.exists(path):
            st.sidebar.subheader(title)
            st.sidebar.image(path, use_container_width=True)


def _input_form() -> tuple[dict, bool]:
    """Render the applicant input form. Returns (row_dict, was_submitted)."""
    with st.form("applicant_form"):
        st.subheader("Demographics & Credit Limit")
        c1, c2, c3, c4, c5 = st.columns(5)
        limit_bal = c1.number_input(
            "Credit Limit (NTD)", min_value=10_000, max_value=1_000_000,
            value=200_000, step=10_000,
        )
        age = c2.number_input("Age", min_value=21, max_value=79, value=35)
        sex = c3.selectbox(
            "Sex", options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )
        education = c4.selectbox(
            "Education", options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate school", 2: "University",
                3: "High school",    4: "Other",
            }[x],
        )
        marriage = c5.selectbox(
            "Marital Status", options=[1, 2, 3],
            format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
        )

        st.subheader("Repayment Status")
        st.caption(
            "Sep = most recent month (PAY_0). "
            "−2 = no consumption · −1 = paid on time · 0 = revolving · 1–9 = months past due"
        )
        p0, p2, p3, p4, p5, p6 = st.columns(6)
        pay_0 = p0.selectbox("PAY_0 (Sep)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])
        pay_2 = p2.selectbox("PAY_2 (Aug)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])
        pay_3 = p3.selectbox("PAY_3 (Jul)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])
        pay_4 = p4.selectbox("PAY_4 (Jun)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])
        pay_5 = p5.selectbox("PAY_5 (May)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])
        pay_6 = p6.selectbox("PAY_6 (Apr)", _PAY_VALUES, index=1,
                              format_func=lambda x: _PAY_OPTIONS[x])

        st.subheader("Bill Statement Amounts (NTD, Sep → Apr)")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        bill1 = b1.number_input("BILL_AMT1 (Sep)", value=50_000, step=1_000)
        bill2 = b2.number_input("BILL_AMT2 (Aug)", value=48_000, step=1_000)
        bill3 = b3.number_input("BILL_AMT3 (Jul)", value=46_000, step=1_000)
        bill4 = b4.number_input("BILL_AMT4 (Jun)", value=44_000, step=1_000)
        bill5 = b5.number_input("BILL_AMT5 (May)", value=42_000, step=1_000)
        bill6 = b6.number_input("BILL_AMT6 (Apr)", value=40_000, step=1_000)

        st.subheader("Payment Amounts (NTD, Sep → Apr)")
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        pay_a1 = a1.number_input("PAY_AMT1 (Sep)", value=2_000, step=500, min_value=0)
        pay_a2 = a2.number_input("PAY_AMT2 (Aug)", value=2_000, step=500, min_value=0)
        pay_a3 = a3.number_input("PAY_AMT3 (Jul)", value=2_000, step=500, min_value=0)
        pay_a4 = a4.number_input("PAY_AMT4 (Jun)", value=2_000, step=500, min_value=0)
        pay_a5 = a5.number_input("PAY_AMT5 (May)", value=2_000, step=500, min_value=0)
        pay_a6 = a6.number_input("PAY_AMT6 (Apr)", value=2_000, step=500, min_value=0)

        submitted = st.form_submit_button(
            "Score Applicant", type="primary", use_container_width=True
        )

    row_dict = {
        "LIMIT_BAL": int(limit_bal), "SEX": int(sex),
        "EDUCATION": int(education), "MARRIAGE": int(marriage), "AGE": int(age),
        "PAY_0": int(pay_0), "PAY_2": int(pay_2), "PAY_3": int(pay_3),
        "PAY_4": int(pay_4), "PAY_5": int(pay_5), "PAY_6": int(pay_6),
        "BILL_AMT1": int(bill1), "BILL_AMT2": int(bill2), "BILL_AMT3": int(bill3),
        "BILL_AMT4": int(bill4), "BILL_AMT5": int(bill5), "BILL_AMT6": int(bill6),
        "PAY_AMT1": int(pay_a1), "PAY_AMT2": int(pay_a2), "PAY_AMT3": int(pay_a3),
        "PAY_AMT4": int(pay_a4), "PAY_AMT5": int(pay_a5), "PAY_AMT6": int(pay_a6),
    }
    return row_dict, submitted


def _show_results(prob: float, threshold: float, result: dict) -> None:
    """Render prediction metrics, adverse-action reasons, and SHAP waterfall."""
    band, color = _risk_band(prob, threshold)

    st.divider()
    st.subheader("Prediction Results")

    col_prob, col_band, col_thr = st.columns(3)
    col_prob.metric("Default Probability", f"{prob:.1%}")
    with col_band:
        st.markdown("**Risk Band**")
        st.markdown(
            f"<h2 style='color:{color}; margin-top:0;'>{band}</h2>",
            unsafe_allow_html=True,
        )
    col_thr.metric(
        "Operating Threshold", f"{threshold:.3f}",
        help="Highest threshold achieving ≥60% recall on the held-out test set",
    )

    if result["adverse_action"]:
        st.subheader("Adverse-Action Reasons")
        st.caption(
            "Top factors pushing this applicant's score toward default. "
            "Required under ECOA/FCRA for any adverse credit decision."
        )
        for i, reason in enumerate(result["adverse_action"], 1):
            st.write(f"**{i}.** {reason}")

    st.subheader("SHAP Waterfall — Per-Applicant Explanation")
    st.caption(
        "Red bars increase default risk · Blue bars decrease default risk · "
        "E[f(X)] = model's average output over the training population"
    )
    fig = plot_waterfall(result, max_display=15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(
        page_title="CreditLens — Credit Default Risk Scorer",
        page_icon="🔍",
        layout="wide",
    )

    ct, xgb, explainer, threshold, err = load_models()

    if err:
        st.error(
            "**Model artifacts not found.** Run the training pipeline first:\n\n"
            "```bash\npython -m src.train\npython -m src.explain\n```\n\n"
            f"Details: {err}"
        )
        st.stop()

    _sidebar(threshold)

    st.title("CreditLens — Credit Default Risk Scorer")
    st.markdown(
        "Enter borrower attributes and click **Score Applicant**. "
        "The XGBoost model returns a **default probability**, a **risk band**, "
        "and a **SHAP waterfall** explaining which features drove the score. "
        "Model diagnostics (global SHAP, ROC/PR, calibration) are in the sidebar."
    )

    row_dict, submitted = _input_form()

    if not submitted:
        st.info(
            "Fill in the borrower attributes above and click **Score Applicant** "
            "to see the prediction and explanation."
        )
        return

    X_row = _preprocess_row(row_dict, ct)
    prob = float(xgb.predict_proba(X_row)[0, 1])
    feature_names = get_feature_names(ct)
    result = explain_one(explainer, X_row[0], feature_names)
    _show_results(prob, threshold, result)


if __name__ == "__main__":
    main()
