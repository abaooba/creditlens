"""
Preprocessing pipeline for the UCI credit card default dataset.

Responsibilities:
- Collapse undocumented EDUCATION (0,5,6) and MARRIAGE (0) categories to "other"
- Treat PAY_* repayment-status columns as ordered ordinal (months past due)
- Build a stratified 80/20 train/test split with no leakage:
  scalers and encoders are fit on the training set only, then applied to test
- Return X_train, X_test, y_train, y_test, and the fitted ColumnTransformer
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
_PREPROCESS_PATH = os.path.join(_MODELS_DIR, "preprocess.joblib")

# PAY_* columns represent ordered repayment-status codes
# -2=no consumption, -1=pay duly, 0=revolving credit, 1–9=months past due
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# Possible ordered categories across all six repayment columns
PAY_CATEGORIES = [[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * len(PAY_COLS)

CONTINUOUS_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Nominal categoricals kept as integers after collapse
NOMINAL_INT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse undocumented categories in EDUCATION and MARRIAGE.

    EDUCATION: values 0, 5, 6 are undocumented; map them to 4 ("others").
    MARRIAGE:  value 0 is undocumented; map to 3 ("others").
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def build_split(
    df: pd.DataFrame = None,
    test_size: float = 0.2,
    random_state: int = 42,
    persist: bool = True,
):
    """Stratified 80/20 split + fit ColumnTransformer on train only.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Raw dataframe from load_raw(). If None, loads it automatically.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Reproducibility seed.
    persist : bool
        If True, save the fitted transformer to models/preprocess.joblib.

    Returns
    -------
    X_train, X_test, y_train, y_test, ct
        Where `ct` is the fitted sklearn ColumnTransformer.
    """
    if df is None:
        from src.data_loader import load_raw
        df = load_raw()

    df = clean(df)

    X = df.drop(columns=["default"])
    y = df["default"]

    # Stratified split — preserves ~22% default rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ColumnTransformer: scale continuous, encode PAY_* as ordinal
    # NOMINAL_INT_COLS (SEX, EDUCATION, MARRIAGE) are passed through as-is
    ct = ColumnTransformer(
        transformers=[
            (
                "continuous",
                StandardScaler(),
                CONTINUOUS_COLS,
            ),
            (
                "pay_ordinal",
                OrdinalEncoder(
                    categories=PAY_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                PAY_COLS,
            ),
            (
                "nominal_passthrough",
                "passthrough",
                NOMINAL_INT_COLS,
            ),
        ],
        remainder="drop",
    )

    # Fit on train only — this is the leakage-prevention guarantee
    ct.fit(X_train)
    X_train_t = ct.transform(X_train)
    X_test_t = ct.transform(X_test)

    if persist:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(ct, _PREPROCESS_PATH)

    return X_train_t, X_test_t, y_train, y_test, ct


def get_feature_names(ct: ColumnTransformer) -> list[str]:
    """Return ordered column names matching the transformer output."""
    return CONTINUOUS_COLS + PAY_COLS + NOMINAL_INT_COLS
