"""
Engineered features grounded in credit-risk domain knowledge.

All four transforms are purely row-level (no statistics shared across rows),
so they can be applied to training and inference data identically — no
leakage risk.

Features
--------
utilization       : BILL_AMT1 / LIMIT_BAL — current credit utilisation ratio.
avg_pay_ratio     : mean(PAY_AMT_i / |BILL_AMT_i|) — payment consistency.
months_delinquent : count of PAY_* > 0 — number of months the borrower was late.
bill_trend        : OLS slope of BILL_AMT1..6 / LIMIT_BAL — rising vs falling balance.
"""

import numpy as np
import pandas as pd

BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

ENGINEERED_COLS = ["utilization", "avg_pay_ratio", "months_delinquent", "bill_trend"]


def _utilization(df: pd.DataFrame) -> pd.Series:
    """BILL_AMT1 / LIMIT_BAL — most-recent credit utilisation ratio.

    Clipped to [0, 5]: negative bills (credit refunds) map to 0; rare
    over-limit balances are capped rather than left as extreme outliers.
    High utilisation (>80 %) is a strong predictor of financial stress.
    """
    return (df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)).clip(0, 5)


def _avg_pay_ratio(df: pd.DataFrame) -> pd.Series:
    """Mean (PAY_AMT_i / |BILL_AMT_i|) across all 6 reporting months.

    ~1.0 means the borrower pays in full each month; ~0.0 means they
    make minimum or no payments.  Division uses the absolute value of
    the bill to handle credit-balance months (negative BILL_AMT).  Each
    monthly ratio is capped at 2.0 to suppress extreme values when the
    bill is very small (e.g., a $1 statement).
    """
    ratios = []
    for pay_col, bill_col in zip(PAY_AMT_COLS, BILL_COLS):
        denom = df[bill_col].abs().clip(lower=1)   # avoid division by zero
        ratio = (df[pay_col] / denom).clip(0, 2.0)
        ratios.append(ratio)
    return pd.concat(ratios, axis=1).mean(axis=1)


def _months_delinquent(df: pd.DataFrame) -> pd.Series:
    """Number of months (0-6) where repayment status > 0 (past due).

    PAY_* encoding: -2=no consumption, -1=paid duly, 0=revolving credit,
    1-9=N months past due.  Any value >= 1 counts as a delinquency event.
    """
    return (df[PAY_STATUS_COLS] > 0).sum(axis=1).astype(float)


def _bill_trend(df: pd.DataFrame) -> pd.Series:
    """Linear (OLS) slope of outstanding balance over 6 months, scaled by limit.

    Positive  -> balance growing month-over-month (taking on more debt).
    Negative  -> balance shrinking (paying down the card).
    Dividing by LIMIT_BAL + 1 makes the feature comparable across borrowers
    with different credit lines and avoids extreme values on high-limit cards.

    Implementation: centred OLS via analytical formula slope = (x . y) / (x . x)
    with x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] for numerical stability.
    """
    bills = df[BILL_COLS].values.astype(float)
    x = np.arange(6, dtype=float) - 2.5          # centred month indices
    x_var = float(x @ x)                          # sum of squared deviations
    slopes = (bills @ x) / x_var                  # per-row OLS slope
    return pd.Series(slopes / (df["LIMIT_BAL"].values + 1), index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all four engineered columns to `df`.

    Returns a new DataFrame (original is not modified).  Safe to call on
    train and test sets independently — all transforms are row-level only.
    """
    df = df.copy()
    df["utilization"] = _utilization(df)
    df["avg_pay_ratio"] = _avg_pay_ratio(df)
    df["months_delinquent"] = _months_delinquent(df)
    df["bill_trend"] = _bill_trend(df)
    return df


def verify_no_leakage() -> None:
    """Assert engineered features are identical whether computed alone or together.

    Creates two synthetic rows, computes features on the pair and on each row
    in isolation, and checks that results agree to floating-point precision.
    Any divergence would indicate cross-row state (a leakage bug).
    Prints a confirmation message on success.
    """
    row_a = {
        "LIMIT_BAL": 50_000,
        "BILL_AMT1": 20_000, "BILL_AMT2": 18_000, "BILL_AMT3": 22_000,
        "BILL_AMT4": 19_000, "BILL_AMT5": 21_000, "BILL_AMT6": 17_000,
        "PAY_AMT1": 5_000, "PAY_AMT2": 4_000, "PAY_AMT3": 6_000,
        "PAY_AMT4": 3_000, "PAY_AMT5": 5_000, "PAY_AMT6": 4_500,
        "PAY_0": 0, "PAY_2": 1, "PAY_3": 0, "PAY_4": 0, "PAY_5": 1, "PAY_6": 0,
    }
    row_b = {
        "LIMIT_BAL": 120_000,
        "BILL_AMT1": 5_000, "BILL_AMT2": 6_000, "BILL_AMT3": 5_500,
        "BILL_AMT4": 4_800, "BILL_AMT5": 5_200, "BILL_AMT6": 5_000,
        "PAY_AMT1": 5_000, "PAY_AMT2": 6_000, "PAY_AMT3": 5_500,
        "PAY_AMT4": 4_800, "PAY_AMT5": 5_200, "PAY_AMT6": 5_000,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    }

    together = add_features(pd.DataFrame([row_a, row_b]))
    alone_a = add_features(pd.DataFrame([row_a]))
    alone_b = add_features(pd.DataFrame([row_b]))

    for col in ENGINEERED_COLS:
        assert abs(together[col].iloc[0] - alone_a[col].iloc[0]) < 1e-12, (
            f"Leakage detected in {col}: row A differs when computed alongside row B"
        )
        assert abs(together[col].iloc[1] - alone_b[col].iloc[0]) < 1e-12, (
            f"Leakage detected in {col}: row B differs when computed alongside row A"
        )

    print("Leakage check PASSED — all engineered features are row-level only.")


if __name__ == "__main__":
    verify_no_leakage()
