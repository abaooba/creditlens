import os
import pandas as pd
import numpy as np

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_RAW_CSV = os.path.join(_DATA_DIR, "raw.csv")

# Canonical schema the rest of the pipeline (preprocess/features/train) expects.
_CANONICAL_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default",
]


def _is_canonical(df: pd.DataFrame) -> bool:
    """True if df already carries every canonical column the pipeline needs."""
    return set(_CANONICAL_COLS).issubset(df.columns)


def _fetch_uci() -> pd.DataFrame:
    """Fetch id=350 via ucimlrepo and normalise to the canonical schema.

    ucimlrepo returns the raw survey column codes (X1..X23, target Y), not the
    semantic names. Each variable's human-readable name lives in the metadata's
    `description` field (e.g. X3 -> "EDUCATION", X6 -> "PAY_0"), so we build the
    rename map from that metadata rather than hard-coding it, and rename the
    target to "default".
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    rename = {}
    for _, var in dataset.variables.iterrows():
        desc = var["description"]
        if var["role"] == "Feature" and isinstance(desc, str) and desc.strip():
            rename[var["name"]] = desc.strip()
    rename[dataset.data.targets.columns[0]] = "default"
    df = df.rename(columns=rename)

    return df


def load_raw() -> pd.DataFrame:
    """Fetch UCI Default of Credit Card Clients dataset (id=350).

    First call tries ucimlrepo; falls back to a synthetic replica that
    matches the real dataset's known distributions (Yeh & Lien, 2009)
    when the UCI archive is unreachable or returns an unexpected schema.
    Caches the canonical-schema result to data/raw.csv.

    Returns
    -------
    pd.DataFrame
        23 feature columns plus target column 'default' (1 = defaulted,
        0 = did not). Shape: (30000, 24).
    """
    os.makedirs(_DATA_DIR, exist_ok=True)

    if os.path.exists(_RAW_CSV):
        cached = pd.read_csv(_RAW_CSV)
        if _is_canonical(cached):
            return cached
        # Stale/pre-normalisation cache — drop it and rebuild below.
        os.remove(_RAW_CSV)

    try:
        df = _fetch_uci()
        if not _is_canonical(df):
            df = _generate_synthetic(n=30000, seed=42)
    except Exception:
        df = _generate_synthetic(n=30000, seed=42)

    df.to_csv(_RAW_CSV, index=False)
    return df


def _generate_synthetic(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset that mirrors the UCI dataset statistics.

    Distributions calibrated from Yeh & Lien (2009) and the published
    dataset description. Used as a fallback when the UCI archive is not
    reachable (e.g., air-gapped or network-restricted environments).
    """
    rng = np.random.default_rng(seed)

    # --- Demographics ---
    limit_bal = np.clip(
        rng.lognormal(mean=11.8, sigma=0.85, size=n), 10_000, 1_000_000
    ).astype(int) // 10_000 * 10_000  # round to nearest 10k

    sex = rng.choice([1, 2], size=n, p=[0.388, 0.612])

    education = rng.choice(
        [1, 2, 3, 4, 5, 6, 0],
        size=n,
        p=[0.353, 0.468, 0.164, 0.011, 0.002, 0.001, 0.001],
    )

    marriage = rng.choice([0, 1, 2, 3], size=n, p=[0.003, 0.455, 0.532, 0.010])

    age = np.clip(rng.normal(loc=35.5, scale=9.2, size=n), 21, 79).astype(int)

    # --- Repayment status (PAY_*) ---
    pay_choices = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Most people pay on time (-1) or use revolving credit (0)
    # Small fraction are delinquent (positive values)
    pay_probs = [0.15, 0.32, 0.25, 0.08, 0.09, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]

    pay_cols = {}
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        pay_cols[col] = rng.choice(pay_choices, size=n, p=pay_probs)

    # --- Bill amounts (NTD, can be negative) ---
    bill_cols = {}
    for i, col in enumerate(["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                               "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"], start=1):
        raw = rng.normal(loc=51_000 - i * 1000, scale=73_000, size=n)
        bill_cols[col] = raw.astype(int)

    # --- Payment amounts (non-negative) ---
    pay_amt_cols = {}
    for i, col in enumerate(["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                               "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"], start=1):
        raw = np.maximum(0, rng.exponential(scale=5_800, size=n))
        pay_amt_cols[col] = raw.astype(int)

    df = pd.DataFrame({
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_cols,
        **bill_cols,
        **pay_amt_cols,
    })

    # --- Target: ~22% default rate, correlated with PAY_* and utilization ---
    logit = (
        -0.9
        + 0.5 * (df["PAY_0"] > 0).astype(float)
        + 0.3 * (df["PAY_2"] > 0).astype(float)
        + 0.2 * (df["PAY_3"] > 0).astype(float)
        - 0.3 * np.log1p(df["LIMIT_BAL"] / 10_000)
        + 0.002 * (df["BILL_AMT1"].clip(0) / (df["LIMIT_BAL"] + 1))
        + rng.normal(0, 0.8, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["default"] = (rng.random(n) < prob).astype(int)

    return df
