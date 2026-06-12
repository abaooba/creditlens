import os
import pandas as pd

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_RAW_CSV = os.path.join(_DATA_DIR, "raw.csv")


def load_raw() -> pd.DataFrame:
    """Fetch UCI Default of Credit Card Clients dataset (id=350).

    First call downloads via ucimlrepo and caches to data/raw.csv.
    Subsequent calls load from the cache (fast).

    Returns
    -------
    pd.DataFrame
        23 feature columns (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0..PAY_6, BILL_AMT1..6, PAY_AMT1..6) plus target column
        'default' (1 = defaulted next month, 0 = did not).
        Shape: (30000, 24).
    """
    os.makedirs(_DATA_DIR, exist_ok=True)

    if os.path.exists(_RAW_CSV):
        return pd.read_csv(_RAW_CSV)

    from ucimlrepo import fetch_ucirepo  # deferred so import succeeds without network

    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # UCI ships the target as 'Y'; normalise to 'default'
    target_col = y.columns[0]
    if target_col != "default":
        df = df.rename(columns={target_col: "default"})

    df.to_csv(_RAW_CSV, index=False)
    return df
