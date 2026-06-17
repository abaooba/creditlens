"""
Model training for CreditLens.

train_logreg() — interpretable baseline: logistic regression with
                 class_weight="balanced" trained on the Phase-2 preprocessed features.
train_xgb()    — gradient-boosted model with scale_pos_weight (added Day 6).

Both functions consume the output of build_split(), train on the preprocessed
feature matrix, and persist the fitted classifier to models/.
The companion preprocessor lives in models/preprocess.joblib.
"""

import os
import joblib

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_MODELS_DIR = os.path.join(_REPO_DIR, "models")
_LOGREG_PATH = os.path.join(_MODELS_DIR, "logreg.joblib")


def train_logreg(random_state: int = 42, persist: bool = True):
    """Train logistic regression baseline on the preprocessed credit feature matrix.

    Uses class_weight="balanced" so sklearn automatically scales the per-sample
    loss to give equal total weight to the default (minority, ~22%) and non-default
    (majority, ~78%) classes.  Without this, the classifier can "cheat" by always
    predicting non-default and still achieve 78% accuracy.

    The logistic regression is the interpretable baseline — its coefficients map
    directly to log-odds changes per unit feature, which is explainable to
    regulators.  XGBoost (Day 6) trades that interpretability for higher PR-AUC.

    Parameters
    ----------
    random_state : int
        Seed passed to LogisticRegression for reproducibility.
    persist : bool
        If True, save the fitted model to models/logreg.joblib.

    Returns
    -------
    lr : sklearn.linear_model.LogisticRegression
        Fitted classifier.
    split : tuple
        (X_train, X_test, y_train, y_test, ct) from build_split().
    metrics : dict
        ROC-AUC and PR-AUC on the held-out test set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    from src.data_loader import load_raw
    from src.preprocess import build_split

    df = load_raw()
    X_train, X_test, y_train, y_test, ct = build_split(df, random_state=random_state)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        C=1.0,
    )
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
    }
    print(f"LogReg — ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")

    if persist:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(lr, _LOGREG_PATH)
        print(f"Persisted → {_LOGREG_PATH}")

    return lr, (X_train, X_test, y_train, y_test, ct), metrics


if __name__ == "__main__":
    lr, split, metrics = train_logreg()
    print("Training complete.")
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  PR-AUC  : {metrics['pr_auc']}")
