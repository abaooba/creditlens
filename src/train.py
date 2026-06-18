"""
Model training for CreditLens.

train_logreg() — interpretable baseline: logistic regression with
                 class_weight="balanced" trained on the Phase-2 preprocessed features.
train_xgb()    — gradient-boosted model with scale_pos_weight and CV search on PR-AUC.

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
_XGB_PATH = os.path.join(_MODELS_DIR, "xgb.joblib")


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


def train_xgb(random_state: int = 42, persist: bool = True):
    """Train XGBoost with scale_pos_weight and CV hyperparameter search on PR-AUC.

    scale_pos_weight is set to the exact negative/positive class ratio from the
    training set (~3.5×), which is XGBoost's native mechanism for imbalanced classes.
    Unlike sklearn's class_weight="balanced", this scales gradient contributions from
    positive (default) samples rather than resampling, making training more efficient.

    A small grid (8 configs: 2 depths × 2 learning rates × 2 n_estimators) is
    evaluated via stratified 3-fold CV scored by average_precision (= PR-AUC).
    The final model is re-fit on the full training set with the best parameters.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.
    persist : bool
        If True, save the fitted model to models/xgb.joblib.

    Returns
    -------
    xgb_model : xgboost.XGBClassifier
        Fitted classifier with best hyperparameters.
    split : tuple
        (X_train, X_test, y_train, y_test, ct) from build_split().
    metrics : dict
        ROC-AUC, PR-AUC on the held-out test set, best_cv_pr_auc, and best_params.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier

    from src.data_loader import load_raw
    from src.preprocess import build_split

    df = load_raw()
    X_train, X_test, y_train, y_test, ct = build_split(df, random_state=random_state)

    # scale_pos_weight = #negative / #positive
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg) / float(pos)
    print(f"scale_pos_weight = {spw:.3f}  (neg={neg}, pos={pos})")

    # 2 × 2 × 2 = 8 configurations
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params: dict = {}

    for max_depth in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for n_est in param_grid["n_estimators"]:
                clf = XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    scale_pos_weight=spw,
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                )
                scores = cross_val_score(
                    clf, X_train, y_train,
                    cv=cv,
                    scoring="average_precision",
                    n_jobs=-1,
                )
                mean_score = float(scores.mean())
                print(
                    f"  depth={max_depth} lr={lr} n={n_est} → "
                    f"CV PR-AUC={mean_score:.4f} ± {scores.std():.4f}"
                )
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                    }

    print(f"\nBest CV PR-AUC: {best_score:.4f}  params: {best_params}")

    # Re-fit on the full training set with best parameters
    xgb_model = XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "best_cv_pr_auc": round(best_score, 4),
        "best_params": best_params,
    }
    print(
        f"XGBoost — ROC-AUC: {metrics['roc_auc']:.4f}  "
        f"PR-AUC: {metrics['pr_auc']:.4f}"
    )

    if persist:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(xgb_model, _XGB_PATH)
        print(f"Persisted → {_XGB_PATH}")

    return xgb_model, (X_train, X_test, y_train, y_test, ct), metrics


if __name__ == "__main__":
    print("=== Logistic Regression Baseline ===")
    lr, split, lr_metrics = train_logreg()
    print(f"  ROC-AUC : {lr_metrics['roc_auc']}")
    print(f"  PR-AUC  : {lr_metrics['pr_auc']}")

    print("\n=== XGBoost Model ===")
    xgb, split, xgb_metrics = train_xgb()
    print(f"  ROC-AUC : {xgb_metrics['roc_auc']}")
    print(f"  PR-AUC  : {xgb_metrics['pr_auc']}")
    print(f"  Best CV PR-AUC : {xgb_metrics['best_cv_pr_auc']}")
    print(f"  Best params    : {xgb_metrics['best_params']}")
