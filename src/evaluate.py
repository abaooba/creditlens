"""
Evaluation suite for CreditLens.

Functions
---------
evaluate()              - ROC-AUC, PR-AUC, recall, precision, F1, Brier at a threshold.
find_recall_threshold() - highest threshold on the PR curve still hitting min_recall.
plot_roc_pr()           - side-by-side ROC and Precision-Recall curves (both models).
plot_confusion()        - confusion-matrix heatmap at the operating threshold.
plot_calibration()      - reliability diagram + Brier scores for both models.

Threshold philosophy
--------------------
For credit default scoring, missing a defaulter (false negative) is the costlier
error — unpaid debt is worse than a rejected good customer. The operating threshold
is therefore chosen from the PR curve as the highest value that still achieves a
stated recall floor (default: 60%). This maximises precision subject to the recall
constraint, rather than splitting errors evenly at 0.5.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display server required
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_DATA_DIR = os.path.join(_REPO_DIR, "data")
_MODELS_DIR = os.path.join(_REPO_DIR, "models")

_DEFAULT_RECALL_TARGET = 0.60


def find_recall_threshold(
    model, X_test, y_test, min_recall: float = _DEFAULT_RECALL_TARGET
) -> float:
    """Return the highest decision threshold still achieving min_recall on the test set.

    Scans precision_recall_curve (which orders thresholds from lowest to highest,
    recall from highest to lowest) and tracks the last threshold at which recall
    is still >= min_recall. That is the maximum-precision point satisfying the
    recall floor — the business-optimal operating point for recall-leaning models.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X_test, y_test : preprocessed test features and true labels
    min_recall : float
        Minimum acceptable recall (fraction of actual defaulters caught).
        Default 0.60 = catch at least 60% of true defaults.

    Returns
    -------
    float : selected decision threshold
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    _, recall_arr, thresholds = precision_recall_curve(y_test, y_prob)
    # recall_arr[-1] == 0.0 and has no corresponding threshold entry
    # thresholds is sorted ascending; recall_arr is sorted descending
    best_threshold = float(thresholds[0])  # fallback: lowest threshold = max recall
    for rec, thr in zip(recall_arr[:-1], thresholds):
        if rec >= min_recall:
            best_threshold = float(thr)  # keep updating — want the LAST (highest) one
    return best_threshold


def evaluate(model, X_test, y_test, threshold: float = None) -> dict:
    """Compute classification metrics at a decision threshold.

    If threshold is None, find_recall_threshold() is called to auto-select
    the highest threshold that still meets the 60% recall floor.

    Parameters
    ----------
    model : fitted sklearn/XGBoost classifier with predict_proba
    X_test : preprocessed feature matrix (n_samples, n_features)
    y_test : true binary labels (1 = default)
    threshold : float, optional
        Decision threshold for the positive class. Auto-selected if None.

    Returns
    -------
    dict with keys:
        roc_auc, pr_auc, threshold, recall, precision, f1, brier
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    if threshold is None:
        threshold = find_recall_threshold(model, X_test, y_test)

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": round(threshold, 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "brier": round(brier, 4),
    }


def plot_roc_pr(
    models: dict,
    X_test,
    y_test,
    save_path: str = None,
) -> str:
    """Plot side-by-side ROC and Precision-Recall curves for a dict of models.

    Parameters
    ----------
    models : dict
        {label: fitted_model}, e.g. {"LogReg": lr, "XGBoost": xgb}.
    X_test, y_test : test set
    save_path : str, optional
        Output file path. Defaults to data/roc_pr.png.

    Returns
    -------
    str : path to the saved figure
    """
    if save_path is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        save_path = os.path.join(_DATA_DIR, "roc_pr.png")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for (label, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{label} (AUC = {roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{label} (AP = {pr_auc:.3f})")

    # ROC diagonal reference
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])

    # PR baseline = class prevalence (random classifier)
    prevalence = float(np.mean(y_test))
    axes[1].axhline(prevalence, color="k", linestyle="--", lw=1, alpha=0.4,
                    label=f"No-skill baseline (prev = {prevalence:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    fig.suptitle("CreditLens — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion(
    model,
    X_test,
    y_test,
    threshold: float = None,
    label: str = "Model",
    save_path: str = None,
) -> str:
    """Plot a confusion matrix heatmap at the given decision threshold.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X_test, y_test : test set
    threshold : float, optional
        Decision threshold. Auto-selected if None.
    label : str
        Model name for the figure title.
    save_path : str, optional
        Output file path. Defaults to data/confusion_<label>.png.

    Returns
    -------
    str : path to the saved figure
    """
    if threshold is None:
        threshold = find_recall_threshold(model, X_test, y_test)

    if save_path is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        safe = label.lower().replace(" ", "_")
        save_path = os.path.join(_DATA_DIR, f"confusion_{safe}.png")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["No Default (0)", "Default (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    mid = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > mid else "black")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{label} — Confusion Matrix\nthreshold = {threshold:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_calibration(
    models: dict,
    X_test,
    y_test,
    n_bins: int = 10,
    save_path: str = None,
) -> dict:
    """Plot reliability (calibration) curves and return Brier scores for each model.

    A calibrated model's reliability curve lies close to the diagonal: when the
    model outputs a 30% default probability, about 30% of those applicants should
    actually default. Lenders care about calibration because they use the raw
    probability to price risk — a systematically biased model misprices loans.

    Brier score = mean((predicted_prob - true_label)^2). Lower is better;
    the no-skill baseline equals the class prevalence * (1 - prevalence) ≈ 0.172.

    Parameters
    ----------
    models : dict
        {label: fitted_model}.
    X_test, y_test : test set
    n_bins : int
        Number of probability bins for the reliability diagram.
    save_path : str, optional
        Output file path. Defaults to data/calibration.png.

    Returns
    -------
    dict : {label: brier_score}
    """
    if save_path is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        save_path = os.path.join(_DATA_DIR, "calibration.png")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    brier_scores = {}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")

    for (label, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        brier = float(brier_score_loss(y_test, y_prob))
        brier_scores[label] = round(brier, 4)

        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(
            prob_pred, prob_true, "o-",
            color=color, lw=2, markersize=6,
            label=f"{label} (Brier = {brier:.4f})",
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed Default Rate)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return brier_scores


if __name__ == "__main__":
    import joblib
    from src.data_loader import load_raw
    from src.preprocess import build_split
    from src.train import train_logreg, train_xgb

    logreg_path = os.path.join(_MODELS_DIR, "logreg.joblib")
    xgb_path = os.path.join(_MODELS_DIR, "xgb.joblib")

    df = load_raw()
    _, X_test, _, y_test, _ = build_split(df)

    if os.path.exists(logreg_path):
        lr = joblib.load(logreg_path)
        print(f"Loaded LogReg  ← {logreg_path}")
    else:
        print("Training LogReg (no persisted model found)...")
        lr, _, _ = train_logreg()

    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)
        print(f"Loaded XGBoost ← {xgb_path}")
    else:
        print("Training XGBoost (no persisted model found)...")
        xgb, _, _ = train_xgb()

    models = {"LogReg": lr, "XGBoost": xgb}

    print("\n=== Threshold Selection (recall floor ≥60%) ===")
    lr_thr = find_recall_threshold(lr, X_test, y_test, min_recall=0.60)
    xgb_thr = find_recall_threshold(xgb, X_test, y_test, min_recall=0.60)
    print(f"  LogReg  threshold : {lr_thr:.4f}")
    print(f"  XGBoost threshold : {xgb_thr:.4f}")

    print("\n=== Test-Set Metrics ===")
    lr_m = evaluate(lr, X_test, y_test, threshold=lr_thr)
    xgb_m = evaluate(xgb, X_test, y_test, threshold=xgb_thr)

    header = f"{'Model':12s}  ROC-AUC  PR-AUC  threshold  recall  precision    F1  Brier"
    print(header)
    print("-" * len(header))
    for name, m in [("LogReg", lr_m), ("XGBoost", xgb_m)]:
        print(
            f"{name:12s}  {m['roc_auc']:.4f}  {m['pr_auc']:.4f}  "
            f"{m['threshold']:.4f}     {m['recall']:.4f}  {m['precision']:.4f}  "
            f"{m['f1']:.4f}  {m['brier']:.4f}"
        )

    print("\n=== Generating Plots ===")
    p = plot_roc_pr(models, X_test, y_test)
    print(f"  ROC/PR      → {p}")

    p = plot_confusion(lr, X_test, y_test, threshold=lr_thr, label="LogReg")
    print(f"  Confusion LR  → {p}")

    p = plot_confusion(xgb, X_test, y_test, threshold=xgb_thr, label="XGBoost")
    print(f"  Confusion XGB → {p}")

    brier = plot_calibration(models, X_test, y_test)
    print(f"  Calibration → {os.path.join(_DATA_DIR, 'calibration.png')}")
    print(f"  Brier scores: {brier}")

    print("\nAll evaluation artefacts written to data/")
