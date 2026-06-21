# CreditLens — ML Credit Default Risk Scorer

End-to-end machine-learning system that predicts credit card default probability and explains each decision via SHAP. Built on 30,000 real cardholder records from UCI; no API key required.

## Problem
Given a credit card holder's repayment history, balance, and demographics, predict whether they will default next month — and explain *why*.

## Dataset
[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — 30,000 Taiwanese cardholders, 23 features.

### Key Data Facts
- **Class imbalance:** ~22% default rate. Accuracy is a misleading metric — a \"never default\" classifier achieves 78% accuracy while catching 0 actual defaulters. **PR-AUC and recall are the real evaluation metrics.**
- **Strongest predictors:** The six `PAY_*` repayment-status columns dominate. `PAY_0` (most recent month's repayment status) is the single strongest feature.
- **Data quirks:** `EDUCATION` values 0, 5, 6 and `MARRIAGE` value 0 are undocumented (absent from the original paper); collapsed to \"other\" during preprocessing.

## Engineered Features
Four domain-driven features are added before model training to capture credit-risk signals not explicit in the raw columns:

| Feature | Formula | Credit Meaning |
|---------|---------|----------------|
| `utilization` | `BILL_AMT1 / LIMIT_BAL` | Current credit utilisation — the #2 FICO factor |
| `avg_pay_ratio` | mean(`PAY_AMT_i / |BILL_AMT_i|`) | Payment consistency across 6 months |
| `months_delinquent` | count(`PAY_* > 0`) | How often the borrower was past due |
| `bill_trend` | OLS slope of `BILL_AMT1..6` / `LIMIT_BAL` | Rising vs. falling balance trajectory |

All features are computed row-by-row with no cross-row state — verified by `src/features.py:verify_no_leakage()`.

## Model Evaluation

### Threshold Selection
The default 0.5 threshold treats false positives and false negatives as equally costly — wrong for credit risk.
**Missing a defaulter** (false negative = unpaid debt) is the costlier error for a lender.

The operating threshold is therefore selected from the Precision-Recall curve as the **highest value
that still achieves ≥60% recall** (catching at least 6 in 10 true defaulters). This maximises
precision subject to the recall floor and is documented as a business decision in the model card.

### Performance (run `python src/evaluate.py` to compute on your dataset)

| Metric | Logistic Regression | XGBoost | Notes |
|--------|--------------------|---------| ------|
| ROC-AUC | — | — | Ranking quality; XGBoost expected ~0.77 on real UCI data |
| PR-AUC | — | — | Imbalanced-safe metric; null baseline = class prevalence (~0.22) |
| Recall @ threshold | ≥ 0.60 | ≥ 0.60 | Guaranteed by threshold selection |
| Precision @ threshold | — | — | XGBoost expected higher precision at same recall |
| F1 @ threshold | — | — | Harmonic mean of the above two |
| Brier Score | — | — | Calibration quality; lower = better probability estimates |

*Dashes are populated by running the evaluation script. On the UCI dataset, XGBoost is expected to
outperform logistic regression on PR-AUC by approximately 0.10–0.15 points.*

### Calibration
A model can rank borrowers correctly (good AUC) yet output miscalibrated probabilities — the
reliability diagram shows whether predicted probabilities match observed default rates bin by bin.
Lenders rely on calibrated probabilities to price loans; a model that says \"30% default risk\" should
observe ~30% defaults in that score band.

Plots saved to `data/` by `evaluate.py`:

| File | Contents |
|------|----------|
| `data/roc_pr.png` | Side-by-side ROC and PR curves for LogReg and XGBoost |
| `data/confusion_logreg.png` | Confusion matrix at LogReg operating threshold |
| `data/confusion_xgboost.png` | Confusion matrix at XGBoost operating threshold |
| `data/calibration.png` | Reliability diagram + Brier scores for both models |

## SHAP Explainability

CreditLens uses SHAP (SHapley Additive exPlanations) to make every XGBoost prediction transparent
— both globally (which features drive risk across the full borrower population) and locally (why
*this specific applicant* was scored as high risk).

### Why SHAP for Tree Models?

SHAP values have a rigorous game-theoretic foundation: each feature's SHAP value is its average
marginal contribution across all possible subsets of features — the unique allocation that satisfies
consistency, efficiency, and local accuracy simultaneously. For tree models, `shap.TreeExplainer`
computes *exact* SHAP values in polynomial time (not a Monte Carlo approximation), making per-
applicant explanations fast enough for production inference.

### Global Feature Importance: PAY_* Dominance

The SHAP beeswarm plot (`data/shap_global.png`) consistently shows that the six `PAY_*`
repayment-status columns — especially `PAY_0` (most recent month) — dominate the XGBoost model's
output. This aligns tightly with the economics of credit risk:

**Why recent late payments are the strongest default signal:**

- A borrower who missed last month's payment is already in financial distress. Missing a payment
  can signal a cash-flow crisis, an unsustainable debt load, or the early stages of strategic
  default — all of which dramatically raise next-month default risk.
- Recency matters: `PAY_0` carries far more predictive weight than `PAY_6` (six months ago)
  because recent behavior better reflects the borrower's *current* financial state. A missed
  payment last month is more alarming than one from half a year ago, even if the borrower
  recovered temporarily in between.
- This mirrors how human underwriters and credit-scoring bureaus operate: a recent derogatory
  event (a missed payment, a collections account) triggers an immediate score drop that a longer
  history of on-time payments cannot fully offset in the short run.
- The six `PAY_*` columns collectively encode the *trajectory* of repayment behaviour. A borrower
  sliding from PAY_6 = -1 (paid on time) to PAY_0 = 2 (two months past due) is on a worsening
  path that the model learns to weight heavily.

**Where the engineered features rank:**

- `months_delinquent` (count of PAY_* > 0) typically appears in the top half of the importance
  ranking. It captures *cumulative* delinquency — a borrower who was late in 4 of the past 6
  months is materially riskier than one who was late once, and this count feature makes that
  pattern explicit in a way individual PAY_* columns do not.
- `utilization` (BILL_AMT1 / LIMIT_BAL) ranks in the mid-tier. High utilisation signals a
  borrower relying heavily on revolving credit — a recognised financial-stress indicator. However,
  utilisation has lower predictive power than repayment status because a high-balance borrower who
  consistently pays in full is not a default risk; it's the *combination* of high balance *and*
  missed payments that's lethal.
- `avg_pay_ratio` and `bill_trend` provide incremental signal but are generally dominated by the
  repayment-status features. They matter most for borderline applicants where PAY_* status is
  ambiguous (e.g., revolving-credit code 0 throughout).

### Adverse-Action Framing

Under the Equal Credit Opportunity Act (ECOA) and the Fair Credit Reporting Act (FCRA), U.S.
lenders must provide rejected applicants with specific **adverse-action reasons** — the primary
factors that negatively affected their credit decision. SHAP's local explanations map directly
to this requirement.

For a high-risk applicant, `explain_one()` surfaces the top-3 features with the largest
*positive* SHAP values (features pushing the model's output toward \"default\") as plain-language
reason codes:

| Rank | Typical adverse-action reason | Underlying SHAP driver |
|------|-------------------------------|------------------------|
| 1 | Recent payment delinquency (most recent month) | PAY_0 high positive SHAP |
| 2 | High number of months with late payments | months_delinquent high positive SHAP |
| 3 | High credit utilisation ratio | utilization high positive SHAP |

This turns SHAP from a research interpretability tool into a legally defensible decision-support
system — exactly what distinguishes a production-grade credit model from a Kaggle notebook.

### SHAP Output Files

| File | Contents |
|------|----------|
| `data/shap_global.png` | Beeswarm (all borrowers) + mean-|SHAP| bar chart side by side |
| `data/shap_waterfall_sample.png` | Per-applicant waterfall chart for a sample test-set row |
| `models/shap_explainer.joblib` | Cached TreeExplainer loaded by `app.py` via `load_explainer()` for fast per-applicant scoring |

## Tech Stack
| Layer | Library |
|-------|----------|
| Data | `ucimlrepo`, `pandas`, `numpy` |
| ML | `scikit-learn`, `xgboost` |
| Explainability | `shap` |
| Visualization | `matplotlib`, `plotly` |
| App | `streamlit` |
| Persistence | `joblib` |

> All dependencies are open-source. No paid API, no account required.

## Project Structure
```
creditlens/
├── data/              # raw + processed data (gitignored — regenerated by data_loader.py)
├── models/            # serialized model artifacts (.joblib)
├── notebooks/
│   └── 01_eda.ipynb   # exploratory analysis
└── src/
    ├── data_loader.py # fetch + cache UCI dataset
    ├── preprocess.py  # cleaning, encoding, train/test split (27-feature matrix)
    ├── features.py    # four engineered credit-risk features
    ├── train.py       # fit logistic regression + xgboost
    ├── evaluate.py    # ROC/PR/calibration metrics + plots
    ├── explain.py     # SHAP global + local explanations
    └── app.py         # Streamlit scoring UI
```

## Running Locally

```bash
pip install -r requirements.txt

# Verify data pipeline (prints (30000, 24))
python -c "from src.data_loader import load_raw; print(load_raw().shape)"

# Verify engineered features are leak-free
python src/features.py

# Train both models
python src/train.py

# Run full evaluation suite (saves plots to data/)
python src/evaluate.py

# Build SHAP explainer + generate global importance + sample waterfall
python -m src.explain

# Load cached explainer in your own code
# from src.explain import load_explainer, explain_one, plot_waterfall

# Run the Streamlit app (available after Phase 5)
streamlit run src/app.py
```

## Interview Talking Points
1. **\"Why accuracy is the wrong metric here.\"** Defaults are ~22% of the dataset; a trivial \"always predict no-default\" classifier achieves 78% accuracy while catching zero actual defaults. PR-AUC and recall-at-threshold are the correct objectives for this imbalanced problem.
2. **\"Calibration vs. discrimination.\"** A model can rank borrowers correctly (high AUC) yet output systematically mis-scaled probabilities. Calibration matters because a lender uses the raw probability to *price risk*, not just to rank applicants.
3. **\"Explainability as a legal requirement.\"** Under fair-lending / adverse-action rules, a lender must disclose *why* a credit decision was made. SHAP produces a defensible per-applicant reason list — the difference between a research model and a deployable one.
4. **\"Feature engineering as domain knowledge.\"** The four engineered features (utilization, payment ratio, delinquency count, balance trend) mirror the actual factors FICO uses to compute credit scores. Encoding domain knowledge directly into features reduces what the model has to learn from data alone.
5. **\"Threshold is a business decision, not a statistical one.\"** The 0.5 default is arbitrary. By scanning the PR curve, you can explicitly choose the precision/recall tradeoff that matches the cost structure of the problem — a skill that distinguishes ML practitioners from ML researchers.
6. **\"Global vs. local explanations.\"** A global SHAP summary tells you which features matter across all borrowers — useful for model audits and regulatory review. A local (per-applicant) SHAP waterfall tells you why *this specific person* was scored as they were — required for adverse-action notices and individual fairness.

## Progress
| Phase | Status | Completed |
|-------|--------|-----------|
| 1 — Setup & Data Acquisition | ✅ complete | 2026-06-15 |
| 2 — Preprocessing & Feature Engineering | ✅ complete | 2026-06-16 |
| 3 — Modeling & Evaluation | ✅ complete | 2026-06-19 |
| 4 — Explainability | ✅ complete | 2026-06-21 |
| 5 — App & Polish | pending | — |
