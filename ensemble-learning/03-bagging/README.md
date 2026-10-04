# 03 — Bagging (Bootstrap Aggregation)

Train the **same** algorithm on many **bootstrap samples** of the data and combine by vote / average.
(CampusX *Ensemble Learning* videos 5–8, including **Bagging vs Boosting**.)

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Bootstrapping, why bagging reduces variance, the four flavours, OOB, scikit-learn parameters, **Bagging vs Boosting** | **1. Read first** |
| [`01-bagging-classifier.ipynb`](01-bagging-classifier.ipynb) | Bagging by hand, variance reduction picture, bagging / pasting / random subspaces / random patches, OOB, SVM vs trees, GridSearchCV | **2** |
| [`02-bagging-regressor.ipynb`](02-bagging-regressor.ipynb) | One tree vs bagging, bagging vs other regressors, number of trees, OOB, GridSearchCV | **3** |

## How to read the notebooks
- ✍️ **Core code: learn this.** · 🎨 **Visualization only** (collapsed) · 🧠 **Quick check** at the end.

## Key results
| Experiment | Single model | Bagging |
|---|---|---|
| 10,000 rows, 10 features: deep tree | 0.927 | bagging 0.945 · pasting 0.946 · subspaces 0.942 · patches 0.938 · tuned **0.953** |
| OOB estimate vs test accuracy | – | 0.943 vs 0.945 |
| SVC (a stable model) | **0.926** | 0.915 (bagging does not help) |
| Regression (make_friedman1), test R² | deep tree 0.624 | **0.849** (100 trees) |

## Learning objectives
- Explain bootstrapping and why bagging reduces variance
- Use the four bagging flavours and the OOB score with `BaggingClassifier` / `BaggingRegressor`
- Explain the differences between bagging and boosting

> Library changes since the videos: `base_estimator=` → `estimator=`; the Boston dataset was removed (regression demos use `make_friedman1`).

➡️ Next: [`../04-random-forest/`](../04-random-forest/)
