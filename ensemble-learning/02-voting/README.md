# 02 — Voting Ensemble

Train several **different** models on the **same** data and let them vote (classification) or average (regression).
(CampusX *Ensemble Learning* videos 2–4.)

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Hard vs soft voting, weights, same algorithm with different hyperparameters, VotingRegressor, when voting does not help | **1. Read first** |
| [`01-voting-classifier.ipynb`](01-voting-classifier.ipynb) | Hard vs soft by hand, VotingClassifier on curved data, Iris (when voting loses), weights, 5 SVMs voting | **2** |
| [`02-voting-regressor.ipynb`](02-voting-regressor.ipynb) | Averaging regressors, VotingRegressor beats its members, weights, trees of different depths, a counter-example | **3** |

## How to read the notebooks
- ✍️ **Core code: learn this.** · 🎨 **Visualization only** (collapsed) · 🧠 **Quick check** at the end.

## Key results
| Experiment | Single model(s) | Voting |
|---|---|---|
| make_moons: LR / SVC / RF | 0.860 / 0.907 / 0.907 | hard 0.907, **soft 0.927** |
| Iris (2 overlapping species): LR / RF / KNN | **0.75** / 0.62 / 0.62 | hard 0.68, soft 0.65, weighted (3,1,1) 0.71 |
| 5 SVMs, polynomial degree 1–5 | best 0.894 | **0.926** |
| Regression (make_friedman1): LR / KNN / DT, R² | 0.709 / 0.654 / 0.617 | **0.760** |
| Trees of depth 3, 5, 7, 9 | best 0.621 | **0.675** |
| LR / DT / SVR | SVR **0.813** | 0.803 (slightly below the best) |

## Learning objectives
- Use `VotingClassifier` (hard / soft / weights) and `VotingRegressor`
- Explain why soft voting usually helps and when voting does not beat the best model

> Note: `SVC(probability=True)` is deprecated in scikit-learn 1.9 (still works); notebook 01 explains this.

➡️ Next: [`../03-bagging/`](../03-bagging/)
