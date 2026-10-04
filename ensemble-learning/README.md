# Ensemble Learning

> **Combine many models into one that predicts better than any of them alone.**
> Follows the CampusX *Ensemble Learning* playlist, extended with Random Forest, AdaBoost, Gradient Boosting,
> XGBoost and Stacking & Blending.

Prerequisite: **Decision Trees** ([`../day48-decision-tree/`](../day48-decision-tree/)): almost every ensemble here is built from trees.

## Roadmap

Work through the folders in order. In every folder: read `theory.md` first, then the notebooks in numbered order.

| # | Folder | Big idea | Notebooks |
|:-:|---|---|---|
| 01 | [`01-introduction/`](01-introduction/) | Wisdom of the crowd: why combining diverse models works (the 0.7 → 0.784 proof) | ensemble intro |
| 02 | [`02-voting/`](02-voting/) | **Different** algorithms, same data → vote / average (hard vs soft voting) | classifier · regressor |
| 03 | [`03-bagging/`](03-bagging/) | **Same** algorithm, different bootstrap samples → reduces **variance**; Bagging vs Boosting | classifier · regressor |
| 04 | [`04-random-forest/`](04-random-forest/) | Bagging of trees **+ random features at every split** | classification · regression · bagging vs RF · OOB · feature importance · tuning |
| 05 | [`05-adaboost/`](05-adaboost/) | **Boosting:** weak learners in sequence, each focusing on the previous mistakes → reduces **bias** | step by step · hyperparameters |
| 06 | [`06-gradient-boosting/`](06-gradient-boosting/) | Each new tree learns the **residuals** of the ensemble so far | step by step · in scikit-learn |
| 07 | [`07-xgboost/`](07-xgboost/) | Regularised, fast gradient boosting: the Kaggle favourite | breast cancer · churn project |
| 08 | [`08-stacking-and-blending/`](08-stacking-and-blending/) | A **meta-model** learns how to combine the base models | stacking · blending & stacking by hand |

## The family at a glance

| | Base models | How they are trained | How they are combined | Mainly reduces |
|---|---|---|---|---|
| **Voting** | different algorithms | independently, same data | majority vote / average | variance (errors cancel) |
| **Bagging / Random Forest** | same algorithm (deep trees) | independently, **in parallel**, on bootstrap samples | equal vote / average | **variance** |
| **Boosting** (AdaBoost, GB, XGBoost) | same weak algorithm (shallow trees) | **sequentially**, each fixing the previous errors | weighted vote / sum | **bias** |
| **Stacking / Blending** | different algorithms | independently | a trained **meta-model** | both, depending on the members |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end of each notebook: click to reveal the answers.

All notebooks are saved **with outputs**, use fixed random seeds, and report results honestly: ensembles usually win,
but several notebooks show cases where they do **not** beat the best single model, and explain why.

## Highlights

| Folder | Result |
|---|---|
| 01 | 3 independent 70% models → majority vote **78.4%**; 11 independent models → ≈ 92%, but ≈ 70% if they share their mistakes |
| 02 | 5 SVMs (degree 1–5) voting: **0.926** vs best single 0.894 · on Iris, LR alone (0.75) beats every voting ensemble |
| 03 | deep tree 0.927 → bagging 0.945 (tuned 0.953) · regression R² 0.62 → **0.85** · bagging an SVM does *not* help |
| 04 | single tree vs forest over 20 data splits: 0.862 vs **0.893** · OOB ≈ test accuracy · impurity importance can favour a pure-noise column |
| 05 | AdaBoost built by hand (3 stumps, α = 0.42 / 0.65 / 0.70) · 50 stumps: 0.81 test vs 0.60 for one stump |
| 06 | hand-made gradient boosting matches scikit-learn exactly · Friedman R²: tree 0.60 → RF 0.83 → GB **0.90–0.915** |
| 07 | breast cancer ≈ 97% · churn project with early stopping and a tuned threshold: catches 278 of 407 churners, ROC-AUC 0.868 |
| 08 | naive stacking: 100% train but worse test (leakage) · K-fold stacking fixes it · on small data stacking ≈ best single model |

## What changed from the original material
This folder replaces the old `day41-random-forest`, `day42-adaboost`, `day43-stacking-and-blending`,
`gradient-boosting/`, `day49-random-forest`, `day50-ensemble-learning`, the root `adaboost_demo.ipynb` (a duplicate)
and `Part 10 - Model Selection & Boosting/Section 49 - XGBoost`. Every old experiment was kept, explained and
fixed where needed, including:
- the AdaBoost demo's hard-coded errors / α values and the third stump trained on the wrong data;
- the gradient-boosting function starting from 0 instead of the mean and ignoring the learning rate in residuals;
- the XGBoost notebook using the patient ID as a feature and crashing on labels 2/4 with XGBoost 3.x;
- the random-forest-by-hand query being sent to trees trained on different columns, and the missing MNIST file (now `load_digits`);
- deprecated / removed APIs (`base_estimator`, `load_boston`, `algorithm=` in AdaBoost, `criterion='mse'`).
