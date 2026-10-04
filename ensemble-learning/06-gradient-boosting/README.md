# Gradient Boosting

**Start with a simple guess (the mean), then add small trees one at a time, each one learning the mistakes that are
still left.** Add them up with a learning rate and you get one of the most accurate methods for tabular data.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Intuition, why "gradient", the algorithm with formulas, a 5-row worked example, learning rate & number of trees, tree size, subsample, early stopping, classification (log-odds), AdaBoost vs Gradient Boosting, HistGradientBoosting, cheat sheet | **1. Read first** |
| [`01-gradient-boosting-step-by-step.ipynb`](01-gradient-boosting-step-by-step.ipynb) | Gradient boosting **by hand** on y = 3x² + noise: mean → residuals → tree 1 → tree 2 → a `gradient_boost` function → plots after 1, 2, 3, 5, 10, 50 trees → effect of the learning rate → train vs test error → **exact match with `GradientBoostingRegressor`** | **2** |
| [`02-gradient-boosting-in-sklearn.ipynb`](02-gradient-boosting-in-sklearn.ipynb) | `GradientBoostingRegressor` on `make_friedman1` and `GradientBoostingClassifier` on breast cancer: `staged_predict`, learning rate × trees, `max_depth`, `subsample`, early stopping, `GridSearchCV`, feature importance, log-odds check, comparison with Random Forest and HistGradientBoosting | **3** |

All data is generated in code or loaded from `sklearn.datasets`: no CSV files needed.

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

**Notebook 1** (100 points, y = 3x² + noise, trees with 8 leaves)

| Model | Train MSE |
|---|:-:|
| Mean only (0.265458) | 0.0557 |
| Mean + tree 1 | 0.0033 |
| Mean + tree 1 + tree 2 | 0.0021 |

| Learning rate | Best test MSE | after | Test MSE after 200 trees |
|:-:|:-:|:-:|:-:|
| 1.0 | 0.0037 | 4 trees | 0.0041 |
| 0.3 | 0.0029 | 11 trees | 0.0041 |
| 0.1 | 0.0029 | 33 trees | 0.0038 |

Hand-made loop vs `GradientBoostingRegressor(max_leaf_nodes=8, max_depth=None)`: identical (max difference 2.2 × 10⁻¹⁶).
With the default `max_depth=3` left in place the predictions differ (max 0.015), because that limit still applies.

**Notebook 2**

| Model | Friedman test R² | Breast cancer test accuracy |
|---|:-:|:-:|
| Single decision tree | 0.599 | – |
| Random Forest (300 trees) | 0.830 | 0.958 |
| Gradient Boosting (defaults) | 0.902 | 0.958 |
| Gradient Boosting (GridSearchCV: lr 0.2, depth 2, 300 trees) | **0.911** | – |
| Gradient Boosting + early stopping | 0.915 (178 trees) | 0.937 (51 trees) |
| HistGradientBoosting (defaults) | 0.905 | **0.972** |

Feature importance found the 5 real Friedman features (x0–x4); the 5 noise features got about 0.005 each.

## Learning objectives
- Explain gradient boosting as "start from the mean, then fit each new tree to the residuals"
- Explain why the residual is the negative gradient of squared loss
- Compute one boosting step by hand (mean, residuals, stump, update with a learning rate)
- Tune `n_estimators` together with `learning_rate`, keep trees shallow, use `subsample` and early stopping
- Use `staged_predict` to see error vs number of trees
- Explain how classification works (adding log-odds, then sigmoid)
- Compare gradient boosting with AdaBoost and Random Forest

## Related
- [`../04-random-forest/`](../04-random-forest/): the other big tree ensemble (bagging: independent trees, averaged).
- [`../05-adaboost/`](../05-adaboost/): the first boosting algorithm (re-weights the rows instead of fitting residuals).
- [`../07-xgboost/`](../07-xgboost/): a faster, regularised implementation of gradient boosting.
