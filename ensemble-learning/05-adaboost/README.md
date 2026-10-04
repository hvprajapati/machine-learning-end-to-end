# 05 · AdaBoost

AdaBoost (*Adaptive Boosting*) trains weak models **one after another**. After every model, the rows it got wrong
become heavier, so the next model focuses on them. At the end, all models vote, and the better ones get a louder vote.

## Files

| Order | File | What it covers |
|---|---|---|
| 1 | [`theory.md`](theory.md) | Boosting idea, the AdaBoost algorithm with formulas, a fully worked numeric example, prediction, hyperparameters, AdaBoostRegressor, bagging vs boosting, sklearn cheat sheet, common mistakes, revision questions |
| 2 | [`01-adaboost-step-by-step.ipynb`](01-adaboost-step-by-step.ipynb) | AdaBoost **by hand** on a 10-row toy dataset: weights, stumps, error, alpha, weight updates, weighted resampling, 3 rounds, weighted-vote prediction; then compared with sklearn's `AdaBoostClassifier` |
| 3 | [`02-adaboost-hyperparameters.ipynb`](02-adaboost-hyperparameters.ipynb) | Noisy circles data: stump vs deep tree vs AdaBoost, effect of `n_estimators`, `learning_rate` and base-tree depth (decision boundaries + accuracy curves), `GridSearchCV`, bonus `AdaBoostRegressor` |

## How to read the notebooks

- Read `theory.md` §1–§5 first (or alongside notebook 01), then §6–§11 with notebook 02.
- Cells marked **✍️ Core code: learn this** are the code you should understand and be able to write.
- Cells marked **🎨 Visualization only** are collapsed plotting code: run them and look at the picture.
- Every notebook ends with **Key takeaways** and a **🧠 Quick check** (click a question to see the answer).
- All results are reproducible (fixed seeds / `random_state`). Notebooks are saved with their outputs.

## Key results

**Notebook 01: 3 rounds by hand (toy data, 10 rows)**

| Stump | Rule | Wrong rows | Weighted error | α = ½ ln((1−ε)/ε) |
|---|---|---|---|---|
| 1 | X2 ≤ 2.5 | 2, 6, 8 | 0.300 | 0.4236 |
| 2 | X1 ≤ 2.5 | 3, 5, 7 | 0.2143 | 0.6496 |
| 3 | X2 ≤ 7.0 | 0, 1, 8 | 0.1970 | 0.7027 |

- Hand-made 3-stump model: **9/10** training rows correct (row 8 = (9, 9) still wrong).
- sklearn `AdaBoostClassifier(n_estimators=3)`: **10/10**. Its stumps differ because of a tie in round 1 and because
  sklearn reweights (`sample_weight`) instead of resampling. Redoing the rounds by hand **with reweighting** reproduces
  sklearn's errors exactly, and sklearn's `estimator_weights_` are exactly **2 × our α** (same predictions).

**Notebook 02: noisy circles (500 points, 5-fold CV on 400 training points, 100 test points)**

| Model | CV acc | Train acc | Test acc |
|---|---|---|---|
| Single stump | 0.580 | 0.610 | 0.60 |
| Single deep tree | 0.740 | 1.000 | 0.72 |
| AdaBoost, 50 stumps, lr = 1.0 (default) | **0.838** | 0.865 | **0.81** |
| AdaBoost, 1500 stumps, lr = 0.1 | 0.830 | 0.855 | 0.79 |
| AdaBoost, 50 trees of depth 5 | 0.790 | 1.000 | 0.78 |
| AdaBoost, `GridSearchCV` best (= the defaults) | 0.838 | 0.865 | 0.81 |

- CV accuracy peaks at 20–100 stumps, then slowly drops (0.820 at 1000): mild overfitting on noisy data.
- Small learning rates need many more stumps (lr 0.1: test 0.65 after 50 stumps, 0.79 after 200);
  lr = 2.0 over-corrects every round and ends at test 0.50.
- Grid search (48 combinations) picked `max_depth=1, learning_rate=1.0, n_estimators=50`: tuning did not beat the defaults here.
- Bonus: `AdaBoostRegressor` (300 depth-4 trees) test R² **0.975** vs a single depth-4 tree **0.896**.

## Learning objectives

After this folder you should be able to:
1. Explain boosting and why it reduces **bias** (weak learners: high bias, low variance).
2. Run AdaBoost by hand: weighted error, α, weight update, normalisation, and the final `sign(Σ α·h)` vote with ±1 labels.
3. Explain resampling vs reweighting, and why sklearn's model weights are 2× the lecture formula.
4. Use `AdaBoostClassifier` / `AdaBoostRegressor` with the modern API (`estimator=`, no `algorithm=`).
5. Tune `n_estimators`, `learning_rate` and `estimator__max_depth` together, and read the trade-offs.
6. Know AdaBoost's limits: sensitive to noise/outliers, sequential (no parallel training).

## Related folders

- [`../03-bagging/`](../03-bagging/): bagging builds models in parallel and reduces variance; compare with boosting (table in [`theory.md`](theory.md) §10).
- [`../06-gradient-boosting/`](../06-gradient-boosting/): boosting generalised: each new model fits the remaining errors (gradients of a loss).
- [`../07-xgboost/`](../07-xgboost/): a fast, regularised gradient boosting library.
