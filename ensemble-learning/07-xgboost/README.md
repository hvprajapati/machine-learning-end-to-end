# 07 — XGBoost (eXtreme Gradient Boosting)

**Gradient boosting, made regularised, fast and practical**: the go-to model for tabular data.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | What XGBoost adds to gradient boosting (regularised objective, 2nd-order gradients, worked leaf-value / split-gain example), subsampling, missing values, early stopping, hyperparameter table, tuning order, imbalance, XGBoost vs GB vs RF, LightGBM/CatBoost, sklearn cheat sheet, common mistakes | **1. Read first** |
| [`01-xgboost-breast-cancer.ipynb`](01-xgboost-breast-cancer.ipynb) | The corrected course notebook: drop the ID column, encode labels 2/4 → 0/1, `XGBClassifier`, confusion matrix, 10-fold CV, feature importance, **XGBoost vs Random Forest vs Gradient Boosting** | **2** |
| [`02-xgboost-churn.ipynb`](02-xgboost-churn.ipynb) | Realistic project on bank churn: preprocessing, train/validation/test, **early stopping + learning curves**, `scale_pos_weight` and **threshold tuning**, `RandomizedSearchCV`, feature importance, final test | **3** |
| `breast_cancer.csv` | 683 tumour samples: ID, 9 cell features (1–10), `Class` (2 = benign, 4 = malignant) | data |
| `Churn_Modelling.csv` | 10,000 bank customers, target `Exited` (20.4% churn) | data |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

**Breast cancer** (test set = 137 samples, default settings):

| Model | Test accuracy | 10-fold CV accuracy |
|---|:-:|:-:|
| XGBoost | 0.9708 | 0.965 ± 0.019 |
| Random Forest | 0.9708 | 0.969 ± 0.022 |
| Gradient Boosting | 0.9562 | 0.960 ± 0.023 |

→ On this small, easy dataset XGBoost is **not** clearly better than Random Forest.

**Bank churn** (validation set, 2,000 customers):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|:-:|:-:|:-:|:-:|
| Baseline (default) | 0.704 | 0.504 | 0.587 | 0.843 |
| Early stopping (144 trees) | 0.779 | 0.450 | 0.570 | 0.862 |
| + `scale_pos_weight = 3.91` | 0.564 | 0.686 | 0.619 | 0.850 |
| Early stopping + threshold 0.28 | 0.607 | 0.681 | **0.642** | **0.862** |
| `RandomizedSearchCV` (25 trials) + threshold 0.33 | 0.675 | 0.602 | 0.636 | 0.860 |

Final model (early stopping + threshold 0.28) on the **test set**: ROC-AUC **0.868**, recall **0.683**,
precision 0.599, F1 0.638. It catches 278 of the 407 customers who left.

## Learning objectives
- Explain what XGBoost adds to gradient boosting (regularisation γ/λ/α, second-order gradients, subsampling, missing values, speed)
- Prepare data correctly: labels 0 … n−1, no ID columns, encoded categoricals
- Use early stopping with a **validation** set and read learning curves from `evals_result()`
- Handle class imbalance with `scale_pos_weight` and threshold tuning, judged by precision / recall / F1 / ROC-AUC
- Tune XGBoost in a sensible order and compare it honestly with Random Forest and Gradient Boosting

## Related
- [`../06-gradient-boosting/`](../06-gradient-boosting/): the algorithm XGBoost builds on (read first).
- [`../05-adaboost/`](../05-adaboost/): the first boosting method (reweighting samples).
- [`../04-random-forest/`](../04-random-forest/): the bagging-based alternative used in the comparison.

Sources: adapted from the course notebook `xg_boost.ipynb` (data `Data.csv`, renamed `breast_cancer.csv`). `Churn_Modelling.csv` was added for the realistic project.
