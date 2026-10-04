# Random Forest (Classification & Regression)

**Many decision trees, voting together**: more accurate and far more stable than a single tree.
Notebooks 01–02 cover the basics; notebooks 03–06 go deeper (bagging vs random forest, OOB score, feature
importance, tuning).

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Bootstrap sampling, random features, why averaging works, OOB score, regression forests, hyperparameters, Tree vs Forest (§1–§9); **bagging vs RF, OOB details, how feature importance is calculated, tuning** (§10–§13); strengths, common mistakes, revision questions | **1. Read first** |
| [`01-random-forest-classification.ipynb`](01-random-forest-classification.ipynb) | Social Network Ads: train → evaluate → **individual trees vs the forest** → **stability over 20 splits** → number of trees & OOB → tuning → feature importance | **2** |
| [`02-random-forest-regression.ipynb`](02-random-forest-regression.ipynb) | Position Salaries: predict a salary → what individual trees say → tree vs forest curve → more trees → **Decision Tree vs Random Forest vs SVR** | **3** |
| [`03-bagging-vs-random-forest.ipynb`](03-bagging-vs-random-forest.ipynb) | Feature subset **per tree** (bagging) vs **per split** (random forest) with `plot_tree`; **a random forest built by hand** (row + column sampling, majority vote); strength vs diversity of the trees | **4** |
| [`04-oob-score.ipynb`](04-oob-score.ipynb) | Why ≈ 37% of rows are out-of-bag; `oob_score_` vs test accuracy on heart data; **OOB computed by hand**; OOB vs number of trees; OOB vs CV vs test over 10 seeds | **5** |
| [`05-feature-importance.ipynb`](05-feature-importance.ipynb) | **Importance by hand** (impurity decrease, worked example = sklearn); forest = average of trees; **digit-pixel importance heatmap**; bias of impurity importance vs `permutation_importance` | **6** |
| [`06-hyperparameter-tuning.ipynb`](06-hyperparameter-tuning.ipynb) | Heart data: RF vs gradient boosting vs SVC vs logistic regression; the main hyperparameters; validation curves; **GridSearchCV vs RandomizedSearchCV** | **7** |
| `Social_Network_Ads.csv` | 400 users: `Age`, `EstimatedSalary`, `Purchased` (0/1) | data |
| `Position_Salaries.csv` | 10 job levels and salaries | data |
| `heart.csv` | 303 patients (UCI heart disease): 13 medical features, `target` (1 = heart disease) | data |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

**Basics (notebooks 01–02, Social Network Ads)**

| Model | Test accuracy |
|---|:-:|
| Random forest, 100 trees (default) | 0.92 |
| Random forest, tuned (`max_depth=3`, `min_samples_leaf=5`) | **0.94** |
| Average over 20 random splits: single tree vs forest | 0.862 vs **0.893** |

Regression, level 6.5: decision tree 150,000 · random forest **158,300** · SVR ≈ 170,000.

**Bagging vs random forest (notebook 03, synthetic data, 5 features, `max_features=2`, 5-fold CV)**

| Model | Avg single tree | Trees disagree | Ensemble CV accuracy |
|---|:-:|:-:|:-:|
| Single tree | – | – | 0.960 |
| Bagging, all columns | 0.932 | 0.070 | 0.974 |
| Bagging, 2 columns per tree | 0.763 | 0.331 | 0.970 |
| Random forest, 2 per split | 0.918 | 0.113 | **0.980** |

Hand-made forest of 100 trees: 0.967 test accuracy (single tree 0.907, sklearn forest 0.980).

**OOB score (notebook 04, heart data)**: `oob_score_` **0.835** vs test accuracy **0.836**; our by-hand OOB
computation matches sklearn exactly. Over 10 seeds: OOB 0.817 ± 0.022, 5-fold CV 0.816 ± 0.025, test 0.833 ± 0.056.

**Feature importance (notebook 05)**: hand-computed importances match `feature_importances_` (0.625 / 0.375 in the
5-row example); a random-noise column got impurity importance 0.067 (rank 8 of 15) but permutation importance ≈ 0.
Digits: 0.967 test accuracy, the centre pixels matter most.

**Tuning (notebook 06, heart data)**

| Model | 5-fold CV accuracy | Test accuracy |
|---|:-:|:-:|
| Random forest (default) | 0.827 | 0.836 |
| Gradient boosting / SVC / Logistic regression (default) | 0.823 / 0.814 / 0.806 | 0.770 / 0.869 / 0.852 |
| GridSearchCV best (81 combinations) | 0.839 | 0.902 |
| RandomizedSearchCV best (30 combinations, ~4–5× faster) | 0.839 | 0.902 |

On only 61 test patients, differences of a few points are partly noise (see the notebooks' honest notes).

## Learning objectives
- Explain bagging (bootstrap samples) and random feature selection
- Explain why voting / averaging many trees reduces overfitting and instability
- Explain the difference between bagging with `max_features` (per tree) and a random forest (per split), and build a simple forest by hand
- Explain why ≈ 37% of rows are out-of-bag and how `oob_score_` is computed
- Calculate a tree's feature importance by hand, explain the forest's average, and know when to use permutation importance instead
- Use `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`, `max_samples` and tune them with GridSearchCV / RandomizedSearchCV
- Apply random forests to both classification and regression

## Related
- **Decision trees:** top-level folder `day48-decision-tree` (the building block of a forest).
- **Bagging:** [`../03-bagging/`](../03-bagging/).
- **Boosting:** [`../05-adaboost/`](../05-adaboost/) and [`../06-gradient-boosting/`](../06-gradient-boosting/).
- **Stacking & blending:** [`../08-stacking-and-blending/`](../08-stacking-and-blending/).

Sources: adapted from `Part 3 - Classification/Section 20`, `Part 2 - Regression/Section 9` and the former
random-forest deep-dive notebooks (bagging vs RF, RF learning tool, OOB demo, feature importance, code example).
