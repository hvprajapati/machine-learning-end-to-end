# Random Forest: Theory (Classification & Regression)

> Builds on decision trees (top-level folder `day48-decision-tree`) and bagging ([`../03-bagging/`](../03-bagging/)).
> §1–§9 go with notebooks 01–02; the deeper topics (bagging vs random forest, OOB score, feature importance,
> tuning) are in §10–§13 and notebooks 03–06.

---

## 1. The idea: the wisdom of the crowd

A single decision tree has two problems (see `day48-decision-tree`):
- it **overfits** easily;
- it is **unstable**: a small change in the data can produce a very different tree.

**Random Forest** trains **many different trees** and combines them:
- **Classification:** each tree votes; the **majority** wins.
- **Regression:** each tree predicts a number; the forest returns the **average**.

Each tree makes mistakes, but **different trees make different mistakes**, so when they are combined the mistakes
largely cancel out and the correct signal remains.

```
           Training data
       ┌─────────┼─────────┐
   sample 1   sample 2   sample 3  ...   (random rows, drawn with replacement)
       │          │          │
    Tree 1     Tree 2     Tree 3   ...   (random features at every split)
       │          │          │
    "buys"   "doesn't"    "buys"
       └──────────┼──────────┘
          majority vote → "buys"
```

---

## 2. Where the randomness comes from

The trees must be **different**, otherwise they would all make the same mistakes. Random Forest uses two tricks.

### a) Bootstrap sampling (rows)
Each tree is trained on a **bootstrap sample**: $n$ rows drawn **at random with replacement** from the $n$ training rows.
- Some rows appear several times, others not at all.
- On average each tree sees about **63%** of the distinct rows; the other ~37% are its **out-of-bag (OOB)** rows.

Training many models on bootstrap samples and combining them is called **bagging** (*bootstrap aggregating*).

### b) Random feature selection (columns)
At **every split**, a tree may only consider a **random subset of the features** (`max_features`):

| Task | scikit-learn default `max_features` |
|---|---|
| Classification | $\sqrt{\text{number of features}}$ |
| Regression | all features (1.0) |

This stops every tree from always choosing the same strongest feature first, making the trees **less similar**.

---

## 3. Why averaging helps

| | Single deep tree | Random forest |
|---|---|---|
| Bias (too simple?) | low | low (same deep trees) |
| Variance (too sensitive to data?) | **high** | **much lower** (averaging) |

Averaging many low-bias, high-variance trees keeps the low bias and reduces the variance.

From the notebook, over **20 different train/test splits** of the Social Network Ads data:

| Model (no tuning) | Mean test accuracy |
|---|:-:|
| Single decision tree | 0.862 |
| Random forest (100 trees) | **0.893** |

The forest was better or equal on **all 20** splits.

---

## 4. The out-of-bag (OOB) score

Each row is out-of-bag for roughly a third of the trees. Predicting each row using **only the trees that did not see
it** gives a validation score **without a separate validation set**:

```python
RandomForestClassifier(n_estimators=200, oob_score=True).fit(X_train, y_train).oob_score_
```

Why a third (≈ 37%), how the score is computed and how reliable it is: see §11.

---

## 5. Random forest regression

Identical procedure; each tree is a regression tree, and the forest **averages** their predictions.

From the notebook (Position Salaries, level 6.5):

| Model | Prediction |
|---|---|
| Single tree | 150,000 (exactly level 6) |
| Random forest, 100 trees | **158,300** (an average of trees predicting 110,000, 150,000, 200,000, 300,000, …) |

The forest's curve has **many more, smaller steps** than a single tree's staircase. Like single trees, a forest
**cannot extrapolate** beyond the range of the training targets.

---

## 6. Important hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_estimators` | 100 | number of trees. More = more stable, slower. **More trees do not cause overfitting.** |
| `max_features` | `'sqrt'` (clf) / 1.0 (reg) | features per split; lower = more diverse trees |
| `max_depth` | None | depth of each tree; limit it to smooth the model |
| `min_samples_leaf` | 1 | minimum points per leaf; higher = smoother |
| `bootstrap` | True | use bootstrap samples |
| `oob_score` | False | compute the OOB score |
| `n_jobs` | None | `-1` trains trees in parallel on all CPU cores |
| `random_state` | None | fix for reproducible results |

Tuned in the notebook: `n_estimators=100, max_depth=3, min_samples_leaf=5` → test accuracy **0.94**
(default forest: 0.92).

---

## 7. Feature scaling is NOT needed

A forest is made of trees, and trees compare one feature with a threshold. Scaling makes no difference.

---

## 8. Using random forests in scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)        # average of the trees' class probabilities
clf.feature_importances_         # importance of each feature
clf.estimators_                  # the individual trees

reg = RandomForestRegressor(n_estimators=100, random_state=0)
reg.fit(X, y)

# deeper topics (notebooks 03-06)
RandomForestClassifier(oob_score=True).fit(X_train, y_train).oob_score_   # free validation score (§11)
clf.estimators_samples_          # rows used by each tree (to find its OOB rows)

from sklearn.inspection import permutation_importance                   # less biased importance (§12)
permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0).importances_mean

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV    # tuning (§13)
RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions, n_iter=30, cv=5, n_jobs=-1)
```

---

## 9. Decision Tree vs Random Forest

| | Decision Tree | Random Forest |
|---|---|---|
| Model | one tree | many trees, voting / averaging |
| Accuracy | good | usually **better** |
| Overfitting | high, unless limited | much lower |
| Stability | unstable | **stable** |
| Interpretability | **high**: rules can be printed | low: hundreds of trees |
| Speed | very fast | slower (but parallelisable) |
| Scaling needed | no | no |

---

## 10. Bagging vs Random Forest: per tree or per split?

Both build many trees on **bootstrap samples** and vote. The difference is **where the random features are drawn**.

| | Bagging (`BaggingClassifier(max_features=k)`) | Random forest (`RandomForestClassifier(max_features=k)`) |
|---|---|---|
| Random features drawn | **once per tree** | **again at every split (node)** |
| Features one tree can use | at most $k$ | all of them (a different $k$ at each node) |
| Effect | trees can be weak if they miss the good features | trees stay strong **and** become different |
| Base model | any model | decision trees only |

Plain `BaggingClassifier` (default) uses **all** features for every tree; a random forest = bagging of trees
**plus** per-split feature sampling.

From notebook 03 (synthetic data, 5 features, `max_features=2`, 100 trees):

| | Avg single-tree accuracy (strength) | Trees disagree (diversity) | Ensemble, 5-fold CV |
|---|:-:|:-:|:-:|
| Single tree | – | – | 0.960 |
| Bagging, all 5 columns | **0.932** | 0.070 | 0.974 |
| Bagging, 2 columns per tree | 0.763 | **0.331** | 0.970 |
| Random forest, 2 per split | 0.918 | 0.113 | **0.980** |

An ensemble needs trees that are **strong and different**. Too little randomness → similar trees (little to
average out); too much → weak trees. Per-split sampling is a good balance. (On average a random-forest tree used
4.86 of the 5 columns; a bagging tree exactly 2.) The differences are small (about 1 point) on this easy dataset.

**Random forest by hand** (notebook 03): (1) bootstrap the rows, (2) pick random columns, (3) fit a tree on each
sample, (4) majority vote; each tree must receive *its own* columns of a new row. Our simplified version (columns
chosen per tree) reached 0.967 test accuracy vs 0.907 for one tree and 0.980 for scikit-learn's forest.

---

## 11. The OOB score in detail

**Why 37%?** A bootstrap sample makes $n$ draws with replacement. A given row is missed by one draw with probability
$1 - \frac1n$, so it is missed by all $n$ draws with probability

$$\left(1-\frac1n\right)^n \longrightarrow e^{-1} \approx 0.368 .$$

So each tree sees ≈ **63.2%** of the distinct rows; the other ≈ **36.8%** are its out-of-bag rows. (Example with
$n = 242$: $(1-1/242)^{242} = 0.367$.)

**How `oob_score_` is computed:**
1. For each training row, take only the trees for which the row was out-of-bag (≈ 37% of the trees).
2. Average their predicted class probabilities → the OOB prediction (`oob_decision_function_`).
3. OOB score = accuracy (classifier) or $R^2$ (regressor) of these predictions over all training rows.

**Results on the heart data** (notebook 04, 242 training / 61 test patients):

| | Value |
|---|:-:|
| `oob_score_` (100 trees) | 0.835 (our by-hand computation gives exactly the same) |
| Test accuracy | 0.836 |
| Over 10 random seeds: OOB / 5-fold CV / test (mean ± std) | 0.817 ± 0.022 / 0.816 ± 0.025 / 0.833 ± 0.056 |

The OOB score behaves like cross-validation **without the extra fits**, and was more stable than a single small
test split.

**Rules:** needs `bootstrap=True`; needs enough trees (with very few trees some rows are never out-of-bag and
scikit-learn warns); each row is judged by only ≈ 37% of the trees, so with few trees OOB is a bit pessimistic.

---

## 12. Feature importance: how it is calculated

**Impurity importance (MDI, `feature_importances_`).** For every split node $t$ in a tree:

$$\Delta(t) = \frac{N_t}{N}G_t - \frac{N_L}{N}G_L - \frac{N_R}{N}G_R$$

($N$ = rows in the tree, $N_t$ = rows at the node, $G$ = Gini impurity, $L/R$ = children). A feature's importance =
the sum of $\Delta(t)$ over the nodes that split on it, **normalised** so that all features sum to 1.

**Worked example** (5 rows, notebook 05):

```
root: f1 <= -0.894   N=5, Gini 0.48
 ├── leaf            N=1, Gini 0
 └── f0 <= 1.01      N=4, Gini 0.375
      ├── leaf       N=3, Gini 0
      └── leaf       N=1, Gini 0
```

| Node | Feature | $\Delta(t)$ |
|---|---|---|
| root | f1 | $\frac55 \cdot 0.48 - \frac15 \cdot 0 - \frac45 \cdot 0.375 = 0.48 - 0.30 = 0.18$ |
| right child | f0 | $\frac45 \cdot 0.375 - 0 - 0 = 0.30$ |

Total 0.48 → f0 = 0.30/0.48 = **0.625**, f1 = 0.18/0.48 = **0.375**, exactly `clf.feature_importances_`.
Note that f1 is at the root but f0 is more important: importance measures **how much impurity a split removes**,
not where it sits.

**Random forest:** compute the normalised importances of **each tree**, then **average** them over the trees.
(Two trees with importances [1, 0] and [0, 1] → forest [0.5, 0.5].)

**On images** (8×8 `load_digits`, 64 pixel features) the importances can be drawn as an 8×8 heatmap: the centre
pixels matter most; border pixels that are blank in nearly every image get (almost) zero.

**⚠️ Bias of impurity importance.** It is measured on the training data and **favours features with many unique
values**. In notebook 05 a column of pure random numbers got importance 0.067 (rank 8 of 15, above six real
features), while a 3-valued random column got 0.024.

**Permutation importance** (`sklearn.inspection.permutation_importance`): on held-out data, shuffle one column and
measure the drop in score. It gave both noise columns ≈ 0. It is slower and noisy on small test sets, and correlated
features can share or hide each other's importance, but it is the more trustworthy check.

---

## 13. Hyperparameters and tuning

The knobs (see §6) control the trees' **strength** (depth, leaf size, features per split) and **diversity**
(`max_features`, `max_samples`, `bootstrap`). `n_estimators` only needs to be large enough.

More parameters:

| Parameter | Default | Effect |
|---|---|---|
| `max_samples` | None (= n rows) | size of each bootstrap sample (fraction or count); smaller = more diverse, faster trees. **Only with `bootstrap=True`.** |
| `min_samples_split` | 2 | minimum rows needed to split a node |
| `criterion` | `'gini'` | split quality (`'entropy'`, `'log_loss'` also possible) |
| `class_weight` | None | `'balanced'` for imbalanced classes |

**Two search strategies** (both with cross-validation):

| | `GridSearchCV` | `RandomizedSearchCV` |
|---|---|---|
| Tries | **every** combination | `n_iter` random combinations |
| Cost | multiplies with every new parameter | fixed by `n_iter` |
| Use when | small grid, final fine-tuning | big search space, first exploration |

From notebook 06 (heart data, 5-fold CV on 242 training rows):

| Model | CV accuracy | Test accuracy (61 rows) |
|---|:-:|:-:|
| Default random forest | 0.827 | 0.836 |
| GridSearchCV best (81 combinations) | 0.839 | 0.902 |
| RandomizedSearchCV best (30 combinations, about 4–5× faster) | 0.839 | 0.902 |

Both best settings used `max_features=0.2` (2–3 features per split). Tuning gained about 1 point in CV; the larger
test gain (4 patients) is partly luck on a tiny test set. The default forest (0.827 CV) was already on par with
gradient boosting, SVC and logistic regression (0.806–0.823 CV) on this data.

**Tip:** `max_samples` with `bootstrap=False` is an error. When searching over `bootstrap`, pass a **list of two
grids** (one with bootstrap + `max_samples`, one without).

---

## 14. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Strong accuracy **with little tuning** | Hard to interpret ("black box" compared with one tree) |
| Robust to overfitting and noise | Slower and larger in memory than a single tree |
| Handles non-linear patterns and interactions | Cannot extrapolate (regression) |
| No scaling; gives feature importance and OOB score | Can be beaten by gradient boosting on tabular data with careful tuning |
| Works for classification and regression | Feature importance can be biased toward features with many unique values |

---

## 15. Common mistakes

| Mistake | Better |
|---|---|
| Thinking `BaggingClassifier(max_features=k)` is a random forest | Bagging draws features once **per tree**; a random forest draws them **per split** (§10) |
| Adding more trees to "fix overfitting" | More trees only stabilise; limit `max_depth`, raise `min_samples_leaf`, lower `max_features` |
| Scaling features before a forest | Not needed (trees compare a feature with a threshold) |
| Comparing models on one tiny test split | Use cross-validation or the OOB score |
| `oob_score=True` or `max_samples` with `bootstrap=False` | Both need bootstrap sampling |
| Trusting `feature_importances_` blindly | Impurity importance favours many-valued features; check with `permutation_importance` |
| Feeding a hand-made tree the wrong columns | Each tree must get exactly the columns it was trained on |

---

## 16. Quick revision questions

1. *What is a random forest?* An ensemble of decision trees trained on bootstrap samples with random feature subsets, combined by voting (classification) or averaging (regression).
2. *What two kinds of randomness are used?* Bootstrap samples of rows, and random feature subsets at each split.
3. *Why does a forest overfit less than one tree?* Averaging many different trees cancels their individual errors (reduces variance).
4. *Does adding more trees cause overfitting?* No; it only makes predictions more stable (and slower).
5. *What is the OOB score?* Accuracy on each row using only the trees that did not see that row during training.
6. *Does a random forest need feature scaling?* No.
7. *Random forest regression prediction?* The average of the trees' predictions.
8. *Bagging vs random forest?* Bagging picks random features once per tree (or uses all); a random forest picks a new random subset at every split.
9. *Why are about 37% of rows out-of-bag?* $(1-1/n)^n \to e^{-1} \approx 0.368$.
10. *How is a tree's feature importance computed?* The sum of weighted impurity decreases of the splits on that feature, normalised; a forest averages over its trees.
11. *What is the alternative to impurity importance?* Permutation importance: shuffle a column on held-out data and measure the score drop.
12. *GridSearchCV vs RandomizedSearchCV?* Every combination vs `n_iter` random combinations; random search is much cheaper on big spaces.

---

## Summary

- **Random forest = bagging + random features + voting/averaging** over many decision trees.
- It keeps the trees' flexibility but removes much of their **variance**: more accurate and more stable.
- Good results with little tuning; no scaling needed; `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` are the main knobs.
- Random forest ≠ bagging with `max_features`: features are re-drawn at **every split**, giving trees that are strong *and* diverse.
- Each tree leaves out ≈ 37% of the rows → a free **OOB score** that behaves like cross-validation.
- `feature_importances_` = averaged impurity decrease; handy but biased, so confirm with **permutation importance**.
- Tune with **RandomizedSearchCV** first; on small data expect small gains.
