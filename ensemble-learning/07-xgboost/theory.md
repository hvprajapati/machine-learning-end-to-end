# XGBoost: Theory

> Builds on **[`../06-gradient-boosting/`](../06-gradient-boosting/)**. XGBoost *is* gradient boosting, with a better
> objective and a lot of engineering. If "fit the next tree on the residuals" is not clear yet, read that folder first.

---

## 1. What is XGBoost?

**XGBoost = eXtreme Gradient Boosting.** It is an open-source library (Tianqi Chen, 2014–2016) that implements
gradient-boosted decision trees in a way that is:

- **regularised**: it penalises complex trees, so it overfits less;
- **fast**: written in C++, parallel, cache-aware, optional GPU;
- **practical**: handles missing values, has early stopping, works with the scikit-learn API.

For years it won most Kaggle competitions on **tabular data** (rows and columns, like spreadsheets), and it is still one
of the first models to try for such data.

**Analogy:** gradient boosting is a student who corrects their mistakes one exam at a time. XGBoost is the same student
with a strict teacher (regularisation), a faster way of studying (engineering) and a rule "stop when the mock exams stop
improving" (early stopping).

---

## 2. Quick recap: gradient boosting

From `06-gradient-boosting`:

1. Start with a simple prediction $F_0$ (e.g. the mean, or the log-odds for classification).
2. Repeat for $m = 1 \dots M$:
   - compute how wrong we are for each row (the **residuals** / negative gradients);
   - fit a small **tree** $h_m$ to them;
   - update: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$, where $\eta$ is the **learning rate**.
3. Final prediction = sum of all the trees' (shrunken) outputs.

```
prediction = base value + η·tree1 + η·tree2 + η·tree3 + ...
                           (each tree fixes what the previous ones got wrong)
```

XGBoost keeps exactly this loop. What changes is **how each tree is built** and **how fast**.

---

## 3. What makes XGBoost better

### 3.1 A regularised objective

Plain gradient boosting only tries to reduce the loss. XGBoost minimises **loss + a penalty for complex trees**:

$$
\text{Obj} = \sum_i \text{loss}(y_i, \hat y_i) \;+\; \sum_{\text{trees}} \Big( \gamma\, T \;+\; \tfrac{1}{2}\lambda \sum_j w_j^2 \;+\; \alpha \sum_j |w_j| \Big)
$$

| Symbol | Name in code | Penalises | Effect |
|---|---|---|---|
| $T$ | (number of leaves) | | |
| $\gamma$ | `gamma` | each extra leaf | a split must improve the loss by at least $\gamma$, otherwise it is not made |
| $\lambda$ | `reg_lambda` (default 1) | large leaf values (L2) | shrinks leaf outputs towards 0 |
| $\alpha$ | `reg_alpha` (default 0) | large leaf values (L1) | can push some leaf outputs exactly to 0 |
| $w_j$ | | the output (score) of leaf $j$ | |

### 3.2 Second-order gradients (kept light)

For every row XGBoost computes two numbers from the loss:

- $g_i$ = **gradient** (first derivative): *which direction* and how wrong;
- $h_i$ = **hessian** (second derivative): *how curved* the loss is, i.e. how confident a step can be.

For squared error, $g_i = \hat y_i - y_i$ (minus the residual) and $h_i = 1$.
Summing them over the rows in a leaf ($G = \sum g_i$, $H = \sum h_i$) gives two short formulas:

$$
\text{best leaf value: } \; w^* = -\frac{G}{H + \lambda}
\qquad\qquad
\text{split gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma
$$

Using the second derivative is like Newton's method instead of plain gradient descent: steps are better sized.

### 3.3 Worked example (squared error)

A leaf contains 3 rows whose residuals $y - \hat y$ are **−4, 6, 8**. So $g = 4, -6, -8$, $G = -10$, $H = 3$.

**Leaf value for different $\lambda$:**

| $\lambda$ | $w^* = -G/(H+\lambda)$ | Comment |
|---|---|---|
| 0 | 10 / 3 = **3.33** | the mean residual: exactly plain gradient boosting |
| 1 (default) | 10 / 4 = **2.50** | shrunk towards 0 |
| 3 | 10 / 6 = **1.67** | shrunk more: a more cautious tree |

**Should we split it into {−4} and {6, 8}?** (with $\lambda = 1$)

- left: $G_L = 4,\ H_L = 1 \Rightarrow 4^2 / 2 = 8$
- right: $G_R = -14,\ H_R = 2 \Rightarrow 14^2 / 3 = 65.3$
- no split: $G = -10,\ H = 3 \Rightarrow 10^2 / 4 = 25$
- gain $= \tfrac12 (8 + 65.3 - 25) - \gamma = 24.2 - \gamma$

With `gamma=0` the split is made. With `gamma=30` the gain is negative, so the split is **pruned**: $\gamma$ directly
controls how "eager" the tree is to grow.

### 3.4 Engineering features

| Feature | What it does | Why it helps |
|---|---|---|
| **Row subsampling** (`subsample`) | each tree sees a random fraction of the rows | less overfitting, faster (like bagging) |
| **Column subsampling** (`colsample_bytree`, `_bylevel`, `_bynode`) | each tree / level / node sees a random fraction of the features | trees are more different, less overfitting (like random forest) |
| **Missing values** | at each split, learns a *default direction* (left or right) for rows with a missing value | no need to impute `NaN` first |
| **Parallel tree building** | split search over features runs on many CPU threads; histogram method (`tree_method='hist'`, the default) bins the values | much faster than sklearn's `GradientBoostingClassifier` |
| **Early stopping** | stop adding trees when the validation score stops improving | picks `n_estimators` automatically, prevents overfitting |
| **Built-in metrics** (`eval_metric`) | logloss, auc, error, rmse, ... computed after every tree | lets you plot learning curves |

> Note: the trees are still built **one after another** (boosting is sequential). The parallelism is *inside* building each tree.

---

## 4. Early stopping

1. Split the data into **train / validation / test**.
2. Set `n_estimators` large (e.g. 1000–5000) and `learning_rate` smaller (e.g. 0.05).
3. Set `early_stopping_rounds=50` and pass `eval_set=[(X_val, y_val)]` to `fit`.
4. After every tree XGBoost computes `eval_metric` on the validation set. If it has not improved for 50 trees, it stops.
5. `best_iteration` stores the best tree count; `predict` automatically uses only those trees.

```python
model = XGBClassifier(n_estimators=2000, learning_rate=0.05, eval_metric='logloss',
                      early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
model.best_iteration          # e.g. 143
model.evals_result()          # loss after every tree -> learning curve
```

⚠️ The eval set must be a **validation set**, never the test set (see §12).

---

## 5. Key hyperparameters

| Parameter | Default | What it controls | Typical range | Higher value → |
|---|---|---|---|---|
| `n_estimators` | 100 | number of trees (boosting rounds) | 100 – 5000 (use early stopping) | more capacity, slower |
| `learning_rate` (`eta`) | 0.3 | how much each tree contributes | 0.01 – 0.3 | faster learning, more overfit risk |
| `max_depth` | 6 | depth of each tree | 3 – 10 | more complex interactions, more overfit |
| `min_child_weight` | 1 | minimum sum of hessians in a leaf (≈ min samples) | 1 – 10 | more conservative |
| `subsample` | 1.0 | fraction of rows per tree | 0.5 – 1.0 | less randomness |
| `colsample_bytree` | 1.0 | fraction of features per tree | 0.5 – 1.0 | less randomness |
| `gamma` | 0 | minimum gain to make a split | 0 – 5 | more pruning, simpler trees |
| `reg_lambda` | 1 | L2 penalty on leaf values | 0 – 10 | smaller leaf values |
| `reg_alpha` | 0 | L1 penalty on leaf values | 0 – 10 | sparser leaf values |
| `scale_pos_weight` | 1 | weight of the positive class (binary) | ≈ negatives / positives | more recall on the minority class |
| `early_stopping_rounds` | None | patience for early stopping | 20 – 100 | trains longer before stopping |
| `eval_metric` | depends on objective | metric watched on `eval_set` | `logloss`, `auc`, `error`, `rmse`, `mae` | |

**`learning_rate` and `n_estimators` work together:** halve the learning rate → you need roughly twice as many trees.
Small learning rate + many trees + early stopping is usually best (just slower).

---

## 6. A recommended tuning order

Do not tune everything at once. A simple, practical order:

1. **Fix** `learning_rate = 0.1` (or 0.05) and use **early stopping** to find `n_estimators`.
2. Tune the **tree shape**: `max_depth` and `min_child_weight`.
3. Tune the **randomness**: `subsample` and `colsample_bytree` (try 0.6 – 1.0).
4. Tune **regularisation**: `gamma`, `reg_lambda`, `reg_alpha`.
5. **Lower** `learning_rate` (e.g. 0.01 – 0.05) and let early stopping add more trees for the final model.

Use `RandomizedSearchCV` (or a tool like Optuna) instead of a huge `GridSearchCV`. Expect small gains: the biggest
improvement usually comes from step 1 (sensible learning rate + early stopping). In `02-xgboost-churn.ipynb` a 25-trial
random search did **not** beat a hand-chosen early-stopping model.

---

## 7. Imbalanced classes

When one class is rare (e.g. 20% churners), accuracy is misleading. Two simple tools:

| Method | How | Effect |
|---|---|---|
| `scale_pos_weight` | set to $\dfrac{\#\text{negatives}}{\#\text{positives}}$ (e.g. 7963 / 2037 ≈ 3.9) | positive rows count more in the loss; recall ↑, precision ↓ |
| **Threshold tuning** | predict churn if `predict_proba ≥ t` with $t < 0.5$; choose $t$ on the **validation set** (e.g. max F1) | same trade-off, without retraining; probabilities stay unchanged |

Judge with **recall, precision, F1, ROC-AUC** (or PR-AUC), not accuracy. ROC-AUC does not depend on the threshold at all.

---

## 8. XGBoost vs Gradient Boosting vs Random Forest

| | Random Forest | Gradient Boosting (sklearn) | XGBoost |
|---|---|---|---|
| How trees are combined | **parallel**, independent, averaged (bagging) | **sequential**, each fixes the previous | **sequential**, each fixes the previous |
| Typical trees | deep, fully grown | shallow (depth 3) | shallow to medium (depth 6 default) |
| Main error reduced | variance | bias | bias (and variance via regularisation) |
| Regularisation | via randomness | learning rate, subsampling | learning rate, subsampling, **γ, λ, α** |
| Missing values | supported in recent scikit-learn (≥ 1.4) | `GradientBoostingClassifier`: impute first (`HistGradientBoosting*` handles `NaN`) | **built in** (learns default direction) |
| Early stopping | not applicable | `n_iter_no_change` | `early_stopping_rounds` + `eval_set` |
| Speed on big data | good (parallel trees) | slow (exact, single-threaded) | **fast** (histograms, multi-threaded, GPU) |
| Tuning effort | low: works well out of the box | medium | **higher**: many knobs |
| Overfitting risk | low | medium | medium (controlled with early stopping) |

**Rule of thumb:** Random Forest is a great quick baseline. A tuned XGBoost (or LightGBM / CatBoost) usually wins on
large tabular data, but on small, easy data the difference can be zero. In `01-xgboost-breast-cancer.ipynb`
all three scored ~96–97% and Random Forest was not worse.

---

## 9. The other two big boosting libraries

- **LightGBM** (Microsoft): grows trees *leaf-wise* and uses aggressive histogram tricks; usually the **fastest** on very large data.
- **CatBoost** (Yandex): handles **categorical features** natively and well (ordered target statistics); good defaults, little tuning.
- (Also scikit-learn's own **`HistGradientBoostingClassifier`**, inspired by LightGBM.)

All three follow the same gradient-boosting idea; results are often close, so pick the one that fits your data and workflow.

---

## 10. scikit-learn API cheat sheet

```python
from xgboost import XGBClassifier, XGBRegressor

# Classification (labels must be 0, 1, ..., n_classes-1)
clf = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, gamma=0,
    scale_pos_weight=1,            # ≈ neg/pos for imbalanced binary problems
    eval_metric='logloss',         # or 'auc', 'error'
    early_stopping_rounds=50,      # needs eval_set in fit
    random_state=42, n_jobs=-1)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
clf.predict(X_test)                # classes
clf.predict_proba(X_test)[:, 1]    # probability of class 1
clf.best_iteration                 # best number of trees - 1
clf.evals_result()                 # {'validation_0': {'logloss': [...]}}
clf.feature_importances_           # gain-based by default

# Regression
reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                   eval_metric='rmse', early_stopping_rounds=50)
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Works with all sklearn tools
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
cross_val_score(XGBClassifier(), X, y, cv=5)
```

Install with `pip install xgboost`. In current XGBoost (deprecated in `fit` since 2.0, removed in 3.x), `eval_metric` and `early_stopping_rounds` go in the **constructor**,
not in `fit`.

---

## 11. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Often the most accurate model on tabular data | Many hyperparameters; easy to overfit without early stopping |
| Fast (multi-threaded, histogram, GPU) | Less interpretable than a single tree or linear model |
| Built-in regularisation and missing-value handling | Not the right tool for images, audio, raw text (use deep learning) |
| Early stopping and evaluation built in | Cannot extrapolate beyond the target range seen in training (like all trees) |
| scikit-learn compatible | Gains over Random Forest can be small on small/easy data |

---

## 12. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| **Class labels not starting at 0** (e.g. `2/4`, `1/2`, strings) | `ValueError: Invalid classes inferred ... Expected: [0 1], got [2 4]` | encode with `.map({2: 0, 4: 1})` or `LabelEncoder` |
| **Keeping ID columns** (`CustomerId`, `RowNumber`, `Sample code number`, `Surname`) | the model learns fake patterns from identifiers | drop them before training |
| **Early stopping on the test set** | the test set chose the number of trees → test score is optimistic | use a separate **validation** set; touch the test set once at the end |
| Choosing the threshold on the test set | same leakage as above | choose it on validation |
| Judging an imbalanced problem by accuracy | 80% "accuracy" by predicting the majority class | use recall / precision / F1 / ROC-AUC |
| High `learning_rate` with many trees, no early stopping | strong overfitting (validation loss rises) | lower `learning_rate`, use early stopping |
| Passing `early_stopping_rounds` to `fit()` | `TypeError` in current XGBoost (3.x; deprecated since 2.0) | pass it to the constructor |
| Scaling features "for XGBoost" | harmless but pointless | trees do not need scaling |

---

## 13. Quick revision questions

1. What does the "X" (eXtreme) add on top of plain gradient boosting? *(Regularised objective, second-order gradients, subsampling, missing-value handling, parallel/histogram tree building, early stopping.)*
2. What do `gamma`, `reg_lambda` and `reg_alpha` penalise? *(γ: each extra leaf / minimum split gain; λ: squared leaf values (L2); α: absolute leaf values (L1).)*
3. In the worked example, why did the leaf value fall from 3.33 to 2.50 when λ went from 0 to 1? *(w* = −G/(H+λ): a larger denominator shrinks the leaf output towards 0.)*
4. How does early stopping choose `n_estimators`? *(It watches the validation metric after each tree and stops after `early_stopping_rounds` trees without improvement; `best_iteration` is kept.)*
5. Why must the eval set not be the test set? *(The test data would influence the model, so the test score would be optimistic.)*
6. How do you set `scale_pos_weight`? *(≈ number of negatives / number of positives.)*
7. If you halve `learning_rate`, what should happen to `n_estimators`? *(Roughly double; early stopping finds it.)*
8. Why does XGBoost's classifier fail on labels 2 and 4? *(It requires classes 0 … n−1.)*
9. Random Forest vs XGBoost: which builds trees in parallel and independently? *(Random Forest.)*

---

## 14. Summary

- **XGBoost = gradient boosting + regularisation + speed + convenience.**
- Each new tree fits the gradients of the loss; leaf values $w^* = -G/(H+\lambda)$ and splits must gain more than $\gamma$.
- Key knobs: `learning_rate` + `n_estimators` (with **early stopping**), `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_lambda`, `reg_alpha`, `scale_pos_weight`.
- Workflow: **train / validation / test** → early stopping on validation → handle imbalance → tune lightly → evaluate on test **once**.
- Encode labels to **0 … n−1**, drop **ID columns**, and do not expect miracles on small, easy datasets.
