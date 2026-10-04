# Gradient Boosting: Theory

> Builds on decision trees (regression trees) and on [`../05-adaboost/`](../05-adaboost/) (the idea of *boosting*:
> models trained one after another, each fixing the previous ones).

---

## 1. The intuition: fix the leftover mistakes

Imagine guessing the price of houses.

1. **First guess:** you know nothing, so you say the **average price** for every house.
2. **Look at your mistakes:** for each house, *actual − guess*. These leftovers are called **residuals**.
3. **Learn the mistakes:** train a **small tree** whose job is to predict those residuals ("big houses: +20, small houses: −15").
4. **Correct the guess:** new guess = old guess + (a fraction of) the tree's correction.
5. **Repeat:** compute the new residuals, train another small tree on them, add it … 100s of times.

```
 prediction = mean  +  η·tree1  +  η·tree2  +  η·tree3  + ...
              │         │           │           │
          first guess  fixes the   fixes what  fixes what
                       mean's      is still    is still
                       mistakes    wrong       wrong
```

- Each tree is **weak** (small, e.g. depth 3 or 8 leaves), but together they become very accurate.
- η (eta) is the **learning rate**: we only add a fraction of each correction (e.g. 0.1), so we take many small careful steps.

Analogy: a golfer. The first shot lands somewhere near the hole. Every next shot is aimed at **the remaining distance**,
not at the hole from the start.

---

## 2. Why is it called *gradient* boosting?

We measure how wrong we are with a **loss function**. For regression the usual one is squared error:

$$L(y, F) = \tfrac{1}{2}(y - F)^2$$

Its **gradient** (slope) with respect to the prediction $F$ is

$$\frac{\partial L}{\partial F} = -(y - F)$$

So the **negative gradient** is $y - F$, which is **exactly the residual**.

- Gradient descent says: to reduce the loss, move **in the direction of the negative gradient**.
- Gradient boosting does this in "prediction space": each tree learns the negative gradient (the residuals) and we
  move the predictions a small step (η) in that direction.

So "fit a tree to the residuals" = "take a gradient-descent step, using a tree". For other losses (absolute error,
Huber, log-loss for classification) the negative gradient is something else (a "pseudo-residual"), but the recipe is the same.
That generality is the real power of the method: **any differentiable loss works**.

---

## 3. The algorithm (regression, squared error)

Given training data $(x_i, y_i)$, $i = 1 \dots n$, a learning rate $\eta$ and a number of trees $M$:

1. **Start with a constant:** $F_0(x) = \bar y$ (the mean of $y$; it minimises squared error among constants).
2. **For** $m = 1, 2, \dots, M$:
   1. Residuals (= negative gradients): $r_{i} = y_i - F_{m-1}(x_i)$
   2. Fit a small regression tree $h_m$ to the pairs $(x_i, r_i)$. Each leaf predicts the **average residual** of the rows in it.
   3. Update: $F_m(x) = F_{m-1}(x) + \eta \, h_m(x)$
3. **Final model:** $F_M(x) = \bar y + \eta \sum_{m=1}^{M} h_m(x)$

Notebook 1 writes exactly this in about ten lines of Python and gets the **same predictions** as scikit-learn's
`GradientBoostingRegressor` (maximum difference 2.2 × 10⁻¹⁶, i.e. floating-point rounding).

---

## 4. Worked example (by hand)

Five houses, one feature (size, in 100 m²) and the price (in lakh). Learning rate η = 0.1, trees are **stumps** (depth 1).

| House | size $x$ | price $y$ |
|:-:|:-:|:-:|
| A | 1 | 10 |
| B | 2 | 12 |
| C | 3 | 20 |
| D | 4 | 22 |
| E | 5 | 26 |

**Step 1: first prediction = mean.** $F_0 = (10 + 12 + 20 + 22 + 26)/5 = 90/5 = 18$ for every house.

**Step 2: residuals** $r = y - 18$:

| House | A | B | C | D | E |
|---|:-:|:-:|:-:|:-:|:-:|
| residual $r_1$ | −8 | −6 | 2 | 4 | 8 |

MSE now: $(64 + 36 + 4 + 16 + 64)/5 = 184/5 = 36.8$.

**Step 3: fit a stump on the residuals.** Try every split and keep the one with the smallest squared error (SSE) of the
residuals around each side's mean:

| Split | left mean | right mean | SSE |
|---|:-:|:-:|:-:|
| $x \le 1.5$ | −8 | 2 | 104 |
| $x \le 2.5$ | **−7** | **4.667** | **20.67** ← best |
| $x \le 3.5$ | −4 | 6 | 64 |
| $x \le 4.5$ | −2 | 8 | 104 |

So tree 1 says: "if size ≤ 2.5 the correction is −7, otherwise +4.667" (14/3).

**Step 4: update with η = 0.1.**

| House | $F_0$ | tree 1 | $F_1 = F_0 + 0.1 \times$ tree 1 | new residual $y - F_1$ |
|---|:-:|:-:|:-:|:-:|
| A | 18 | −7 | **17.3** | −7.3 |
| B | 18 | −7 | **17.3** | −5.3 |
| C | 18 | 4.667 | **18.467** | 1.533 |
| D | 18 | 4.667 | **18.467** | 3.533 |
| E | 18 | 4.667 | **18.467** | 7.533 |

MSE drops from 36.8 to **30.59**. A small step in the right direction; tree 2 would now be trained on the new
residuals (−7.3, −5.3, 1.533, 3.533, 7.533), and so on.

With η = 1 the same tree would give predictions 11, 11, 22.667, 22.667, 22.667 and MSE 4.13 in one go: much faster on
the training data, but with real, noisy data such big steps quickly start fitting noise (see §5).

Verified with code (scikit-learn gives the same numbers):

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
X = np.array([[1], [2], [3], [4], [5]]); y = np.array([10, 12, 20, 22, 26])
gb = GradientBoostingRegressor(n_estimators=1, learning_rate=0.1, max_depth=1).fit(X, y)
gb.predict(X)    # [17.3, 17.3, 18.4667, 18.4667, 18.4667]
```

---

## 5. Learning rate and number of trees (shrinkage)

`learning_rate` (η) scales every tree's contribution. This is called **shrinkage**.

| learning_rate | Effect |
|---|---|
| large (e.g. 1.0) | learns fast, few trees needed, but each tree over-corrects → overfits quickly |
| small (e.g. 0.05–0.1) | learns slowly, **needs more trees**, usually generalises as well or better |
| tiny (e.g. 0.01) | very slow; needs thousands of trees |

`learning_rate` and `n_estimators` **work together**: halving one roughly means doubling the other.

From notebook 2 (`make_friedman1`, 1000 trees, test MSE):

| learning_rate | best test MSE | reached after | test MSE after 1000 trees |
|:-:|:-:|:-:|:-:|
| 1.0 | 3.61 | 10 trees | 4.29 |
| 0.3 | 1.89 | 84 trees | 2.00 |
| 0.1 | 1.93 | 243 trees | 1.96 |
| 0.05 | **1.83** | 963 trees | 1.84 |
| 0.01 | 2.08 | 998 trees (still improving) | 2.08 |

Unlike a random forest, **more trees can overfit** in gradient boosting: training error always goes down, but test
error eventually flattens or rises. In notebook 1 (y = 3x² + noise, 8-leaf trees) the test MSE was lowest after 4 trees
with η = 1 (0.0037), and after 11 trees (η = 0.3) or 33 trees (η = 0.1) at a lower 0.0029. After 200 trees all three had
overfitted (0.0038–0.0041). So pick the number of trees with validation data or early stopping (§8).

---

## 6. Tree size: `max_depth` / `max_leaf_nodes`

Each tree should be **weak**: small trees, many of them.

| Parameter | Meaning | Typical |
|---|---|---|
| `max_depth` | maximum depth of each tree (default **3**) | 2–5 |
| `max_leaf_nodes` | maximum number of leaves (best-first growth) | 4–32 |
| `min_samples_leaf` | minimum rows per leaf; larger = smoother | 1–50 |

- Depth 1 (stumps): each tree uses one feature → cannot model **interactions** well.
- Depth $d$ allows interactions between up to $d$ features.
- Deep trees overfit fast. Notebook 2: depth 1 → test R² 0.875, depth 2–3 → 0.910–0.911, depth 8 → 0.833.

> ⚠️ Gotcha (seen in notebook 1): `GradientBoostingRegressor(max_leaf_nodes=8)` **still keeps the default
> `max_depth=3`**, so its trees can be shallower than a `DecisionTreeRegressor(max_leaf_nodes=8)`. Set `max_depth=None`
> if you want only the leaf limit.

---

## 7. Stochastic gradient boosting: `subsample`

With `subsample < 1.0`, each tree is trained on a **random fraction of the rows** (without replacement).

- Adds randomness (like bagging) → can reduce overfitting.
- Each tree trains faster.
- Typical values: 0.5–0.8.

Notebook 2: subsample 1.0 / 0.8 / 0.5 → test R² 0.911 / 0.918 / 0.920 (a small gain on one split; worth trying, not guaranteed).

---

## 8. Early stopping

Let the model choose the number of trees:

```python
GradientBoostingRegressor(n_estimators=5000,        # just an upper limit
                          validation_fraction=0.1,  # 10% of the TRAINING data held out
                          n_iter_no_change=10,      # stop after 10 trees without improvement
                          tol=1e-4)
model.n_estimators_                                 # trees actually built
```

Notebook 2 (Friedman): early stopping used **178** trees and reached test R² 0.915 (defaults with 100 trees: 0.902).
On the small breast-cancer data it stopped at 51 trees and scored slightly *lower* (0.937 vs 0.958): with only ~43
validation rows the stopping point is noisy. Early stopping shines on larger datasets.

---

## 9. Classification (briefly)

For classification we cannot "add up" class labels. Instead the model adds up a **score** $F(x)$ = the **log-odds** of class 1:

$$F(x) = F_0 + \eta \sum_m h_m(x), \qquad P(y=1 \mid x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

- **Loss:** log-loss (binary cross-entropy).
- **Start:** $F_0 = \log\frac{p}{1-p}$, the log-odds of the class balance $p$. Breast cancer training set: $p = 0.627$, $F_0 = 0.518$.
- **Negative gradient (pseudo-residual):** $y - p$ (actual label 0/1 minus predicted probability). Each tree is still a
  **regression** tree fitted to these values (scikit-learn then adjusts the leaf values with a Newton step).
- `decision_function` returns $F(x)$; `predict_proba` returns $\sigma(F(x))$. Notebook 2 checks that they match.
- More than 2 classes: one tree per class per round (with softmax).

---

## 10. AdaBoost vs Gradient Boosting

| | AdaBoost | Gradient Boosting |
|---|---|---|
| How the next model focuses on mistakes | **re-weights rows**: misclassified rows get bigger weights | **fits the residuals** (negative gradient) directly |
| Weak learner | usually stumps (depth 1) | small trees (depth 3, 8–32 leaves) |
| Combining | weighted vote, weight α per model from its error | sum of trees × learning rate |
| Loss | exponential loss (implicitly) | any differentiable loss (squared, absolute, Huber, log-loss, …) |
| Starting point | no initial prediction | a constant (mean, or log-odds) |
| Sensitivity to outliers / label noise | high | lower with robust losses (Huber, absolute) |
| Main knobs | `n_estimators`, `learning_rate` | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

Both are boosting: models built **sequentially**, each fixing the previous ones. AdaBoost can be seen as a special case
of gradient boosting with exponential loss.

---

## 11. HistGradientBoosting (the fast version)

`HistGradientBoostingRegressor` / `HistGradientBoostingClassifier`:
- First **bins** every feature into at most 255 bins, so finding splits is much faster. Recommended for large data
  (tens of thousands of rows and more).
- Handles **missing values** natively and supports categorical features.
- Uses `max_iter` instead of `n_estimators`; early stopping is on by default when there are more than 10,000 rows.
- Similar in spirit to LightGBM. Notebook 2: test R² 0.905 on Friedman, accuracy 0.972 on breast cancer, with defaults.

XGBoost ([`../07-xgboost/`](../07-xgboost/)) is another fast, regularised implementation of the same idea.

---

## 12. scikit-learn cheat sheet

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

reg = GradientBoostingRegressor(
    loss='squared_error',    # also 'absolute_error', 'huber', 'quantile'
    n_estimators=100,        # number of trees (boosting rounds)
    learning_rate=0.1,       # shrinkage
    max_depth=3,             # size of each tree
    subsample=1.0,           # < 1.0 → stochastic gradient boosting
    n_iter_no_change=None,   # e.g. 10 → early stopping
    validation_fraction=0.1, # used only with early stopping
    random_state=42)
reg.fit(X_train, y_train)
reg.predict(X_test)
reg.staged_predict(X_test)     # predictions after 1, 2, ..., n trees (a generator)
reg.feature_importances_
reg.estimators_                # array of trees, shape (n_estimators, 1)
reg.init_                      # the starting model (predicts the mean)
reg.n_estimators_              # trees actually used (after early stopping)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X_train, y_train)
clf.predict_proba(X_test)
clf.decision_function(X_test)  # raw log-odds score F(x)
clf.staged_predict_proba(X_test)
```

No feature scaling is needed (trees).

---

## 13. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Often the most accurate method on tabular data | Trees are built **one after another**: cannot be parallelised across trees, slower to train than a random forest |
| Works with any differentiable loss (robust losses, quantiles, log-loss) | More hyperparameters to tune (learning rate, trees, depth, subsample) |
| Shallow trees + shrinkage → good bias–variance balance | **Can overfit** with too many trees or a high learning rate |
| No scaling needed; gives feature importance | Sensitive to noisy targets/outliers with squared loss |
| `staged_predict` and early stopping make choosing the number of trees easy | Like all trees: cannot extrapolate beyond the training range |

Notebook 2: on Friedman regression it clearly beat a random forest (test R² 0.902 default / 0.911 tuned vs 0.830); on
breast cancer they tied (0.958).

---

## 14. Common mistakes

- **Lowering the learning rate without adding trees** → underfitting (lr = 0.01 with 100 trees: test MSE 9.35 in notebook 2).
- **Using `learning_rate=1.0`** with many trees → overfitting (worst setting in both notebooks).
- **Assuming more trees can't hurt** (true for random forests, *not* for boosting).
- **Choosing `n_estimators` on the test set.** Use cross-validation or early stopping on a validation split.
- **Deep trees** (`max_depth=8+`) in boosting → overfit quickly. Keep them shallow.
- **Hard-coding numbers** (e.g. typing the mean `0.265458` instead of `y.mean()`).
- **Forgetting the learning rate when computing residuals by hand**: residuals must be $y - F_{m-1}$ where $F$ already
  includes η × each tree. (The original version of notebook 1 had this bug.)
- Expecting `max_leaf_nodes` to remove the default `max_depth=3` limit (it does not, §6).

---

## 15. Quick revision questions

1. *What is the first prediction of gradient boosting for regression?* The mean of y.
2. *What does each new tree learn?* The residuals (negative gradients) of the current model.
3. *Why "gradient"?* For squared loss, the residual $y - F$ is the negative gradient of the loss; each tree is a gradient-descent step.
4. *What does the learning rate do?* Scales each tree's contribution (shrinkage); smaller = more trees needed, usually better generalisation.
5. *Can gradient boosting overfit with too many trees?* Yes; use early stopping / validation.
6. *What is stochastic gradient boosting?* Training each tree on a random subsample of the rows (`subsample < 1`).
7. *How does early stopping work in scikit-learn?* `n_iter_no_change` + `validation_fraction`: stop when the validation score stops improving; `n_estimators_` stores the number used.
8. *What do the trees add up in classification?* Log-odds; the probability is the sigmoid of the sum.
9. *AdaBoost vs gradient boosting in one line?* AdaBoost re-weights rows; gradient boosting fits residuals (gradients) of any loss.
10. *In the worked example, what is the prediction for house A after one tree with η = 0.1?* 18 + 0.1 × (−7) = 17.3.

---

## Summary

- **Gradient boosting = start from a constant, then repeatedly fit a small tree to the residuals and add η × tree.**
- The residual is the **negative gradient** of the loss → it is gradient descent, one tree at a time, and works for any differentiable loss.
- Key knobs: `n_estimators` × `learning_rate` (tune together), `max_depth` (keep small), `subsample`, early stopping.
- Training error always falls; test error has a minimum, so choose the number of trees with validation data.
- For classification it adds up **log-odds** and applies a sigmoid.
- `HistGradientBoosting*` (and XGBoost / LightGBM) are faster implementations of the same idea.
