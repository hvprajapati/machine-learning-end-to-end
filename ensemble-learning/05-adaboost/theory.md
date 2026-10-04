# AdaBoost: Theory

> Companion notebooks: [`01-adaboost-step-by-step.ipynb`](01-adaboost-step-by-step.ipynb) (the algorithm by hand) and
> [`02-adaboost-hyperparameters.ipynb`](02-adaboost-hyperparameters.ipynb) (sklearn, tuning, comparisons).

## Contents
1. [What is boosting?](#1-what-is-boosting)
2. [Weak learners and the bias–variance view](#2-weak-learners-and-the-biasvariance-view)
3. [The AdaBoost algorithm, step by step](#3-the-adaboost-algorithm-step-by-step)
4. [Worked numeric example](#4-worked-numeric-example)
5. [How prediction works](#5-how-prediction-works)
6. [Hyperparameters: `n_estimators` and `learning_rate`](#6-hyperparameters-n_estimators-and-learning_rate)
7. [The base estimator](#7-the-base-estimator)
8. [AdaBoostRegressor (briefly)](#8-adaboostregressor-briefly)
9. [Strengths and limitations](#9-strengths-and-limitations)
10. [Bagging vs boosting](#10-bagging-vs-boosting)
11. [sklearn cheat sheet](#11-sklearn-cheat-sheet)
12. [Common mistakes](#12-common-mistakes)
13. [Quick revision questions](#13-quick-revision-questions)
14. [Summary](#14-summary)

---

## 1. What is boosting?

**Boosting** builds an ensemble **sequentially**: model 2 is trained *after* model 1 and tries to fix model 1's mistakes,
model 3 fixes what is still wrong, and so on. At the end, all models vote.

> **Analogy.** A student takes a practice test, then spends the next study session mostly on the questions they got
> wrong. After a few rounds, the weak spots are covered. Each study session is a "weak learner"; the student's final
> knowledge is the ensemble.

**AdaBoost** (*Adaptive Boosting*, Freund & Schapire, 1997) was the first very successful boosting algorithm.
It is "adaptive" because it changes the **weight of every training row** after each model:

- rows the last model got **wrong** become **heavier** → the next model pays more attention to them;
- rows it got **right** become **lighter**.

It also gives each model a **say** (weight α) in the final vote: accurate models get a louder voice.

## 2. Weak learners and the bias–variance view

A **weak learner** is a model that is only a bit better than random guessing. The classic choice in AdaBoost is a
**decision stump**: a decision tree with `max_depth=1` (one question, two leaves, e.g. "is X2 ≤ 2.5?").

| | Decision stump | Fully grown tree |
|---|---|---|
| Bias | **high** (one straight cut, too simple) | low |
| Variance | **low** (barely changes with new data) | high (memorises noise) |
| Typical problem | underfitting | overfitting |

Boosting starts from models with **high bias, low variance** and adds them up. Each new stump corrects part of the
remaining error, so the combined model becomes more flexible: **boosting mainly reduces bias.**
(Compare: bagging/random forest start from low-bias, high-variance deep trees and average them to **reduce variance**.)

Example from notebook 01: one stump can only draw one straight line, but three stumps together draw a "staircase"
boundary that no single stump can make.

## 3. The AdaBoost algorithm, step by step

Data: n rows (xᵢ, yᵢ), with labels written as **yᵢ ∈ {−1, +1}**. We train M weak learners h₁ … h_M.

**Step 1: equal weights.**
$$w_i = \frac{1}{n} \quad \text{for every row}$$

**Repeat for m = 1 … M:**

**Step 2: train a weak learner** hₘ on the data, taking the weights into account. Two ways:
- *reweighting*: pass the weights to the learner (`fit(X, y, sample_weight=w)`), which is what sklearn does;
- *resampling* ("upsampling"): draw a new dataset of n rows **with replacement**, where row i is picked with
  probability wᵢ (cumulative ranges + random numbers), then train on it. This is what the CampusX lecture shows.

**Step 3: weighted error.** Add up the weights of the rows the model gets wrong (always measured on the **original**
rows with their current weights):
$$\varepsilon_m = \sum_{i:\ h_m(x_i) \neq y_i} w_i$$

**Step 4: model weight ("amount of say").**
$$\alpha_m = \frac{1}{2}\ln\!\left(\frac{1-\varepsilon_m}{\varepsilon_m}\right)$$

| error ε | α | meaning |
|---|---|---|
| 0.05 | +1.47 | very good model, loud vote |
| 0.30 | +0.42 | decent model |
| 0.50 | 0 | coin flip, no say |
| 0.70 | −0.42 | worse than random, its vote is flipped |

**Step 5: update the row weights.**
$$w_i \leftarrow w_i \cdot e^{-\alpha_m y_i h_m(x_i)} =
\begin{cases} w_i\, e^{-\alpha_m} & \text{if row } i \text{ is correct } (y_i h_m(x_i)=+1)\\ w_i\, e^{+\alpha_m} & \text{if row } i \text{ is wrong } (y_i h_m(x_i)=-1)\end{cases}$$

**Step 6: normalise** so the weights sum to 1: $w_i \leftarrow w_i / \sum_j w_j$.

> **Handy fact.** After normalising, the rows that hₘ got wrong always hold **exactly half** of the total weight,
> and the correct rows hold the other half. So on the new weights, hₘ is a pure coin flip, and the next model is forced
> to learn something different.

**Final model:**
$$H(x) = \operatorname{sign}\left(\sum_{m=1}^{M} \alpha_m\, h_m(x)\right)$$

(Behind the scenes, AdaBoost is "forward stagewise" fitting of an additive model $F(x)=\sum_m \alpha_m h_m(x)$ that
minimises the **exponential loss** $e^{-yF(x)}$. You will meet the general version of this idea in gradient boosting.)

## 4. Worked numeric example

The toy data from notebook 01 (labels 1 → +1, 0 → −1):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| X1 | 1 | 2 | 3 | 4 | 5 | 6 | 6 | 7 | 9 | 9 |
| X2 | 5 | 3 | 6 | 8 | 1 | 9 | 5 | 8 | 9 | 2 |
| y | +1 | +1 | −1 | +1 | −1 | +1 | −1 | +1 | −1 | −1 |

All numbers below were computed by code in notebook 01 (fractions shown where they are exact).

**Round 1.** All weights = 0.1. Stump 1: `X2 ≤ 2.5 → −1, else +1`. Wrong rows: 2, 6, 8.
- ε₁ = 0.1 + 0.1 + 0.1 = **0.3**
- α₁ = ½ ln(0.7 / 0.3) = **0.4236**
- correct rows: 0.1 × e^(−0.4236) = 0.0655; wrong rows: 0.1 × e^(0.4236) = 0.1528; sum = 7(0.0655) + 3(0.1528) = 0.9165
- normalised: correct rows **0.0714** (= 1/14), wrong rows **0.1667** (= 1/6). Check: 3 × 1/6 = 0.5 ✔ (handy fact)

**Round 2.** Stump 2: `X1 ≤ 2.5 → +1, else −1`. Wrong rows: 3, 5, 7 (each 1/14).
- ε₂ = 3/14 = **0.2143**, α₂ = ½ ln(11/3) = **0.6496**
- new weights: rows 3, 5, 7 → **0.1667** (1/6); rows 2, 6, 8 → **0.1061** (7/66); rows 0, 1, 4, 9 → **0.0455** (1/22)

**Round 3.** Stump 3: `X2 ≤ 7 → −1, else +1`. Wrong rows: 0, 1, 8.
- ε₃ = 1/22 + 1/22 + 7/66 = 13/66 = **0.1970**, α₃ = ½ ln(53/13) = **0.7027**
- It misses 3 rows, yet its error is the lowest so far: two of those rows were light (0.0455 each).
  **Error counts weight, not rows.**

Weight table (weight each row had when that stump was judged; **bold** = that stump got it wrong):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| round 1 | 0.1 | 0.1 | **0.1** | 0.1 | 0.1 | 0.1 | **0.1** | 0.1 | **0.1** | 0.1 |
| round 2 | 0.0714 | 0.0714 | 0.1667 | **0.0714** | 0.0714 | **0.0714** | 0.1667 | **0.0714** | 0.1667 | 0.0714 |
| round 3 | **0.0455** | **0.0455** | 0.1061 | 0.1667 | 0.0455 | 0.1667 | 0.1061 | 0.1667 | **0.1061** | 0.0455 |
| after round 3 | 0.1154 | 0.1154 | 0.0660 | 0.1038 | 0.0283 | 0.1038 | 0.0660 | 0.1038 | 0.2692 | 0.0283 |

**Prediction.**

| point | votes (h₁, h₂, h₃) | score Σ αₘhₘ | sign | true |
|---|---|---|---|---|
| (1, 5) | +1, +1, −1 | 0.4236 + 0.6496 − 0.7027 = **+0.3706** | +1 | +1 ✅ |
| (9, 9) | +1, −1, +1 | 0.4236 − 0.6496 + 0.7027 = **+0.4767** | +1 | −1 ❌ |

The 3-stump model gets **9/10** training rows right; only row 8 = (9, 9) is wrong, and it is now the heaviest row
(0.2692), so a 4th stump would focus on it.

## 5. How prediction works

1. Every weak learner votes **+1 or −1** (that is why labels must be ±1, not 0/1).
2. Each vote is multiplied by the model's α.
3. Add them up and take the **sign**. The size of the sum is a confidence score (`decision_function` in sklearn).

It is a **weighted majority vote**: a stump with α = 0.70 counts more than one with α = 0.42.
Two weaker models can still outvote one strong model (0.4236 + 0.6496 > 0.7027 above).

**sklearn's version (SAMME).** sklearn uses
$\alpha_m = \eta\left[\ln\frac{1-\varepsilon_m}{\varepsilon_m} + \ln(K-1)\right]$ (η = `learning_rate`, K = number of
classes) and multiplies only the wrong rows by $e^{\alpha_m}$. For K = 2 and η = 1 this is exactly **2 × our α**, and
after normalising the row weights are identical. Scaling every α by 2 does not change any sign, so predictions are
the same. (Notebook 01 verifies this to 4 decimals.) The ln(K − 1) term is what lets SAMME work for more than 2 classes.

**Resampling vs reweighting.** Both make the next learner focus on heavy rows. Resampling is random (fix a seed!);
reweighting is deterministic. Results can differ in the details (notebook 01: our resampled run reached 9/10, sklearn's
reweighted run 10/10), but the algorithm is the same.

## 6. Hyperparameters: `n_estimators` and `learning_rate`

| Hyperparameter | What it does | Too small | Too large |
|---|---|---|---|
| `n_estimators` (M) | number of weak learners (rounds) | underfits (too few corrections) | slowly overfits noisy data, slower |
| `learning_rate` (η) | shrinks every α (and therefore every weight update) by η | needs many more estimators | each model makes big jumps; can be unstable/overfit |

They work **together**: a smaller learning rate takes smaller steps, so you need **more estimators** to reach the same
fit. A popular recipe is *small learning rate (e.g. 0.1) + many estimators*, which often generalises a little better
than *η = 1 + few estimators*, at the cost of training time. This idea is called **shrinkage**
([discussion on shrinkage in AdaBoost](https://stats.stackexchange.com/questions/82323/shrinkage-parameter-in-adaboost)).
It is not guaranteed: in notebook 02, 1500 stumps at η = 0.1 (CV 0.830) were no better than the default 50 stumps at
η = 1.0 (CV 0.838), and η = 2.0 broke the model completely (it over-corrects every round; test accuracy 0.50).
Tune them together with cross-validation (`GridSearchCV`), as in notebook 02.

AdaBoost often keeps improving on the test set for a long time even after training error is 0 ("margin" effect), but
on **noisy** data it can overfit, because it keeps raising the weight of mislabelled rows.

## 7. The base estimator

- Default in sklearn: `DecisionTreeClassifier(max_depth=1)` (a stump).
- Any classifier that accepts `sample_weight` in `fit` works (e.g. a shallow tree, logistic regression, Naive Bayes).
- Deeper trees (`max_depth=2, 3`) let each round capture feature interactions → fewer rounds needed, but more risk of
  overfitting. The base learner should stay **weak**: if it is already strong (a fully grown tree), its weighted error
  can be 0 and boosting stops after one round.
- Tune its depth inside `GridSearchCV` with the `estimator__max_depth` parameter name.

## 8. AdaBoostRegressor (briefly)

sklearn's `AdaBoostRegressor` implements **AdaBoost.R2** (Drucker, 1997):

1. Train a regressor (default: `DecisionTreeRegressor(max_depth=3)`) on a weighted bootstrap sample of the data.
2. Each row's loss is its error scaled to [0, 1] (`loss='linear'`, `'square'` or `'exponential'`).
3. Average loss → model confidence; rows with a big loss get more weight next round.
4. Prediction = **weighted median** of all models' predictions (not a weighted mean).

Use it the same way: `n_estimators`, `learning_rate`, `estimator`. In practice, gradient boosting is more common for
regression.

## 9. Strengths and limitations

**Strengths**
- Turns very simple models (stumps) into a strong classifier; simple idea, few hyperparameters.
- Reduces bias; works well on clean, tabular data.
- No feature scaling needed with tree-based weak learners.
- Gives a confidence score (`decision_function`) and feature importances.

**Limitations**
- **Sensitive to noise and outliers:** a mislabelled row is misclassified again and again, so its weight grows
  exponentially and later models waste effort on it.
- **Sequential:** each round needs the previous one, so it cannot be trained in parallel (unlike bagging).
- Usually beaten by gradient boosting / XGBoost on large, messy tabular data.
- Needs tuning of `n_estimators` and `learning_rate` together.

## 10. Bagging vs boosting

| | Bagging / Random Forest | Boosting / AdaBoost |
|---|---|---|
| How models are built | **in parallel**, independently | **sequentially**, each depends on the previous |
| Data for each model | bootstrap sample (uniform random) | weighted data: focus on previous mistakes |
| Typical base model | deep trees (low bias, high variance) | stumps / shallow trees (high bias, low variance) |
| Mainly reduces | **variance** | **bias** |
| Combining | simple vote / average | **weighted** vote (α per model) |
| Noise / outliers | robust | sensitive |
| Parallel training | yes (`n_jobs`) | no |

See [`../03-bagging/`](../03-bagging/) for the bagging side.

## 11. sklearn cheat sheet

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner (default = stump); NOT base_estimator=
    n_estimators=50,                                # number of rounds (default 50)
    learning_rate=1.0,                              # shrinks each model's weight (default 1.0)
    random_state=42,
)                                                   # no `algorithm=` argument: SAMME is the only one (sklearn >= 1.6)
ada.fit(X_train, y_train)

ada.predict(X_test)                # class labels
ada.predict_proba(X_test)          # class probabilities
ada.decision_function(X_test)      # weighted vote score (sign = class for binary)
ada.staged_score(X_test, y_test)   # accuracy after 1, 2, ..., M rounds (cheap learning curve)

ada.estimators_                    # the fitted weak learners
ada.estimator_weights_             # their weights (alpha; = 2 x the 1/2-ln formula for 2 classes)
ada.estimator_errors_              # their weighted errors
ada.feature_importances_

# tuning (note the estimator__ prefix for the base tree)
from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': [50, 100, 500],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator__max_depth': [1, 2, 3]}
GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42), grid, cv=5)

AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=42)
```

## 12. Common mistakes

| Mistake | Fix |
|---|---|
| Typing errors/alphas by hand (`model_weight(0.3)`) | Compute ε from the actual wrong rows' weights every round. |
| Error = count of mistakes / n | Error = **sum of the weights** of the wrong rows. |
| Final vote with labels 0/1 | Use **±1** votes before `sign()`. |
| Measuring error / updating weights on the resampled dataset | Train on the resample, but judge and update the **original** rows' weights. |
| Training the next stump on the old dataset (copy-paste bug) | Each round trains on the **newest** weighted data. |
| Resampling without a seed | Fix the seed (`np.random.default_rng(seed)`) for reproducible results. |
| `base_estimator=` / `algorithm='SAMME.R'` | Modern sklearn: `estimator=`; `algorithm` was removed (SAMME only). |
| Using a deep, fully grown tree as the base learner | Keep the learner weak (stump or shallow tree). |
| Tuning `n_estimators` and `learning_rate` separately | Tune them together; a lower rate needs more estimators. |
| Using AdaBoost on very noisy labels without checking | Check CV score; consider bagging/RF or gradient boosting with regularisation. |

## 13. Quick revision questions

1. Why is AdaBoost called "adaptive"? → *It adapts the row weights after every round, so later models focus on the
   rows earlier models got wrong.*
2. Write the formula for α. What happens at ε = 0.5? → *α = ½ ln((1−ε)/ε); at ε = 0.5, α = 0 (no say).*
3. How are row weights updated? → *Wrong rows × e^α, correct rows × e^(−α), then normalise.*
4. After normalising, how much total weight do the wrong rows have? → *Exactly 0.5.*
5. Does boosting reduce bias or variance mainly? Bagging? → *Boosting: bias. Bagging: variance.*
6. Why are decision stumps used? → *They are weak (high bias, low variance), fast, and work with sample weights.*
7. What does `learning_rate` do and how does it interact with `n_estimators`? → *It shrinks each α; a smaller rate
   needs more estimators.*
8. Why is AdaBoost sensitive to outliers? → *A mislabelled row keeps being wrong, so its weight grows exponentially.*
9. sklearn's `estimator_weights_` are 2× the α from the lecture. Is that a problem? → *No; scaling all α by the same
   positive number keeps every sign, so predictions are identical.*
10. How does AdaBoostRegressor combine its models? → *Weighted median of the predictions.*

## 14. Summary

- Boosting = **sequential** ensemble; each model fixes the previous models' mistakes.
- AdaBoost: equal weights → train a stump → ε = sum of wrong weights → α = ½ ln((1−ε)/ε) → wrong × e^α,
  correct × e^(−α) → normalise → repeat.
- Prediction = **sign(Σ αₘ hₘ(x))** with ±1 votes.
- Weak learners (stumps: high bias, low variance) + boosting → **lower bias**.
- Main knobs: `n_estimators` × `learning_rate` (tune together) and the base-estimator depth.
- Great on clean data; sensitive to noise and outliers; cannot be parallelised.
- Next: [`../06-gradient-boosting/`](../06-gradient-boosting/) generalises the idea to any differentiable loss,
  and [`../07-xgboost/`](../07-xgboost/) makes it fast and regularised.
