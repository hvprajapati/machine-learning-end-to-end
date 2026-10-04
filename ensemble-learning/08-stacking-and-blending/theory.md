# Stacking and Blending — Theory

> Notebooks: [`01-stacking.ipynb`](01-stacking.ipynb) (sklearn `StackingClassifier` / `StackingRegressor`) ·
> [`02-blending-and-stacking-by-hand.ipynb`](02-blending-and-stacking-by-hand.ipynb) (build both methods yourself, see the leakage problem)

---

## §1. The idea in one picture

In **voting** (see [`../02-voting/`](../02-voting/)) every model gets a vote and we take the majority (or the average
probability). The rule for combining is **fixed by us**.

In **stacking** (Stacked Generalization, Wolpert 1992) the rule for combining is **learned from data**:

```
                 Level 0: base models                      Level 1: meta-model
               ┌───────────────────────┐
         ┌───► │  Random Forest        │ ──► p_rf   ─┐
         │     └───────────────────────┘             │
  X ─────┼───► │  KNN                  │ ──► p_knn  ─┼──►  Logistic Regression  ──►  final prediction
         │     └───────────────────────┘             │     ("who should I trust,
         └───► │  Gradient Boosting    │ ──► p_gb   ─┘      and how much?")
               └───────────────────────┘
```

- **Level-0 (base) models**: ordinary models trained on the original features `X`.
- **Level-1 (meta) model**: a model whose *input features are the predictions* of the base models.
  It learns how to combine them.

### Analogy: democracy vs. a smart manager

| | Voting | Stacking |
|---|---|---|
| Analogy | **Democracy**: every expert has one equal vote | **A manager** who has watched the experts for a while and learned *who is usually right* |
| Weights | Equal (or hand-set by you) | **Learned** by the meta-model |
| Can it learn "trust KNN when RF is unsure"? | No | Yes (if the meta-model is flexible enough) |

So voting is the special case "meta-model = plain average". Stacking replaces the average with a trained model.

---

## §2. The trap: naive stacking overfits

The obvious way to do stacking is:

1. Train every base model on the training set.
2. Ask each base model to predict the **same training set**.
3. Train the meta-model on those predictions.

**This is wrong.** The base models have already *seen* those rows. A random forest, for example, is almost 100 %
sure (and correct) on its own training rows. The meta-model therefore learns "RF is always right — trust RF
completely". On new data RF is *not* always right, so the meta-model's weights are wrong.

> It is like a teacher grading students on the exact questions they practised with the answer key.
> Everybody looks perfect, so the teacher learns nothing about who really understands.

This is a form of **data leakage**: the meta-model is trained on predictions that were made on data the base models
already knew. The fix: **the meta-model must only see predictions made on data the base model did NOT train on.**
There are two standard ways to get such predictions → **Blending** (§3) and **K-fold stacking** (§4).

Notebook 02 shows this on `heart.csv`: the naive meta-model gives the two tree models (RF and Gradient Boosting) huge
weights, almost ignores KNN, and reaches ≈ 100 % training accuracy — but over 20 random splits it has the *worst* test
accuracy of the three stacking recipes (0.820 vs 0.828 blending vs 0.841 K-fold stacking).

---

## §3. Fix 1 — Blending (hold-out set)

Split the training data once more, into a **train part** and a **hold-out (blend) part**.

```
 full data ─┬─ training data ─┬─ D_train  (e.g. 75 %) ──► fit base models
            │                 └─ D_hold   (e.g. 25 %) ──► base models predict  ──► Z_hold ──► fit meta-model
            └─ test data ────────────────────────────────► base models predict ──► Z_test ──► meta-model predicts
```

**Algorithm**

1. Split training data into `D_train` and `D_hold`.
2. Fit every base model on `D_train`.
3. Each base model predicts `D_hold` → these predictions form a new table `Z_hold` (one column per base model).
4. Fit the meta-model on (`Z_hold`, `y_hold`).
5. For a new point: base models predict → meta-model combines.

| ✅ Pros | ❌ Cons |
|---|---|
| Very simple, fast (each base model trained once) | Base models see less data (only `D_train`) |
| No leakage, easy to reason about | Meta-model is trained on a small hold-out set → noisy on small datasets |

The name comes from the Netflix Prize (2009), where teams "blended" hundreds of models this way.

---

## §4. Fix 2 — K-fold stacking (out-of-fold predictions)

Instead of one hold-out set, use **cross-validation** so that *every* training row gets a prediction from a model
that did not see it. These are called **out-of-fold (OOF) predictions**.

```
 Training data split into K = 5 folds:   [F1][F2][F3][F4][F5]

 round 1: fit on F2..F5  → predict F1  ┐
 round 2: fit on F1,F3.. → predict F2  │
 round 3: ...            → predict F3  ├──► OOF column for this base model (one honest prediction per row)
 round 4: ...            → predict F4  │
 round 5: fit on F1..F4  → predict F5  ┘

 Repeat for each base model → Z_oof (n_rows × n_base_models) → fit meta-model on (Z_oof, y)
 Finally refit each base model on ALL training data → used at prediction time.
```

**Algorithm**

1. For each base model: get OOF predictions on the training set with K-fold CV
   (in sklearn: `cross_val_predict(model, X, y, cv=K, method='predict_proba')`).
2. Stack these columns into `Z_oof`; fit the meta-model on (`Z_oof`, `y`).
3. Refit every base model on the **full** training set.
4. Predict: base models (full-data versions) → meta-model.

This is **exactly what `sklearn.ensemble.StackingClassifier(cv=K)` does.** Notebook 02 builds it by hand and shows the
hand-made version gives the same predictions as sklearn.

| ✅ Pros | ❌ Cons |
|---|---|
| Every training row is used for both levels | K extra fits per base model → slower |
| Meta-model has more rows → more stable | A bit harder to code by hand (sklearn does it for you) |

---

## §5. Small numeric illustration

Binary problem, 4 training rows, 2 base models (RF and KNN). Each cell = predicted P(class 1).

**Naive (in-sample) predictions** — the base models have seen these rows:

| row | y | RF (in-sample) | KNN (in-sample) |
|---|---|---|---|
| 1 | 1 | 0.98 | 0.70 |
| 2 | 0 | 0.03 | 0.40 |
| 3 | 1 | 0.95 | 0.45 |
| 4 | 0 | 0.05 | 0.60 |

The meta-model sees: RF separates the classes perfectly, KNN makes 2 mistakes (rows 3 and 4 on the wrong side of 0.5)
→ it learns "trust RF, ignore KNN".

**Out-of-fold predictions** — each row predicted by a model that did not train on it:

| row | y | RF (OOF) | KNN (OOF) |
|---|---|---|---|
| 1 | 1 | 0.70 | 0.75 |
| 2 | 0 | 0.55 | 0.30 |
| 3 | 1 | 0.60 | 0.65 |
| 4 | 0 | 0.35 | 0.40 |

Now RF makes 1 mistake (row 2) and KNN makes 0 mistakes. The honest picture is: *both are useful, KNN is at least as
good*. A meta-model trained on this table gives KNN a real weight. The in-sample table told a false story.

Once trained, a logistic-regression meta-model combines like this:

```
P(y=1) = sigmoid( b + w_rf · p_rf + w_knn · p_knn )
e.g.   = sigmoid( -3 + 2.5·0.60 + 3.5·0.65 ) = sigmoid(0.775) ≈ 0.68  → class 1
```

The weights `w` tell you **how much the meta-model trusts each base model** (notebook 01, §5).

---

## §6. Multi-layer stacking (brief)

You can stack more than once: level-0 models → several level-1 models → one level-2 model. In sklearn you do this by
using a `StackingClassifier` as the `final_estimator` of another `StackingClassifier`.
It is used in Kaggle competitions, but every extra layer adds training time and overfitting risk, and the gains are
usually tiny. For learning and for most real projects, **one meta-layer is enough**.

---

## §7. Choosing the models

**Base models — be diverse.** Stacking helps only when base models make *different* mistakes.
Five slightly different random forests give the meta-model nothing new to combine. Good mixes:

- a linear model (logistic / linear regression),
- a distance-based model (KNN, SVM),
- a tree ensemble (Random Forest, Gradient Boosting / XGBoost),
- maybe Naive Bayes or a neural net.

**Meta-model — keep it simple.** Its input is just a handful of columns (one per base model) and those columns are
highly correlated (all models try to predict the same `y`). A complex meta-model overfits easily.
**Logistic regression** (classification) and **Ridge / linear regression** (regression) are the usual choices —
sklearn's defaults are exactly these. Bonus: their coefficients are easy to read.

**Scale features** for scale-sensitive base models (KNN, SVM, logistic regression) with a `Pipeline`, e.g.
`make_pipeline(StandardScaler(), KNeighborsClassifier())`.

---

## §8. sklearn cheat sheet

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV

clf = StackingClassifier(
    estimators=[('rf', rf), ('knn', knn_pipe), ('gb', gb)],   # level-0 models (name, model)
    final_estimator=LogisticRegression(),   # level-1 model (default: LogisticRegression)
    cv=5,                    # K for the out-of-fold predictions (default 5; int, CV splitter, or 'prefit')
    stack_method='auto',     # which output of base models to use (see below)
    passthrough=False,       # True → meta-model also gets the original features X
    n_jobs=None,
)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.transform(X_test)              # the meta-features the meta-model sees
clf.final_estimator_.coef_         # how much the meta-model trusts each base model
clf.named_estimators_['rf']        # base models refitted on the full training set

reg = StackingRegressor(estimators=[...], final_estimator=RidgeCV(), cv=5)
```

| Parameter | Meaning |
|---|---|
| `cv` | How OOF predictions are made. Integer K → (stratified) K-fold. `'prefit'` → base models are already fitted and are **not** refitted; you must then make sure the meta-model is fitted on data the base models never saw (this is how you do blending with sklearn). |
| `stack_method` | `'auto'` tries `predict_proba`, then `decision_function`, then `predict`. Can force one. For a **binary** problem sklearn keeps only one probability column per base model (the second is redundant: `p0 = 1 − p1`), so 3 base models → 3 meta-features. For K > 2 classes → K columns per model. |
| `passthrough` | `False`: meta-model sees only base predictions. `True`: meta-model sees predictions **+ original features**. Can help if the meta-model needs context ("trust KNN for young patients"), but makes the meta-model's job bigger → more overfitting risk; scale the features if the meta-model is linear. |
| `final_estimator` | The meta-model. Default `LogisticRegression()` / `RidgeCV()`. |

`StackingRegressor` works the same way; base models give numeric predictions (`predict`) and the meta-model is a
regressor.

---

## §9. Voting vs Stacking vs Blending

| | Voting | Blending | K-fold Stacking |
|---|---|---|---|
| How models are combined | Fixed rule: majority / average | Learned by meta-model | Learned by meta-model |
| Meta-model trained on | — | Predictions on a single hold-out set | Out-of-fold predictions on the whole training set |
| Data used by base models | All training data | Only the train part (loses the hold-out) | All training data (final refit) |
| Training cost | 1 fit per model | 1 fit per model | K + 1 fits per model |
| Leakage risk | None | Low (if hold-out is truly separate) | Low (if done with proper CV) |
| Works well on small data? | Yes | Weak (tiny hold-out) | Better |
| sklearn | `VotingClassifier` / `VotingRegressor` | by hand, or `cv='prefit'` | `StackingClassifier` / `StackingRegressor` |

---

## §10. Strengths and limitations

**Strengths**
- Learns the best way to combine models instead of guessing weights.
- Can use very different kinds of models together.
- Often gives a small extra gain on large datasets and in competitions.

**Limitations**
- Slower to train and to predict (many models, plus K-fold refits).
- Harder to explain than a single model.
- On **small datasets** (like `heart.csv`, 303 rows) the gain is often zero or within noise: the meta-model has few
  rows to learn from and the base models are similar in skill. Notebook 01 shows this honestly.
- Easy to implement wrongly (leakage, §11).

---

## §11. Common mistakes

1. **Leakage: training the meta-model on in-sample predictions** (naive stacking, §2). Always use OOF predictions or a
   hold-out set.
2. **Using the test set as the blending hold-out.** Then the test score is no longer an honest estimate. The hold-out
   must come from the *training* data.
3. **Fitting the scaler on all data** before splitting. Put scaling inside a `Pipeline` so it is fitted only on
   training folds.
4. **Using only similar base models** (e.g. three tree ensembles) — little diversity, little gain.
5. **A too-powerful meta-model** (e.g. a deep tree or a big GBM) on a few meta-features → overfits.
6. **Judging on one small test split.** With 61 test rows, one patient = 1.6 % accuracy. Use cross-validation to
   compare models.
7. **With `cv='prefit'`, fitting the stack on the same data the base models were trained on** — this is naive stacking
   again.

---

## §12. Quick revision questions

<details><summary>1. What are level-0 and level-1 models?</summary>
Level-0 (base) models are trained on the original features. The level-1 (meta) model is trained on the base models'
predictions and learns how to combine them.</details>

<details><summary>2. How is stacking different from soft voting?</summary>
Soft voting averages probabilities with fixed (equal or hand-set) weights. Stacking learns the combination with a
meta-model, so the weights come from data.</details>

<details><summary>3. Why does naive stacking overfit?</summary>
The meta-model is trained on predictions the base models made on their own training rows. Those predictions are
over-confident and too accurate, so the meta-model learns to trust the most overfit model.</details>

<details><summary>4. What is the difference between blending and K-fold stacking?</summary>
Blending trains the meta-model on predictions for one hold-out set. K-fold stacking trains it on out-of-fold
predictions for the whole training set, then refits the base models on all training data.</details>

<details><summary>5. What does `cv=5` do in StackingClassifier?</summary>
It uses 5-fold cross-validation to create out-of-fold predictions for training the meta-model. The base models
used at prediction time are refitted on the full training set.</details>

<details><summary>6. Why is logistic regression a good meta-model?</summary>
It is simple (few meta-features → low overfitting risk), fast, and its coefficients show how much each base model
is trusted.</details>

<details><summary>7. What does `passthrough=True` do?</summary>
The meta-model receives the original features in addition to the base models' predictions.</details>

---

## §13. Summary

- **Stacking** = base models + a **meta-model that learns how to combine them** (voting = fixed combination).
- **Never** train the meta-model on in-sample predictions → leakage and overfitting.
- **Blending**: meta-model trained on predictions for a hold-out set. Simple, but wastes data.
- **K-fold stacking**: meta-model trained on out-of-fold predictions. What `StackingClassifier(cv=K)` does.
- Use **diverse** base models and a **simple** meta-model (logistic / ridge regression).
- Stacking is not magic: on small datasets it often only matches the best single model. Measure with
  cross-validation and report honestly.
