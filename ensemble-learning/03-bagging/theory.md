# Bagging: Theory

> Companion to `01-bagging-classifier.ipynb` and `02-bagging-regressor.ipynb` (CampusX videos 5–8).
> Background: [`../01-introduction/theory.md`](../01-introduction/theory.md).

---

## 1. Bagging: the idea (`01-bagging-classifier.ipynb`)

**Bagging = Bootstrap + Aggregation**

1. **Bootstrap:** from the training set of $n$ rows, draw $n$ rows **at random with replacement**, many times. Each
   sample has repeated rows and is missing others.
2. Train the **same algorithm** on each sample: model 1 on sample 1, model 2 on sample 2, …
3. **Aggregate:** majority vote (classification) or mean (regression).

Voting creates diversity with **different algorithms**; bagging creates it with **different data**.

### Facts about bootstrap samples
- Each bootstrap sample contains about **63%** of the distinct rows (the chance a row is never picked is
  $(1 - 1/n)^n \approx e^{-1} \approx 0.37$).
- The ~**37%** of rows a model never saw are its **out-of-bag (OOB)** rows.

### Why bagging works: it reduces variance
| Model type | Bias | Variance | Example |
|---|---|---|---|
| Fully grown decision tree | **low** (fits training data perfectly) | **high** (changes a lot with the data) | `DecisionTreeClassifier()` |
| Linear / logistic regression | high | low | `LinearRegression()` |

Bagging uses **low-bias, high-variance** models. Each model is trained on different data, so changes or noise in the
data affect only some of the models; averaging smooths those over-reactions out. **Bias stays low, variance drops.**

Notebook evidence:
- moons data: one deep tree 0.81 → 200 bagged trees 0.85;
- 10,000-row dataset: one tree 0.927 → bagging 0.945;
- regression (friedman1): one tree test R² 0.62 → bagging 0.85.

**Bagging stable models gains little.** Bagged SVMs scored 0.915 vs 0.926 for one SVM: an SVM has low variance
already, and each bagged copy only sees part of the data. **Use bagging with decision trees** (KNN can also benefit).

**Random Forest** = bagging of decision trees **plus** a random subset of features at every split ([`../04-random-forest/`](../04-random-forest/)).

---

## 2. The four flavours of bagging

| Flavour | Rows | Columns | `BaggingClassifier` settings |
|---|---|---|---|
| **Bagging** | sample **with** replacement | all | `bootstrap=True, max_samples<1` |
| **Pasting** | sample **without** replacement | all | `bootstrap=False, max_samples<1` |
| **Random Subspaces** | all rows | sample columns | `max_samples=1.0, bootstrap=False, max_features<1` |
| **Random Patches** | sample rows | sample columns | `max_samples<1, max_features<1` |

Notebook (10,000 rows × 10 features, 500 trees): bagging 0.945, pasting 0.946, subspaces 0.942, patches 0.938 vs a
single tree 0.927. A grid search then chose **random subspaces with 100 trees: 0.9525**.

**Tips**
- Start with plain bagging (or pasting) and `max_samples` around 0.25–0.5, then **tune**.
- Column sampling (subspaces, patches) is especially useful when there are **many features** (images, text).
- Use `n_jobs=-1` to train the models in parallel.

---

## 3. Bagging in scikit-learn (classification and regression)

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # the base model (older versions: base_estimator=)
    n_estimators=500,                    # number of models
    max_samples=0.25,                    # rows per model (fraction or count)
    bootstrap=True,                      # rows with replacement? (False = pasting)
    max_features=1.0,                    # columns per model
    bootstrap_features=False,            # columns with replacement?
    oob_score=True,                      # free validation score
    n_jobs=-1,
    random_state=42,
)
bag.fit(X_train, y_train)
bag.oob_score_               # out-of-bag accuracy (R² for BaggingRegressor)
bag.estimators_samples_      # which rows each model saw
bag.estimators_features_     # which columns each model saw
```

`BaggingRegressor` has the same parameters; its prediction is the **mean** of the models' predictions.
Its default base model is a decision tree.

**OOB score:** each row is predicted only by the models that did not see it, giving a validation score without a
separate validation set. Notebook: OOB 0.943 vs test 0.945 (classification); OOB R² 0.821 vs test 0.848 (regression).

**Tune with `GridSearchCV`:** `estimator`, `n_estimators`, `max_samples`, `bootstrap`, `max_features`,
`bootstrap_features`. Rules of thumb are only starting points.

> Note: the CampusX videos use `base_estimator=` and the Boston dataset. In current scikit-learn, use `estimator=`,
> and Boston has been removed (the notebooks use `make_friedman1` and other built-in datasets instead).

---

## 4. Bagging vs Boosting

Both are the most widely used ensemble techniques, and this comparison is a classic interview question.

| | **Bagging** | **Boosting** |
|---|---|---|
| **Base models** | **low bias, high variance** (e.g. fully grown decision trees) | **high bias, low variance** (e.g. shallow trees / decision stumps) |
| **Goal** | reduce **variance** | reduce **bias** |
| **Training** | **parallel**: models are independent, each on a random sample | **sequential**: each model learns from the previous model's mistakes |
| **Weight of each model** | **equal** (every model has one vote: a democracy) | **different**: better models get a bigger say |
| **Examples** | Bagging, Random Forest | AdaBoost, Gradient Boosting, XGBoost |

**Rule of thumb:**
- Your model does very well on training data but its results change a lot when the data changes → it has high
  variance → **bagging**.
- Your model is stable but not accurate enough even on training data → it has high bias → **boosting**.

---

## 5. Quick revision questions

1. *What is bootstrapping?* Sampling n rows with replacement.
2. *Which models benefit most from bagging, and why?* Low-bias, high-variance models such as deep trees, because averaging reduces variance.
3. *Bagging vs pasting? Random subspaces vs random patches?* Rows with vs without replacement; columns only vs rows and columns.
4. *What is the OOB score?* Accuracy on each row using only the models that did not see it during training.
5. *Bagging vs boosting in one line?* Bagging: parallel, equal votes, reduces variance. Boosting: sequential, weighted, reduces bias.

---

## Summary

- **Bagging = Bootstrap + Aggregation**: same algorithm, different random samples, combined by vote / average.
- It reduces the **variance** of low-bias models such as deep decision trees; four flavours; free **OOB** score.
- **Random Forest** ([`../04-random-forest/`](../04-random-forest/)) is bagging of trees plus random features at every split.
- **Boosting** ([`../05-adaboost/`](../05-adaboost/), [`../06-gradient-boosting/`](../06-gradient-boosting/), [`../07-xgboost/`](../07-xgboost/)) reduces bias instead.
