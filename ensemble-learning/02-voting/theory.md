# Voting Ensemble: Theory

> Companion to `01-voting-classifier.ipynb` and `02-voting-regressor.ipynb` (CampusX videos 2–4).
> Background (wisdom of the crowd, why ensembles work): [`../01-introduction/theory.md`](../01-introduction/theory.md).

---

## 1. Voting ensemble: classification (`01-voting-classifier.ipynb`)

Train several **different** classifiers on the **same** data; combine by voting.

### Hard vs soft voting

| | Rule | Needs |
|---|---|---|
| **Hard voting** (default) | majority of the predicted **labels** | only `predict` |
| **Soft voting** | **average the predicted probabilities**; pick the class with the highest average | `predict_proba` from every model |

**Example:** P(class 1) from three models = 0.45, 0.45, 0.99.
- Hard: labels 0, 0, 1 → **class 0**
- Soft: average = 0.63 → **class 1**

Soft voting uses **how confident** each model is, so it is usually better, but not always: **try both**.

### Weights
`weights=[w1, w2, w3]` gives stronger models a bigger say. Choose weights with cross-validation.

### Same algorithm, different hyperparameters
Diversity can also come from one algorithm with different settings. In the notebook, 5 SVMs with polynomial degrees
1–5 voting together reach **0.926**, better than every single SVM (best: 0.894).

### Results to remember from the notebook
| Dataset | Best single model | Voting |
|---|---|---|
| make_moons (curved) | 0.907 | hard 0.907, **soft 0.927** |
| Iris, 2 overlapping species, 2 features | **LR 0.75** | hard 0.68, soft 0.65, weighted (3,1,1) 0.71 |
| 5 SVMs, polynomial degrees 1–5 | 0.894 | **0.926** |

The Iris result is the important warning: **voting does not always beat the best model**. When the other members
are much weaker, an equal vote pulls the ensemble down.

```python
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('svc', SVC(probability=True)), ('rf', RandomForestClassifier())],
    voting='soft',            # or 'hard'
    weights=None,             # e.g. [2, 1, 1]
)
```

---

## 2. Voting ensemble: regression (`02-voting-regressor.ipynb`)

The prediction is the **average** (or weighted average) of the base regressors' predictions.

| Experiment (make_friedman1, 10-fold CV R²) | Members | Voting |
|---|---|---|
| LR + KNN + DT(depth 6) | 0.709, 0.654, 0.617 | **0.760** (beats all) |
| Trees of depth 3, 5, 7, 9 | best 0.621 | **0.675** |
| LR + DT + SVR | SVR alone **0.813** | 0.803 (slightly below the best) |

Voting works best when the members are **diverse and of similar quality**; one much stronger member is better
used alone (or with a higher weight).

```python
from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators=[('lr', LinearRegression()), ('knn', KNeighborsRegressor()),
                                 ('dt', DecisionTreeRegressor(max_depth=6))], weights=None, n_jobs=-1)
```

---

## 3. Quick revision questions

1. *Hard vs soft voting?* Hard: majority of the predicted labels. Soft: highest average predicted probability.
2. *Probabilities 0.45, 0.45, 0.99 for class 1: hard and soft predictions?* Hard 0, soft 1 (average 0.63).
3. *Does voting always beat the best single model?* No: on Iris (2 overlapping species) LR alone (0.75) beat every voting ensemble.
4. *What do `weights` do?* Give stronger models a bigger say; choose them with cross-validation.
5. *How does a VotingRegressor predict?* The (weighted) average of its members' predictions.

---

## Summary

- **Voting:** different algorithms (or settings) trained on the same data; combine by vote (classification) or average (regression).
- **Soft voting** usually beats hard voting, but try both.
- Works best with **diverse members of similar quality**; always compare against the best single model.
