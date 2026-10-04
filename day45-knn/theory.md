# K-Nearest Neighbors (K-NN): Theory

> Read this first, then open the notebooks. Section numbers (§) are referenced from them.

---

## 1. The idea

> **"Tell me who your neighbours are, and I'll tell you who you are."**

To predict the label of a new point, K-NN finds the **K training points closest to it** and:
- **Classification:** takes a **majority vote** of their labels.
- **Regression:** takes the **average** of their values.

There is no equation to fit and no weights to learn. The training data itself is the model.

| Property | K-NN |
|---|---|
| Learning type | Supervised (needs labels) |
| Tasks | Classification and regression |
| Model type | **Non-parametric**: no fixed set of parameters; complexity grows with the data |
| Learner type | **Lazy** (instance-based): no work at training time, all work at prediction time |
| Decision boundary | **Non-linear**, any shape |

---

## 2. The algorithm

```
Training:   store X_train and y_train.                      (that's all)

Predicting a new point x:
  1. Compute the distance from x to EVERY training point.
  2. Sort the distances and take the K nearest points.
  3. Classification -> return the most common label among them.
     Regression     -> return the mean of their target values.
```

The probability estimate for a class is simply its share of the K votes:
$P(\text{class } c \mid x) = \frac{\#\{\text{neighbours with label } c\}}{K}$. This is what `predict_proba` returns.

---

## 3. Measuring "near": distance metrics

For two points $a$ and $b$ with $d$ features:

| Metric | Formula | Notes |
|---|---|---|
| **Euclidean** (default) | $\sqrt{\sum_{j}(a_j-b_j)^2}$ | straight-line distance |
| **Manhattan** | $\sum_{j}\lvert a_j-b_j\rvert$ | "city-block" distance; less sensitive to one large difference |
| **Minkowski** | $\left(\sum_{j}\lvert a_j-b_j\rvert^{p}\right)^{1/p}$ | general form: $p=1$ Manhattan, $p=2$ Euclidean |
| **Cosine** | $1 - \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}$ | compares direction, not magnitude; common for text vectors |
| **Hamming** | share of features that differ | for categorical / binary features |

In scikit-learn, `KNeighborsClassifier(metric='minkowski', p=2)` (the default) is Euclidean distance.

---

## 4. Worked example by hand

Training data (two classes, R = red, B = blue) and a query point **Q = (3, 2)**:

| Point | Coordinates | Class | Distance to Q |
|:-:|:-:|:-:|---|
| F | (4, 2) | B | $\sqrt{1^2+0^2} = 1.000$ |
| B | (2, 1) | R | $\sqrt{1^2+1^2} = 1.414$ |
| C | (2, 3) | R | $\sqrt{1^2+1^2} = 1.414$ |
| A | (1, 2) | R | $\sqrt{2^2+0^2} = 2.000$ |
| D | (5, 4) | B | $\sqrt{2^2+2^2} = 2.828$ |
| E | (6, 5) | B | $\sqrt{3^2+3^2} = 4.243$ |

(sorted by distance)

| K | Nearest points | Votes | Prediction |
|:-:|---|---|:-:|
| 1 | F | B: 1 | **Blue** |
| 3 | F, B, C | R: 2, B: 1 | **Red** |
| 5 | F, B, C, A, D | R: 3, B: 2 | **Red** |

**The choice of K changes the answer.** One blue point happens to be closest, but most of the neighbourhood is red.
`02-knn-from-scratch.ipynb` reproduces this table in code.

---

## 5. Choosing K

K controls the **bias–variance trade-off**:

| K | Boundary | Problem |
|---|---|---|
| **small** (K = 1) | jagged, with islands around single points | **overfitting / high variance**: follows noise and outliers |
| **large** (K → n) | very smooth; at K = n, one class everywhere | **underfitting / high bias**: predicts the majority class |
| **moderate** | smooth but flexible | best generalisation |

Notebook results (Social Network Ads, 300 train / 100 test):

| K | Train accuracy | Test accuracy |
|:-:|:-:|:-:|
| 1 | 1.00 | 0.87 |
| 5 | 0.91 | 0.93 |
| 51 | 0.89 | 0.92 |
| 101 | 0.78 | 0.84 |
| 299 (all points) | 0.63 | 0.68 (always "did not buy") |

Practical rules:
- **Choose K by cross-validation** (e.g. `GridSearchCV`), never by test-set accuracy. Here 10-fold CV picks **K = 11**.
- Use an **odd K** for binary classification to avoid tied votes.
- A common starting point is $K \approx \sqrt{n}$, but always confirm with cross-validation.
- Training accuracy at K = 1 is always 1.0 (each point is its own nearest neighbour), so never judge K-NN on training accuracy.

---

## 6. Feature scaling is mandatory

Distances add up feature differences, so a feature with large numbers dominates. In the notebook:

- Age ranges 18–60, salary 15,000–150,000.
- Two users 10 years and 1,000 salary apart: $\sqrt{10^2 + 1000^2} \approx 1000.05$. Age contributes essentially nothing.

| | Test accuracy |
|---|:-:|
| Without scaling | 0.83 (the boundary becomes horizontal salary bands) |
| With `StandardScaler` | **0.93** |

Rules:
- Use `StandardScaler` (Day 7) or `MinMaxScaler` (Day 8).
- **Fit the scaler on the training set only**, and transform new data with the same scaler.
- Put the scaler inside a `Pipeline` so cross-validation scales each fold correctly.

---

## 7. Weighted voting

By default every neighbour gets one equal vote (`weights='uniform'`). With `weights='distance'`, each vote is
weighted by $1/\text{distance}$, so closer neighbours count more:

- Reduces the influence of far-away neighbours when K is large.
- Breaks ties naturally.
- A training point at distance 0 from the query decides the prediction alone.

Treat it as a hyperparameter and let cross-validation choose (on this dataset `uniform` won).

---

## 8. K-NN for regression

Same neighbour search; the prediction is the **mean** (or distance-weighted mean) of the neighbours' target values:

$$
\hat y = \frac{1}{K}\sum_{i \in N_K(x)} y_i
$$

```python
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
```

The same neighbour idea powers **`KNNImputer`** (Day 22): missing values are filled with the average of the
nearest rows.

---

## 9. Computational cost

| Phase | Brute force | With KD-tree / Ball-tree |
|---|---|---|
| Training | $O(1)$: store the data | $O(n \log n)$: build the index |
| One prediction | $O(n \cdot d)$: distance to every point | roughly $O(\log n)$ in low dimensions |
| Memory | stores the **entire** training set | same, plus the index |

($n$ = training points, $d$ = features.) scikit-learn picks the method automatically with `algorithm='auto'`.
Tree indexes help a lot for low $d$ (roughly < 20) and lose their advantage as $d$ grows.

**Consequence:** K-NN trains instantly but predicts slowly on large datasets, the opposite of most models.

---

## 10. The curse of dimensionality

As the number of features grows:
- Points spread out and the space becomes mostly empty.
- The distance to the nearest and the farthest point become almost the same, so "nearest" loses its meaning.
- Irrelevant features add noise to every distance.

**Remedies:** feature selection, dimensionality reduction (PCA, Day 28), or switch to a model that learns feature
importance (trees, linear models).

---

## 11. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Simple and intuitive; easy to explain | Slow prediction and high memory on large datasets |
| No training time; new data can be added instantly | **Requires feature scaling** |
| Naturally **non-linear** boundaries | Suffers from the curse of dimensionality |
| Works for multi-class problems and regression | Sensitive to irrelevant features and noisy labels |
| Few hyperparameters (K, metric, weights) | Needs a meaningful distance; categorical data must be encoded |
| Makes no assumption about the data's distribution | Imbalanced classes: the majority class tends to dominate the votes |

**Good fit:** small-to-medium datasets, few (relevant) features, irregular class boundaries, recommendation
("similar items / users"), and as a quick baseline.

---

## 12. scikit-learn cheat sheet

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = {'kneighborsclassifier__n_neighbors': range(1, 31),
        'kneighborsclassifier__weights': ['uniform', 'distance']}
search = GridSearchCV(pipe, grid, cv=10).fit(X_train, y_train)

search.predict(X_new)                       # labels
search.predict_proba(X_new)                 # share of votes per class
search.best_estimator_[-1].kneighbors(...)  # which neighbours were used (pass scaled data)
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_neighbors` | 5 | K |
| `weights` | `'uniform'` | `'distance'` = closer neighbours vote more |
| `metric`, `p` | `'minkowski'`, 2 | distance function (p=1 Manhattan, p=2 Euclidean) |
| `algorithm` | `'auto'` | `'brute'`, `'kd_tree'`, `'ball_tree'` |
| `n_jobs` | None | parallelise the neighbour search (`-1` = all cores) |

---

## 13. K-NN vs related algorithms

| | **K-NN** | **K-Means** (Day 44) | **Logistic Regression** (Day 38) |
|---|---|---|---|
| Type | supervised | unsupervised | supervised |
| Meaning of K | number of neighbours that vote | number of clusters | – |
| Learns | nothing (stores data) | K centroids | weights $w$, $b$ |
| Boundary | non-linear, any shape | – | linear |
| Prediction cost | high (search training set) | low | very low |
| Interpretability | "these similar examples" | cluster centres | coefficients / odds ratios |

---

## 14. Common mistakes

1. **Not scaling features**, or fitting the scaler on the full dataset (leakage).
2. **Choosing K from test accuracy** instead of cross-validation.
3. **Judging the model by training accuracy**, which is always 1.0 at K = 1.
4. **Using even K** in binary problems, causing tied votes.
5. **Feeding raw categorical columns**: encode them first (Days 9–10), or use a suitable metric.
6. **Using K-NN on high-dimensional data** without feature selection or PCA.
7. **Ignoring class imbalance**: check precision/recall, not only accuracy (Day 39).

---

## 15. Interview questions

1. *Is K-NN parametric?* No, it is non-parametric and stores the data instead of learning a fixed set of parameters.
2. *Why is it called a lazy learner?* It does no work at training time; all computation happens at prediction.
3. *What happens with K = 1 and with K = n?* K = 1 overfits (training accuracy 1.0); K = n always predicts the majority class.
4. *Why must features be scaled?* Distances are dominated by features with larger numeric ranges.
5. *How do you choose K?* Cross-validation; prefer odd K for binary problems.
6. *Time complexity of prediction?* $O(n \cdot d)$ per query with brute force; KD/Ball trees speed up low-dimensional data.
7. *What is the curse of dimensionality?* In high dimensions all points become roughly equidistant, so nearest neighbours stop being meaningful.
8. *How does K-NN regression work?* It averages the target values of the K nearest neighbours.
9. *K-NN vs K-Means?* K-NN is supervised classification/regression by neighbour voting; K-Means is unsupervised clustering into K groups.

---

## Summary

- K-NN predicts from the **K closest training points**: majority vote (classification) or average (regression).
- It learns nothing at training time; prediction searches the stored data.
- **Scale features** (0.83 → 0.93 accuracy here) and **choose K by cross-validation** (K = 11 here).
- Small K overfits, large K underfits; odd K avoids ties.
- Great for small, low-dimensional data with irregular boundaries; weak on large or high-dimensional data.
