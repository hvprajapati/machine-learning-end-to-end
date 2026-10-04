# Support Vector Machine (SVM): Theory

> Read this first, then open `01-svm-classification.ipynb`.

---

## 1. The idea

Imagine two groups of points on paper. Many different straight lines can separate them. **Which line is best?**

SVM's answer: **the line that leaves the widest possible gap (margin) between the two groups.**

```
   ●  ●                     ○  ○
 ●   ●   ●   ┊      │      ┊   ○   ○
   ●   ●     ┊      │      ┊  ○  ○
 ●    ●      ┊      │      ┊     ○  ○
             ┊◄── margin ──►┊
          margin   boundary   margin
           edge               edge
```

A wide margin means the model is not "squeezed" against any training point, so it is more likely to classify
**new** points correctly.

---

## 2. Key words

| Term | Meaning |
|---|---|
| **Hyperplane / decision boundary** | The line that separates the classes (a plane in 3-D, a "hyperplane" in more dimensions) |
| **Margin** | The width of the empty "street" around the boundary |
| **Support vectors** | The training points closest to the boundary, on the margin edges or inside the street. **They alone decide where the boundary goes.** |
| **C** | How strongly SVM punishes points that are inside the margin or on the wrong side |
| **Kernel** | A trick that lets SVM draw **curved** boundaries |

**Why "support vectors"?** They *support* (hold up) the boundary like pillars. Move or delete any other point and
the boundary stays exactly where it is. Move a support vector and the boundary moves.

---

## 3. The maths, kept simple

The boundary is a straight line written as

$$
w_1x_1 + w_2x_2 + b = 0
$$

- Points with $w_1x_1 + w_2x_2 + b > 0$ → class 1
- Points with $w_1x_1 + w_2x_2 + b < 0$ → class 0

The two margin edges are where this score equals **+1** and **−1**. The width of the street turns out to be

$$
\text{margin width} = \frac{2}{\lVert w \rVert}
$$

So **a wider margin means smaller weights $w$**. SVM training is: *make $\lVert w\rVert$ as small as possible
(wide street) while keeping the points on the correct side.* You don't need more maths than this to use SVM well.

---

## 4. Hard margin vs soft margin, and the `C` parameter

**Hard margin:** no point is allowed inside the street. This only works if the classes are perfectly separable, and
a single outlier can ruin it.

**Soft margin (what scikit-learn uses):** some points *may* sit inside the street or even on the wrong side, but
each one costs a **penalty**. `C` controls how big that penalty is:

| `C` | Attitude | Margin | Risk |
|---|---|---|---|
| **small** (e.g. 0.01) | "Some mistakes are fine; keep the street wide" | wide | underfitting |
| **large** (e.g. 100) | "Avoid mistakes at all costs" | narrow | overfitting |

From the notebook (Social Network Ads):

| C | Support vectors | Test accuracy |
|:-:|:-:|:-:|
| 0.01 | 206 | 0.87 |
| 1 (default) | 128 | 0.90 |
| 100 | 125 | 0.89 |

A wider street (small C) contains more points, so more of them become support vectors.
**Choose C with cross-validation**, trying values like 0.01, 0.1, 1, 10, 100.

---

## 5. Feature scaling is required

The margin is a **distance**. Without scaling, the feature with the biggest numbers (salary: 15,000–150,000)
dominates the feature with small numbers (age: 18–60). Always use `StandardScaler` before SVM, fitted on the
training data only.

---

## 6. Curved boundaries: the kernel trick (preview)

A straight line cannot separate every dataset. Example: one class in a circle in the middle, the other class around it.

**Idea:** add a new feature that makes the data separable. For the circle example, add $x_1^2 + x_2^2$ (the distance
from the centre). In that new 3-D space, a flat plane separates the classes; back in 2-D it looks like a **circle**.

The **kernel trick** lets SVM do this *without actually computing* the new features, which keeps it fast.

| Kernel | Boundary shape | When to use |
|---|---|---|
| `'linear'` | straight line | many features (e.g. text), or data that is roughly linearly separable |
| `'rbf'` (default) | smooth curves, islands | the usual first choice for non-linear data |
| `'poly'` | polynomial curves | when you expect polynomial relationships |

In the notebook, switching from `'linear'` to `'rbf'` raises test accuracy from **0.90 to 0.93**. (Kernel SVM is
covered in detail in its own lesson.)

---

## 7. Using SVM in scikit-learn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

model = SVC(kernel='linear', C=1.0)
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

model.support_vectors_      # the support vectors
model.n_support_            # how many per class
model.coef_, model.intercept_   # w and b (linear kernel only)
```

| Parameter | Default | Meaning |
|---|---|---|
| `kernel` | `'rbf'` | `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'` |
| `C` | 1.0 | penalty for margin violations (small = wide margin) |
| `gamma` | `'scale'` | for rbf / poly: how far one point's influence reaches |
| `probability` | False | set True to enable `predict_proba` (slower training) |

For a regression version, use `SVR`; for very large linear problems, `LinearSVC` is faster.

---

## 8. SVM vs Logistic Regression vs K-NN

| | **SVM (linear)** | **Logistic Regression** | **K-NN** |
|---|---|---|---|
| Boundary | straight line, **maximum margin** | straight line, best probabilities | any shape |
| Which points matter | only the **support vectors** | all points | the K neighbours of each query |
| Outputs probabilities | not by default | yes | yes (vote share) |
| Needs scaling | yes | recommended | yes |
| Curved boundaries | yes, with kernels | needs polynomial features | naturally |
| Test accuracy here | 0.90 (rbf: 0.93) | 0.89 | 0.93 |

---

## 9. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Maximum margin gives good generalisation | Slow to train on very large datasets (> ~100,000 rows) |
| Works well with many features (e.g. text) | Needs feature scaling |
| Only support vectors matter, so robust to far-away points | Choosing C (and gamma, kernel) needs tuning |
| Kernels give flexible, curved boundaries | No probabilities by default; harder to interpret than logistic regression |

---

## 10. Common mistakes

1. **Forgetting to scale** the features.
2. **Using the default `kernel='rbf'`** when you wanted a linear model (always set `kernel` explicitly).
3. **Not tuning `C`**: the default is not always good.
4. **Expecting `predict_proba` to work** without `probability=True`.

---

## 11. Quick revision questions

1. *What does SVM maximise?* The margin: the distance between the boundary and the nearest points of each class.
2. *What are support vectors?* The training points closest to the boundary; only they determine it.
3. *What does a large C do?* Punishes mistakes heavily, giving a narrower margin (risk of overfitting).
4. *What does a small C do?* Allows more mistakes, giving a wider margin (risk of underfitting).
5. *Why scale features?* The margin is a distance, so large-valued features would dominate.
6. *How can SVM draw a curved boundary?* With a kernel, e.g. `kernel='rbf'`.

---

## 12. Support Vector Regression (SVR) (`02-svr-regression.ipynb`)

SVR uses the same ideas to predict **numbers**.

### The ε-tube
Instead of a street that keeps the classes **apart**, SVR fits a **tube** of half-width **ε (epsilon)** around the
prediction curve and tries to keep the data points **inside** it:

```
 salary
   │                 ●  ← outside the tube: penalised (a support vector)
   │           ╭──────────
   │      ╭────●───╯         ← tube of width ±ε around the curve
   │  ────●──╯               ● inside the tube: costs nothing
   │ ──╯
   └──────────────────────── level
```

- Points **inside** the tube cost **nothing**: "close enough is good enough".
- Points **on the edge or outside** are the **support vectors**; only they shape the curve.
- Each outside point costs a penalty proportional to its distance from the tube, multiplied by **C**.

| Parameter | Meaning | Larger value → |
|---|---|---|
| `epsilon` | half-width of the tube (default 0.1) | more points are "free", smoother / flatter curve |
| `C` | penalty for points outside the tube (default 1) | the curve bends more to reach outliers |
| `kernel` | `'rbf'` (curve) or `'linear'` (straight line) | |

### Scale X **and** y
ε is a fixed number (0.1). With salaries in the hundreds of thousands, it is meaningless unless the target is
standardised too. Use two scalers and convert predictions back:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sc_X, sc_y = StandardScaler(), StandardScaler()
X_sc = sc_X.fit_transform(X)
y_sc = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_sc, y_sc)
pred = sc_y.inverse_transform(svr.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
```

### Results on Position Salaries

| Setting | Level 6.5 | Level 10 (actual 1,000,000) |
|---|:-:|:-:|
| scaled, `C=1` (default) | ≈ 170,000 | ≈ 558,000 (CEO underpredicted) |
| scaled, `C=10` | ≈ 164,000 | ≈ 972,000 |
| **not scaled** | ≈ 130,000 | ≈ 130,000 (the same for every level: useless) |

---

## Summary

- SVM separates classes with the line that has the **widest margin**.
- **Support vectors**, the closest points, define that line.
- **C** balances a wide margin against training mistakes; tune it with cross-validation.
- Scale features first; use **kernels** for curved boundaries.
- **SVR** applies the same idea to regression with an ε-tube; scale both X and y.
