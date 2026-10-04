# Lasso Regression: Theory

> Builds on Day 35 (`day35-regularized-linear-models/theory.md`): overfitting, bias–variance, Ridge, scaling.

---

## 1. Definition

**Lasso** (*Least Absolute Shrinkage and Selection Operator*, Tibshirani 1996) replaces Ridge's squared penalty
with an **absolute-value** penalty:

$$
\boxed{\;J(w, b) = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat y_i)^2 \;+\; \alpha\sum_{j=1}^{p}\lvert w_j\rvert\;}
$$

- $\sum \lvert w_j\rvert = \lVert w\rVert_1$, the **L1 norm**. Lasso = **L1 regularisation**.
- The intercept is not penalised.
- scikit-learn divides the error term by $2n$ for Lasso but not for Ridge, so **α values are not comparable between `Lasso` and `Ridge`**.

---

## 2. The key property: exact zeros

Lasso shrinks coefficients **and sets some of them to exactly 0**, which is **automatic feature selection**.

| α | Effect on the diabetes data (notebook 01) |
|---|---|
| 0.01 | all 10 features kept |
| 0.1 | 7 kept |
| 0.5 | 4 kept |
| 1 | 3 kept (`bmi`, `bp`, `s5`) |
| 2 | 2 kept (`bmi`, `s5`) |

The order in which features drop out is a ranking of their importance to the model.

---

## 3. Why the L1 penalty produces zeros

### 3a. The size of the push
| Penalty | Gradient (push toward 0) | Near $w = 0$ |
|---|---|---|
| Ridge $\alpha w^2$ | $2\alpha w$, proportional to $w$ | the push **fades to nothing** |
| Lasso $\alpha\lvert w\rvert$ | $\alpha \cdot \text{sign}(w)$, **constant** | the push stays **full strength** |

Ridge's push weakens as the weight gets small, so the weight never quite reaches 0. Lasso pushes equally hard all the
way, and once the push exceeds what the data "pays" for that feature, the weight lands on 0 and stays there.

### 3b. The closed form for one standardised feature
For a single standardised feature, with $\hat w$ the least-squares coefficient and λ the penalty in matching units:

$$
w_{\text{ridge}} = \frac{\hat w}{1 + \lambda}
\qquad\qquad
w_{\text{lasso}} = \text{sign}(\hat w)\,\max\big(\lvert\hat w\rvert - \lambda,\; 0\big)
$$

Ridge **divides**: it never reaches 0. Lasso **subtracts and clips**: it hits 0 once $\lambda \ge |\hat w|$.
This is called **soft-thresholding**. The notebook's "slope vs α" plot shows exactly these two shapes:
a smooth asymptote for Ridge, and a straight line that stops at 0 for Lasso.

### 3c. The loss curve
$\lvert w\rvert$ has a **sharp kink at 0**. Adding it to the smooth error bowl creates a V-shaped point at $w = 0$;
for large enough α that kink becomes the global minimum.

### 3d. Geometry
Lasso is equivalent to minimising the error subject to $\sum\lvert w_j\rvert \le t$, a **diamond** (a cross-polytope in
higher dimensions). Its **corners lie on the axes**. The error ellipse usually touches the diamond first at a corner or
edge, where one or more weights are exactly 0. Ridge's circle has no corners.

---

## 4. Solving Lasso

$\lvert w\rvert$ is not differentiable at 0, so **there is no closed-form solution** and plain gradient descent does not
work directly. scikit-learn uses **coordinate descent**: optimise one coefficient at a time (each step is a
soft-thresholding update) and cycle until convergence.

If you see `ConvergenceWarning`, increase `max_iter`, scale the features, or increase α.

---

## 5. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Automatic feature selection gives **sparse**, interpretable models | **Correlated features:** keeps one somewhat arbitrarily, drops the rest |
| Works well when only a few features truly matter | When $p > n$, selects **at most n** features |
| Reduces overfitting like Ridge | Selection can be unstable: small data changes can swap the chosen feature |
| Cheaper predictions and data collection (fewer features) | Usually slightly worse than Ridge when *many* features each contribute a little |

The correlated-features weakness is what **Elastic Net** (Day 37) fixes.

---

## 6. Practical usage

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))
model.fit(X_train, y_train)

lasso = model[-1]
print(lasso.alpha_)                              # chosen alpha
selected = X_train.columns[lasso.coef_ != 0]     # features kept
```

| Class | Use |
|---|---|
| `Lasso(alpha=1.0)` | fixed α |
| `LassoCV(cv=5)` | builds its own α grid and picks the best by k-fold CV |
| `LassoLarsCV` | efficient when there are many more features than samples |
| `LogisticRegression(l1_ratio=1, solver='saga')` | L1 for classification |

**Scaling is essential**, even more than for Ridge: which features survive depends directly on coefficient sizes.

---

## 7. Ridge vs Lasso at a glance

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | $\alpha\sum w_j^2$ | $\alpha\sum\lvert w_j\rvert$ |
| Exact zeros / feature selection | no | **yes** |
| Constraint shape | circle | diamond |
| Closed-form solution | yes | no (coordinate descent) |
| Correlated features | shares the weight | picks one |
| Best when | many features each matter a little | few features matter |

---

## 8. Interview questions

1. *Difference between Ridge and Lasso?* L2 vs L1 penalty; Lasso produces exact zeros (feature selection), Ridge only shrinks.
2. *Why does L1 give sparse solutions?* Constant-size gradient that doesn't fade near 0; geometrically, the diamond's corners lie on the axes.
3. *Does Lasso have a closed-form solution?* No; it is solved by coordinate descent.
4. *What does Lasso do with two highly correlated features?* Tends to keep one and drop the other, somewhat arbitrarily.
5. *What happens when α is very large?* All coefficients are 0 and the model predicts the mean.
6. *Can Lasso be used for feature selection before another model?* Yes, e.g. `SelectFromModel(LassoCV())`.

---

## Summary

- Lasso = least squares + $\alpha\sum\lvert w_j\rvert$.
- It shrinks **and** zeroes coefficients, giving built-in feature selection.
- Zeros come from the constant-strength L1 push (soft-thresholding) and the diamond's corners.
- No closed form (coordinate descent); scale features; tune α with `LassoCV`.
- Weakness with correlated features leads to Elastic Net.
