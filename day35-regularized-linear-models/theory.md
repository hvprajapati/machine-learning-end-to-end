# Regularisation & Ridge Regression: Theory

> Companion to the notebooks in this folder. Days 36 (Lasso) and 37 (Elastic Net) build directly on this file.

---

## 1. The problem regularisation solves

A model **overfits** when it learns the noise in the training data instead of the underlying pattern:
low training error, high test error.

Typical causes in linear models:
- **Too many features** relative to the number of samples (including polynomial features)
- **Multicollinearity**: features that are strongly correlated with each other
- **Noisy data**

The tell-tale symptom is **very large coefficients** that partly cancel each other out. In the notebook, the
unregularised diabetes model has weights up to ≈ 900, and a degree-15 polynomial reaches a test MSE of 166 against 0.5
for the right model.

**Regularisation** = adding a penalty on coefficient size to the loss, so the model prefers simpler solutions.

---

## 2. Ridge regression (L2 regularisation)

$$
\boxed{\;J(w, b) = \sum_{i=1}^{n}\big(y_i - \hat y_i\big)^2 \;+\; \alpha\sum_{j=1}^{p} w_j^2\;}
\qquad \hat y_i = w^\top x_i + b
$$

| Term | Role |
|---|---|
| $\sum (y_i - \hat y_i)^2$ | fit the training data (ordinary least squares) |
| $\alpha \sum w_j^2$ | penalise large weights (the **L2 norm** squared, $\lVert w\rVert_2^2$) |
| $\alpha \ge 0$ | regularisation strength, a **hyperparameter** |

- The intercept $b$ is **not** penalised; it only shifts predictions and has nothing to do with model complexity.
- $\alpha = 0$ → ordinary linear regression. $\alpha \to \infty$ → all $w_j \to 0$, so the model predicts $\bar y$.
- Also called **Tikhonov regularisation** or **weight decay** (in neural networks).

---

## 3. What Ridge does to the coefficients

1. **Shrinks all coefficients toward 0**, smoothly, as α increases.
2. **Never sets a coefficient exactly to 0** (no feature selection). See Day 36 for Lasso.
3. **Large coefficients are shrunk the most.** The penalty's gradient is $2\alpha w_j$, proportional to the weight itself.
4. **Stabilises correlated features.** Instead of one huge positive and one huge negative weight, correlated features
   share the effect. Small weights can even grow temporarily while large correlated ones are tamed.
5. **Always has a unique solution.** $X^\top X + \alpha I$ is invertible for any α > 0, even with perfectly collinear
   features or more features than samples. This was Ridge's original motivation (Hoerl & Kennard, 1970).

---

## 4. Bias–variance trade-off

$$
\text{Expected test error} = \text{Bias}^2 + \text{Variance} + \text{irreducible noise}
$$

| α | Bias | Variance | Result |
|---|---|---|---|
| too small | low | **high** | overfitting |
| well chosen | moderate | moderate | **lowest total error** |
| too large | **high** | low | underfitting |

Ridge deliberately accepts a little bias (coefficients are pulled away from their least-squares values) in exchange for
a large reduction in variance. The notebook simulates 200 training sets to show this U-shaped total error directly.

---

## 5. Solving Ridge

### 5a. One feature (closed form)
$$
m = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2 + \alpha}, \qquad b = \bar y - m\bar x
$$
α sits in the denominator: a larger α gives a smaller slope.

### 5b. Many features (normal equation)
With a column of 1s prepended to $X$ for the intercept:
$$
w = \big(X^\top X + \alpha I'\big)^{-1} X^\top y, \qquad I' = I \text{ with } I'_{00} = 0 \text{ (intercept not penalised)}
$$

### 5c. Gradient descent
$$
\nabla_w J = 2\big(X^\top X w - X^\top y + \alpha I' w\big), \qquad w \leftarrow w - \eta\,\nabla_w J
$$
Converges to the same solution, but can need many iterations when features are correlated (an ill-conditioned problem).
In the notebook, 500 epochs leave coefficients off by ≈ 56, while 20 000 epochs match scikit-learn exactly.

---

## 6. Geometric interpretation

Ridge is equivalent to the constrained problem

$$
\min_w \sum (y_i - \hat y_i)^2 \quad \text{subject to} \quad \sum w_j^2 \le t
$$

(every α corresponds to some budget $t$). The constraint region is a **circle** (a hypersphere in higher dimensions).
The solution is where the smallest error contour (an ellipse) touches the circle. A circle has no corners, so the
touching point almost never lies on an axis, which is why Ridge does not produce zeros.

---

## 7. Practical usage

### Always scale features first
The penalty treats all coefficients equally, but a coefficient's size depends on its feature's units (metres vs
millimetres would change it by ×1000). Standardise so the penalty is fair:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import numpy as np

model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 50)))
model.fit(X_train, y_train)
best_alpha = model[-1].alpha_
```

(The diabetes dataset used in the notebooks is already standardised, which is why it skips this step.)

### Choose α by cross-validation
Search α on a **log scale** (0.0001 … 1000). `RidgeCV` does this efficiently. Never tune α on the test set.

### scikit-learn reference

| Class | Use |
|---|---|
| `Ridge(alpha=1.0)` | fixed α |
| `RidgeCV(alphas=[...])` | picks α by (efficient leave-one-out) cross-validation |
| `SGDRegressor(penalty='l2', alpha=...)` | very large datasets (stochastic gradient descent) |
| `RidgeClassifier` / `LogisticRegression(C=...)` | the same L2 idea for classification ($C = 1/\alpha$) |

---

## 8. Common mistakes

1. **Not scaling features**, so the penalty punishes features with small units unfairly.
2. **Tuning α on the test set**, which gives over-optimistic results. Use cross-validation.
3. **Searching α on a linear grid** (1, 2, 3 …). Its effect spans orders of magnitude, so use `np.logspace`.
4. **Expecting Ridge to remove features.** It shrinks; it does not select.
5. **Penalising the intercept** in a from-scratch implementation.

---

## 9. Interview questions

1. *What is the Ridge loss?* $\sum (y-\hat y)^2 + \alpha\sum w_j^2$.
2. *What happens as α → 0 and α → ∞?* OLS; all weights → 0 (predicts the mean).
3. *Why does Ridge help with multicollinearity?* Adding $\alpha I$ makes $X^\top X$ well-conditioned and invertible, and correlated features share the weight.
4. *Can Ridge set a coefficient to zero?* No, only shrink it toward zero.
5. *Why scale features before Ridge?* The penalty depends on coefficient magnitude, which depends on feature units.
6. *How is α chosen?* Cross-validation over a log-spaced grid (`RidgeCV`).
7. *Which coefficients shrink most?* The largest ones; the penalty gradient $2\alpha w$ is proportional to $w$.

---

## Summary

- Overfitting shows up as large, unstable coefficients; regularisation penalises them.
- Ridge adds $\alpha\sum w_j^2$: it shrinks every weight, the large ones most, but never to exactly 0.
- α trades variance for bias; choose it by cross-validation on a log scale.
- Closed form $(X^\top X + \alpha I)^{-1}X^\top y$; always scale features first.
