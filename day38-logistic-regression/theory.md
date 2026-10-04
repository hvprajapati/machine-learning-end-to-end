# Logistic Regression: Theory

> Read this before the notebooks. The notebooks follow the same path as this file:
> **Perceptron trick → Sigmoid → Log-loss + Gradient descent**.
> Section numbers (§) are referenced from the notebooks.

---

## 1. The problem: binary classification

We have features $x = (x_1, \dots, x_d)$ and a label $y \in \{0, 1\}$
(placed / not placed, spam / not spam, disease / healthy). We want a model that:

1. separates the two classes, and
2. tells us **how confident** it is: a probability $P(y = 1 \mid x)$.

### Why not use linear regression?
| Problem | Effect |
|---|---|
| Output $w^\top x + b$ is unbounded ($-\infty$ to $+\infty$) | It cannot be read as a probability (e.g. 1.7 or −0.3) |
| Squared error punishes points that are "too correct" | A far-away but correctly labelled point drags the line and shifts the 0.5 threshold |

Logistic regression keeps the **linear part** but passes it through a function that maps any
number into $(0, 1)$, and trains with a loss designed for probabilities.

> Despite its name, logistic regression is a **classification** algorithm. "Regression" refers to
> the fact that it regresses the **log-odds** linearly on the features (§5).

---

## 2. Geometry: the decision boundary

A linear classifier computes a score

$$
z = w^\top x + b = w_1x_1 + w_2x_2 + \dots + w_dx_d + b
$$

- $z > 0$ → predict class **1** (the "positive side")
- $z < 0$ → predict class **0** (the "negative side")
- $z = 0$ → the **decision boundary**: a line in 2-D, a plane in 3-D, a hyperplane in general

The magnitude $|z|$ grows with the distance from the boundary (the true distance is $|z| / \lVert w \rVert$),
so $z$ also measures *how far into a region* a point is.

**Plotting the boundary in 2-D** (used in every notebook). Solve $w_1x_1 + w_2x_2 + b = 0$ for $x_2$:

$$
x_2 = -\frac{w_1}{w_2}\,x_1 - \frac{b}{w_2}
\qquad\Rightarrow\qquad
\text{slope } m = -\frac{w_1}{w_2},\quad \text{intercept } c = -\frac{b}{w_2}
$$

In code: `m = -(coef_[0]/coef_[1])` and `b = -(intercept_/coef_[1])`.

**Bias trick.** Add a constant column of 1s to $X$ and put $b$ inside the weight vector as $w_0$.
Then $z = w^\top x$ with $x_0 = 1$. This is what `np.insert(X, 0, 1, axis=1)` does in the notebooks.

---

## 3. Step 1: the Perceptron trick (`01-perceptron-trick.ipynb`)

The simplest way to learn $w$: start with any line and **nudge it toward every misclassified point**.

```
initialise w (e.g. all ones)
repeat for many epochs:
    pick a random point (x_i, y_i)
    ŷ_i = step(w · x_i)          # step(z) = 1 if z > 0 else 0
    w   = w + η (y_i − ŷ_i) x_i
```

$(y_i - \hat y_i)$ can only be:

| Case | $y_i - \hat y_i$ | Update |
|---|---|---|
| Correctly classified | 0 | no change |
| Positive point predicted 0 | +1 | $w \leftarrow w + \eta x_i$: the line rotates/shifts so $x_i$ moves to the positive side |
| Negative point predicted 1 | −1 | $w \leftarrow w - \eta x_i$: the line moves so $x_i$ goes to the negative side |

$\eta$ (learning rate) controls the size of each nudge.

**Weakness.** Once every point is correctly classified, *all updates become 0* and learning stops.
The final line is simply the first one that happened to separate the data, often hugging one class.
It is a valid separator, but not the *best* one, and it gives no probabilities.

---

## 4. Step 2: replace the step with the Sigmoid (`02-perceptron-trick-sigmoid.ipynb`)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

| $z$ | −∞ | −2 | 0 | 2 | +∞ |
|---|---|---|---|---|---|
| $\sigma(z)$ | 0 | 0.12 | **0.5** | 0.88 | 1 |

Properties used later:
- Output is always in $(0, 1)$, so it can be read as a **probability**.
- $\sigma(0) = 0.5$, so the 0.5 probability threshold is exactly the boundary $z = 0$.
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$.
- Simple derivative: $\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)$.

**Effect on the update rule.** With $\hat y = \sigma(w^\top x)$, the term $(y - \hat y)$ is **never exactly 0**:
- A correctly classified point close to the line (e.g. $y=1$, $\hat y = 0.6$) still gives a push of $+0.4$ and moves the line *away* from itself.
- A point far on its correct side ($\hat y \approx 0.999$) pushes with almost no force.

So every point keeps pushing the line away, and points near the line push hardest. The line settles
**in the middle** of the two classes instead of stopping at the first separator. This is already logistic
regression, trained with stochastic updates; §6–7 show *why* this exact rule is correct.

---

## 5. The model and its interpretation

$$
\boxed{\;\hat p = P(y = 1 \mid x) = \sigma(w^\top x + b)\;}
\qquad
\hat y = \begin{cases} 1 & \hat p \ge 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

### Log-odds (logit)
Invert the sigmoid:

$$
\log\frac{\hat p}{1 - \hat p} = w^\top x + b
$$

$\frac{p}{1-p}$ is the **odds** (p = 0.8 → odds 4 : 1). Logistic regression says the **log-odds are a linear
function of the features**. That is its core assumption.

### Reading a coefficient
- $w_j > 0$: increasing $x_j$ raises the probability of class 1; $w_j < 0$ lowers it.
- **Odds ratio** $e^{w_j}$: a one-unit increase in $x_j$ (others fixed) **multiplies the odds** by $e^{w_j}$.
  Example: $w_j = 0.7$ → $e^{0.7} \approx 2.01$ → the odds roughly double.
- Compare magnitudes only when features are on the same scale (standardise first).

---

## 6. The loss function: Log-loss (Binary Cross-Entropy)

### Derived from Maximum Likelihood
For one point, the model gives probability $\hat p_i$ to class 1 and $1 - \hat p_i$ to class 0, so the
probability it assigns to the **true** label is

$$
P(y_i \mid x_i) = \hat p_i^{\,y_i}\,(1 - \hat p_i)^{1 - y_i}
$$

Training = choose $w$ that makes the observed labels most probable (maximise the product over all points).
Taking $-\log$ and averaging turns *maximise product* into *minimise sum*:

$$
\boxed{\;L(w) = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\Big]\;}
$$

### What it does per point
| True $y$ | Predicted $\hat p$ | Loss |
|---|---|---|
| 1 | 0.9 | $-\log 0.9 = 0.105$ (small) |
| 1 | 0.5 | $-\log 0.5 = 0.693$ |
| 1 | 0.1 | $-\log 0.1 = 2.303$ (large) |
| 1 | 0.001 | 6.9; confident and wrong is punished very hard |

### Why not Mean Squared Error?
With a sigmoid inside, MSE is **non-convex** in $w$ (local minima, flat regions), and its penalty for
confident mistakes is capped at 1. Log-loss is **convex**: one global minimum, so gradient descent
reliably finds it.

---

## 7. Training with Gradient Descent (`03-gradient-descent.ipynb`)

Using $\sigma' = \sigma(1-\sigma)$ and the chain rule, the gradient of the log-loss is remarkably simple:

$$
\frac{\partial L}{\partial w} = \frac{1}{n}\,X^\top(\hat p - y)
$$

(with the bias trick, this includes $b$). The gradient-descent update is therefore

$$
\boxed{\;w \leftarrow w - \eta\,\frac{1}{n}X^\top(\hat p - y) \;=\; w + \eta\,\frac{1}{n}X^\top(y - \hat p)\;}
$$

```python
y_hat   = sigmoid(X @ w)
w       = w + lr * (X.T @ (y - y_hat)) / n
```

**Connection to §3–4.** This is the same formula as the sigmoid-perceptron update, applied to *all*
points at once and averaged. The perceptron trick with sigmoid was gradient descent on log-loss all along
(stochastic, one point at a time).

| Variant | Points per update | Notebook |
|---|---|---|
| Stochastic GD | 1 random point | 02 (sigmoid perceptron) |
| Batch GD | all $n$ points | 03 |
| Mini-batch GD | a small batch | (Day 33) |

There is **no closed-form solution** (unlike the normal equation in linear regression), so an iterative
optimiser is always required. scikit-learn uses faster second-order solvers (`lbfgs` by default) instead
of plain gradient descent.

### Worked example: one gradient step
One point $x = (1, 2)$, $y = 1$; start $w = (0.5, -0.25)$, $b = 0$; $\eta = 0.1$.

| | Before | After one step |
|---|---|---|
| $z = w^\top x + b$ | $0.5 - 0.5 + 0 = 0$ | $0.55 - 0.30 + 0.05 = 0.30$ |
| $\hat p = \sigma(z)$ | 0.500 | 0.574 |
| Loss $-\log \hat p$ | 0.693 | 0.554 |

Gradient $= (\hat p - y)\,x = -0.5 \cdot (1, 2) = (-0.5, -1)$, bias gradient $= -0.5$
→ $w = (0.5, -0.25) - 0.1(-0.5, -1) = (0.55, -0.15)$, $b = 0.05$.
The probability of the correct class went up and the loss went down.

---

## 8. Perfectly separable data and regularisation

If a line separates the classes perfectly, the loss can always be lowered further by **scaling $w$ up**:
predictions move closer to exactly 0 and 1, and the loss approaches 0 without ever reaching it.
Unregularised training therefore never converges; the weights keep growing.
All three notebooks use separable data, which is why our own gradient descent and scikit-learn
give boundaries with similar position but **very different weight sizes**.

**Regularisation** adds a penalty on large weights (same idea as Ridge / Lasso, Days 35–37):

$$
L_{\text{reg}}(w) = L(w) + \frac{1}{C}\cdot\text{penalty}(w)
$$

In scikit-learn (≥ 1.8):

| Setting | Meaning |
|---|---|
| `C=1.0` (default) | **Inverse** regularisation strength: smaller C = stronger penalty, simpler model |
| `l1_ratio=0.0` (default) | L2 penalty (Ridge-like, shrinks all weights) |
| `l1_ratio=1.0` | L1 penalty (Lasso-like, can set weights exactly to 0); needs `solver='saga'` or `'liblinear'` |
| `C=np.inf` | No regularisation (replaces the old `penalty='none'` / `penalty=None`) |

> The `penalty` argument is deprecated since scikit-learn 1.8. Use `C` and `l1_ratio` instead.

---

## 9. Logistic regression in scikit-learn

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train, y_train)

clf.predict(X_test)            # class labels (threshold 0.5)
clf.predict_proba(X_test)      # [[P(y=0), P(y=1)], ...]
clf.decision_function(X_test)  # raw score z = w·x + b
clf.coef_, clf.intercept_      # learned w and b
```

| Parameter | Default | Notes |
|---|---|---|
| `C` | 1.0 | Inverse regularisation strength (§8) |
| `l1_ratio` | 0.0 | 0 = L2, 1 = L1, in between = Elastic-Net |
| `solver` | `'lbfgs'` | `'liblinear'` for small data + L1; `'saga'` for large data / L1 / Elastic-Net |
| `max_iter` | 100 | Raise it if you see a `ConvergenceWarning` |
| `class_weight` | None | `'balanced'` re-weights classes for imbalanced data |

**Changing the threshold.** `predict` always uses 0.5. To trade precision for recall (Day 39), threshold
the probabilities yourself: `(clf.predict_proba(X)[:, 1] >= 0.3).astype(int)`.

**More than two classes.** scikit-learn uses the **softmax** (multinomial) generalisation automatically
(Day 40).

**Non-linear boundaries.** The boundary is always linear in the *input features*. For curved boundaries,
add polynomial features first (Day 40).

---

## 10. Assumptions and practical requirements

| Assumption / requirement | Why it matters |
|---|---|
| Log-odds are linear in the features | Otherwise the linear boundary underfits; add polynomial/interaction features |
| Observations are independent | Standard errors and probabilities assume it |
| Low multicollinearity | Correlated features make individual coefficients unstable and hard to interpret |
| Features scaled (StandardScaler) | Faster, more stable optimisation; regularisation treats all features fairly |
| Enough data per feature | Rule of thumb: ≥ 10 events of the minority class per feature |

**Not required:** normally distributed features, or equal variances (unlike LDA).

---

## 11. Strengths and limitations

| Strengths | Limitations |
|---|---|
| Outputs calibrated **probabilities** | Only **linear** boundaries without feature engineering |
| Fast to train and predict, scales well | Sensitive to outliers in feature space and to multicollinearity |
| **Interpretable** coefficients (odds ratios) | Weights blow up on perfectly separable data without regularisation |
| Convex loss, so a unique solution with regularisation | Usually outperformed by tree ensembles on complex tabular data |
| Strong baseline for any classification task | Needs encoded categorical features and imputed missing values |

---

## 12. Common mistakes

1. **Calling it a regression model.** It predicts classes and probabilities.
2. **Using accuracy alone** on imbalanced data. Use precision, recall, F1 and ROC-AUC (Day 39).
3. **Skipping scaling**, then wondering about `ConvergenceWarning` or unfair regularisation.
4. **Interpreting raw coefficients as probability changes.** A coefficient changes the **log-odds**; $e^{w_j}$ is the odds multiplier.
5. **Using `penalty='none'`.** Removed in modern scikit-learn; use `C=np.inf`.
6. **Assuming 0.5 is always the right threshold.** Pick it from the business cost of false positives vs false negatives.

---

## 13. Interview questions

1. *Why is it called regression if it classifies?* It linearly models the log-odds of the positive class.
2. *Why sigmoid?* Maps any real score to $(0,1)$; $\sigma(0)=0.5$ matches the boundary; simple derivative.
3. *Which loss and why not MSE?* Log-loss, from maximum likelihood; it is convex, while MSE with a sigmoid is not.
4. *Write the gradient.* $\frac{1}{n}X^\top(\hat p - y)$.
5. *Closed-form solution?* No; it needs an iterative optimiser.
6. *What happens on perfectly separable data?* Weights grow without bound; regularisation fixes it.
7. *What does `C` do?* Inverse of regularisation strength; smaller C = stronger regularisation.
8. *How do you interpret $w_j = 0.7$?* One unit more of $x_j$ multiplies the odds of class 1 by $e^{0.7} \approx 2$.
9. *Perceptron vs logistic regression?* Perceptron uses a step function and stops at any separator; logistic regression uses a sigmoid + log-loss, gives probabilities, and finds a better-placed boundary.

---

## Summary

- Score $z = w^\top x + b$ defines a **linear decision boundary** $z = 0$.
- The **perceptron trick** moves the line toward misclassified points, but stops at the first separator.
- Replacing step with **sigmoid** makes every point contribute, giving probabilities and a well-centred line.
- That update rule is **gradient descent on log-loss**: $w \leftarrow w + \eta\frac{1}{n}X^\top(y - \hat p)$.
- Use **regularisation** (`C`), **scale features**, and evaluate with proper **classification metrics** (Day 39).
