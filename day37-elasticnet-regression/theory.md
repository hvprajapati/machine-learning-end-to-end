# Elastic Net Regression: Theory

> Builds on Day 35 (Ridge) and Day 36 (Lasso).

---

## 1. Motivation

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Feature selection | ❌ keeps every feature | ✅ |
| Correlated features | ✅ shares weight across the group | ❌ keeps one, drops the rest arbitrarily |
| $p > n$ (more features than samples) | ✅ | ⚠️ selects at most $n$ features |

**Elastic Net** (Zou & Hastie, 2005) combines both penalties to get feature selection **and** stable handling of
correlated features.

---

## 2. Definition (scikit-learn form)

$$
\boxed{\;J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(y_i-\hat y_i)^2 \;+\; \alpha\,\rho\sum_{j}\lvert w_j\rvert \;+\; \frac{\alpha\,(1-\rho)}{2}\sum_j w_j^2\;}
$$

| Symbol | sklearn name | Meaning |
|---|---|---|
| α | `alpha` | **total** amount of regularisation |
| ρ | `l1_ratio` | **mix** between the penalties, $0 \le \rho \le 1$ |

| `l1_ratio` | Behaves like |
|---|---|
| 1 | Lasso |
| close to 0 | Ridge (use `Ridge` itself for exactly 0; coordinate descent is unreliable there) |
| in between | both: sparse **and** grouped |

---

## 3. The grouping effect

When features are highly correlated, Elastic Net tends to give them **similar coefficients** and to keep or drop them
**together**. The L2 part makes the problem strictly convex, so the solution is unique and spreads weight evenly across
near-identical features; the L1 part still removes irrelevant ones.

Notebook experiment: three copies of one signal (correlation ≈ 0.997) plus three pure-noise features.

| Model | Signal copies | Noise |
|---|---|---|
| Linear Regression | −0.82, 1.69, 2.18 (unstable, one negative) | small, non-zero |
| Ridge | 0.45, 1.23, 1.37 | small, non-zero |
| Lasso | **0.00**, 1.22, 1.62 (one copy dropped) | 0 |
| **Elastic Net** | **0.90, 0.97, 0.98** (shared evenly) | **0** |

---

## 4. Geometry

The constraint region $\rho\lVert w\rVert_1 + (1-\rho)\lVert w\rVert_2^2 \le t$ lies **between** the diamond and the
circle: a diamond with outward-curved edges.
- It **keeps the corners** on the axes, so exact zeros are still possible (L1 behaviour).
- Its **curved edges** make it strictly convex, so correlated features are treated evenly (L2 behaviour).

---

## 5. Solving and tuning

- Solved by **coordinate descent** (like Lasso); no closed form.
- Two hyperparameters, tuned together by cross-validation:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, random_state=0),
)
model.fit(X_train, y_train)
enet = model[-1]
print(enet.alpha_, enet.l1_ratio_)
```

Put more `l1_ratio` values close to 1 (0.9, 0.95, 0.99): that region is often where the best models are.
For each `l1_ratio`, `ElasticNetCV` builds its own α grid automatically.

If cross-validation picks `l1_ratio = 1`, the data preferred pure Lasso, which is a valid outcome (as on the diabetes
data in the notebook). Elastic Net *contains* Lasso and approximates Ridge as special cases.

---

## 6. Choosing between the three

```
Need feature selection?
├── No  ───────────────────────────────────────────► Ridge
└── Yes
    ├── Groups of correlated features, or p > n? ───► Elastic Net
    └── Otherwise ──────────────────────────────────► Lasso
Not sure? ─────────────────────────────────────────► ElasticNetCV (searches the whole range)
```

Typical Elastic Net domains: **genomics** (thousands of correlated genes), **text** (correlated words / n-grams),
**sensor and financial data** (correlated signals).

---

## 7. The whole chapter in one table

| | Linear | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| Penalty | – | $\alpha\sum w^2$ | $\alpha\sum\lvert w\rvert$ | both, mixed by `l1_ratio` |
| Shrinks coefficients | ❌ | ✅ | ✅ | ✅ |
| Exact zeros | ❌ | ❌ | ✅ | ✅ |
| Correlated features | unstable | shared | one picked | shared **and** selected |
| Closed form | ✅ | ✅ | ❌ | ❌ |
| Hyperparameters | – | α | α | α, `l1_ratio` |
| Tuned sklearn class | `LinearRegression` | `RidgeCV` | `LassoCV` | `ElasticNetCV` |

**Always:** scale features, tune by cross-validation, compare against plain Linear Regression as a baseline.

---

## 8. Interview questions

1. *What is Elastic Net?* Linear regression with a weighted combination of L1 and L2 penalties.
2. *What do `alpha` and `l1_ratio` control?* Total penalty strength; the L1/L2 mix.
3. *What is the grouping effect?* Correlated features get similar coefficients and are selected or dropped together.
4. *When prefer Elastic Net over Lasso?* Correlated feature groups, or more features than samples.
5. *What does `l1_ratio = 1` mean?* Pure Lasso.
6. *Why does Elastic Net still produce zeros?* Its constraint region keeps the diamond's corners on the axes.

---

## Summary

- Elastic Net = least squares + L1 + L2.
- `alpha` controls how much, `l1_ratio` controls what kind.
- It selects features like Lasso and keeps correlated features together like Ridge.
- Tune both with `ElasticNetCV`; scale features first.
