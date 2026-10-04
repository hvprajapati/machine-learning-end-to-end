# Regression Evaluation Metrics: A Clear, Practical Guide

> **Running example:** predict a student's **placement package (in LPA)** from their **CGPA**.

## Contents

1. [Why do we need regression metrics?](#1-why-do-we-need-regression-metrics)
2. [Mean Absolute Error (MAE)](#2-mean-absolute-error-mae)
3. [Mean Squared Error (MSE)](#3-mean-squared-error-mse)
4. [Root Mean Squared Error (RMSE)](#4-root-mean-squared-error-rmse)
5. [MAE vs MSE vs RMSE](#5-mae-vs-mse-vs-rmse)
6. [R² Score](#6-r-score-coefficient-of-determination)
7. [Adjusted R²](#7-adjusted-r)
8. [Applying the metrics in the notebook](#8-applying-the-metrics-correctly-in-the-notebook)
9. [Final summary](#9-final-summary-which-metric-answers-which-question)

---

## 1. Why do we need regression metrics?

A regression model predicts a **continuous numerical value**. After training it, we need to answer three questions:

| # | Question | Answered by |
|:-:|---|---|
| 1 | How far are the predictions from the actual values? | MAE, MSE, RMSE |
| 2 | How much better is the model than simply predicting the average package? | R² |
| 3 | Does adding more features genuinely help, or just make the model more complicated? | Adjusted R² (+ test set / cross-validation) |

No single metric answers all three, which is why we use several.

### Notation

| Symbol | Meaning |
|:-:|---|
| $y_i$ | actual value for observation $i$ |
| $\hat{y}_i$ | predicted value for observation $i$ |
| $\bar{y}$ | mean of the actual values |
| $n$ | number of observations being evaluated |
| $p$ | number of input features (predictors) used by the model |

The **residual** is the prediction error for one observation:

$$
e_i = y_i - \hat{y}_i
$$

> **Example:** actual package = 8 LPA.
> Predicted 7 LPA → residual $= 8 - 7 = +1$ LPA (underprediction).
> Predicted 9 LPA → residual $= 8 - 9 = -1$ LPA (overprediction).

---

## 2. Mean Absolute Error (MAE)

> **At a glance:** the average size of the errors, ignoring direction. **Unit:** same as the target (LPA). **Lower is better.**

### What does it tell us?

MAE takes the **absolute value** of every error, so positive and negative errors do not cancel each other out.

### Formula

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i-\hat{y}_i\rvert
$$

### Example

Actual packages `[6, 8, 10]` LPA, predicted `[5, 9, 9]` LPA:

| Actual | Predicted | Absolute error |
|---:|---:|---:|
| 6 | 5 | 1 |
| 8 | 9 | 1 |
| 10 | 9 | 1 |

$$
\mathrm{MAE} = \frac{1+1+1}{3} = 1 \text{ LPA}
$$

### How do we interpret it?

An MAE of **1 LPA** means the predictions are off by **1 LPA on average** (in absolute terms).

- MAE uses the same unit as the target: package in LPA → MAE in LPA.
- Easy to explain: each error contributes in direct proportion to its size.

### Why do we need MAE?

It gives a straightforward answer to: **"Typically, how far are our predicted packages from the actual packages?"**

---

## 3. Mean Squared Error (MSE)

> **At a glance:** the average of the **squared** errors. **Unit:** squared target unit (LPA²). **Lower is better.** Punishes large errors heavily.

### What does it tell us?

Squaring makes every error non-negative and **penalizes large errors much more than small ones**.

### Formula

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### Example

Actual packages `[6, 8, 10]`, predicted `[5, 9, 7]`:

| Actual | Predicted | Error | Squared error |
|---:|---:|---:|---:|
| 6 | 5 | 1 | 1 |
| 8 | 9 | −1 | 1 |
| 10 | 7 | 3 | **9** |

$$
\mathrm{MSE} = \frac{1+1+9}{3} = \frac{11}{3} \approx 3.67\ \mathrm{LPA}^2
$$

The error of 3 becomes **9** after squaring, so it has a much larger influence than an error of 1.

### How do we interpret it?

- MSE is in **squared units** (LPA²), so it is hard to explain directly.
- A few large mistakes can increase MSE substantially.

### Why do we need MSE?

- When we want to **penalize large prediction errors more strongly**.
- It is the most widely used **loss function** for training regression models.

---

## 4. Root Mean Squared Error (RMSE)

> **At a glance:** the square root of MSE. **Unit:** same as the target (LPA). **Lower is better.** Keeps MSE's strong penalty on large errors.

### Formula

$$
\mathrm{RMSE} = \sqrt{\mathrm{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

### Example

Continuing from the MSE example:

$$
\mathrm{RMSE} = \sqrt{3.67} \approx 1.91\ \mathrm{LPA}
$$

### How do we interpret it?

- RMSE is measured in LPA when the target is measured in LPA.
- Large errors affect RMSE more than they affect MAE.

### Why do we need RMSE if we already have MSE?

MSE is useful mathematically and for training, but its squared units are hard to interpret.
**RMSE gives a similar error summary in the target's original unit**, while still penalizing large errors more strongly.

> ⚠️ **Important:** RMSE is **not** the average absolute error. It is the square root of the average *squared* error,
> so it is always **≥ MAE**, and the gap grows when there are a few large errors.

---

## 5. MAE vs MSE vs RMSE

| Metric | What is averaged? | Unit (target = LPA) | Effect of large errors | Main use |
|---|---|:-:|---|---|
| **MAE** | absolute errors | LPA | proportional to error size | easy-to-explain typical error |
| **MSE** | squared errors | LPA² | strong penalty | penalize large errors; common loss function |
| **RMSE** | √ of mean squared error | LPA | strong penalty | error summary in the target's unit |

**Remember:**

| Metric | The question it answers |
|---|---|
| MAE | "How far off are predictions on average?" |
| MSE | "What is the average squared error?" |
| RMSE | "What is the squared-error-based error scale, in the original unit?" |

> These metrics measure **prediction error**. On their own they do **not** tell us how much better the model is than a simple baseline. That is R²'s job.

---

## 6. R² Score (Coefficient of Determination)

> **At a glance:** how much better the model is than always predicting the mean. **Unit:** none. **Higher is better** (max 1, can be negative).

### Why do we need R² if we already have MAE, MSE and RMSE?

MAE, MSE and RMSE measure the **size of errors**, and their values depend on the scale of the target.
An RMSE of 2 may be small for a target measured in thousands, but large for a target ranging from 0 to 5.

**R² compares the model's errors with those of a simple baseline that always predicts the mean** of the actual values.

### Formula

$$
R^2 = 1-\frac{\displaystyle\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\displaystyle\sum_{i=1}^{n}(y_i-\bar{y})^2}
= 1 - \frac{\text{model's squared error}}{\text{mean baseline's squared error}}
$$

- **Numerator:** the model's **sum of squared errors** (SSE / SS<sub>res</sub>).
- **Denominator:** the **total squared variation** around the actual mean (SST / SS<sub>tot</sub>).
- $\bar{y}$ is the mean of the actual target values in the evaluation set.

### How do we interpret R²?

| R² value | Meaning |
|:-:|---|
| $1.0$ | perfect predictions on the evaluated data |
| $0.0$ | no better than predicting the mean for every observation |
| $< 0$ | **worse** than predicting the mean |

An R² of **0.80** means the model explains **80% of the variation** in the target relative to the mean baseline.
It does **not** mean that 80% of predictions are correct.

### Important points

- Higher R² is better only when comparing models on the **same evaluation data and target**.
- R² can be **negative** when predictions are worse than the mean baseline.
- R² does not tell you the error in LPA; use MAE or RMSE for that.
- A high R² does not prove that the model is causal, unbiased, or reliable on new data.

---

## 7. Adjusted R²

> **At a glance:** R² with a penalty for each extra feature. **Unit:** none. **Higher is better.** Used to compare models with different numbers of features.

### The problem with ordinary R²

On the training data, ordinary R² **never decreases** when a feature is added to a linear regression, even if the
new feature is random noise.

Suppose we predict package using:

1. **CGPA**: a feature that contains useful information.
2. **Random feature**: a randomly generated number with no real link to placement.

Ordinary R² can reward the second, more complex model for fitting the training data, even though the extra feature adds
nothing real. This is one reason training performance alone is not enough to judge a model.

### What does Adjusted R² do?

It **penalizes each added predictor** relative to the number of observations, so a new feature must improve the fit
**enough to pay for the added complexity**.

### Formula

$$
R^2_{\mathrm{adj}} = 1-(1-R^2)\,\frac{n-1}{n-p-1}
$$

| Symbol | Meaning |
|:-:|---|
| $R^2$ | ordinary R² |
| $n$ | number of observations used to calculate R² |
| $p$ | number of input features (**not** counting the intercept) |

### Why does the formula use $n$ and $p$?

- $n-1$: degrees of freedom of the variation around the mean.
- $n-p-1$: degrees of freedom left after estimating $p$ coefficients and an intercept.
- More predictors → fewer remaining degrees of freedom → a larger penalty factor $\frac{n-1}{n-p-1}$.

### How do we interpret it?

- It **increases** when a useful feature improves the fit enough.
- It **decreases** when a feature does not improve the fit enough to justify its complexity.
- It is always **≤ ordinary R²** (for linear regression with an intercept, on the same data).
- It can be **negative**.

### R² vs Adjusted R²

| | R² | Adjusted R² |
|---|---|---|
| Measures | fit relative to a mean-prediction baseline | the same fit, adjusted for the number of predictors |
| Adding a predictor | never penalized | penalized unless the fit improves enough |
| Best used for | describing model fit | comparing linear models with different numbers of predictors on the same observations |

> ⚠️ **Important:** Adjusted R² is **not** a replacement for testing on unseen data. Use a held-out test set or
> cross-validation to check whether extra features actually help the model generalize.

---

## 8. Applying the metrics correctly in the notebook

### Step 1: Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
```

The model learns from `X_train` / `y_train`. The test set is held out to estimate performance on data the model has never seen.

### Step 2: Predict and compute the metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = lr.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

r2 = r2_score(y_test, y_pred)
print("R²  :", r2)
```

### Step 3: Adjusted R², using the correct values

```python
n = X_test.shape[0]   # number of test observations
p = X_test.shape[1]   # number of input features

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R²:", adjusted_r2)
```

> 💡 **Do not hard-code `40`.** With 200 rows and `test_size=0.2` the test set has 40 rows, but computing `n` from
> `X_test.shape[0]` keeps the code correct if the dataset or split changes.
>
> The denominator $n-p-1$ must be positive. The formula is undefined when there are too few observations relative to the number of features.

### Note: the random-feature experiment

The notebook compares a model using **CGPA** with one using **CGPA + a random feature**. This is useful for showing
why adding features can make ordinary R² misleading **on training data**.

However, the notebook calculates R² on the **test set**:
- Test-set R² is **not** guaranteed to increase when a feature is added, because the test rows were not used for fitting.
- Adjusted R² on a test set is a descriptive calculation; its classic complexity-penalty meaning applies to the data the model was fitted on.
- To judge whether a feature helps, compare models on the same held-out test set or use **cross-validation**. Do not rely on adjusted R² alone.

### Note: the `iq` experiment ⚠️ target leakage

```python
new_df2['iq'] = new_df2['package'] + (np.random.randint(-12, 12, 200) / 10)
```

This feature is built from the **answer itself** (`package`). In a real prediction task the actual package is not
available at prediction time, so this is **target leakage**: the model looks excellent only because the feature
contains the target.

It is a good demonstration of leakage, but `iq` must **not** be treated as a legitimate feature for a real placement model.

---

## 9. Final summary: which metric answers which question?

| Question | Metric |
|---|---|
| By how many LPA are predictions typically off, ignoring direction? | **MAE** |
| Should large errors get a stronger penalty? | **MSE** |
| What is the squared-error-based error scale, in LPA? | **RMSE** |
| How does the model compare with predicting the mean? | **R²** |
| Does the fit justify adding more predictors? | **Adjusted R²**, alongside test-set or cross-validation results |

### Key takeaway

> **MAE, MSE and RMSE measure prediction error. R² compares squared-error performance with a mean baseline.
> Adjusted R² adds a penalty for the number of predictors.**
>
> Use them together: each answers a different question, and none makes the others unnecessary.
