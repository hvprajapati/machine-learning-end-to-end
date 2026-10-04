# Day 46 — Support Vector Machine (SVM) & Support Vector Regression (SVR)

Separate the classes with the line that has the **widest possible margin**, and the regression version that fits
a curve inside an **ε-tube**.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Margin, support vectors, simple maths, the `C` parameter, scaling, kernel preview, comparison with other classifiers, **SVR (§12)** | **1. Read first** |
| [`01-svm-classification.ipynb`](01-svm-classification.ipynb) | Social Network Ads: scale → train linear SVM → evaluate → **see the margin and support vectors** → effect of `C` → linear vs RBF kernel | **2** |
| [`02-svr-regression.ipynb`](02-svr-regression.ipynb) | Position Salaries: scale X **and** y → train SVR → **see the ε-tube** → effect of `C` → what breaks without scaling | **3** |
| `Social_Network_Ads.csv` | 400 users: `Age`, `EstimatedSalary`, `Purchased` (0/1) | data |
| `Position_Salaries.csv` | 10 job levels and salaries | data |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

| Model | Result |
|---|---|
| Linear SVM, C = 1 | test accuracy 0.90 (128 support vectors) |
| Linear SVM, C = 0.01 | test accuracy 0.87 (206 support vectors, wide margin) |
| RBF-kernel SVM | test accuracy **0.93** |
| SVR, C = 1 | level 6.5 → ≈ 170,000; CEO underpredicted (≈ 558,000 vs 1,000,000) |
| SVR, C = 10 | CEO → ≈ 972,000 |
| SVR without scaling | the same ≈ 130,000 for every level |

## Learning objectives
- Explain the maximum-margin idea and what support vectors are
- Explain how `C` trades margin width against mistakes
- Explain SVR's ε-tube and why both X and y must be scaled
- Train, evaluate and visualise SVM and SVR in scikit-learn

Sources: adapted from `Part 3 - Classification/Section 16` and `Part 2 - Regression/Section 7`.
