# Day 45 — K-Nearest Neighbors (K-NN)

A supervised algorithm that classifies a point by a **majority vote of its K closest training points**.
Dataset: **Social Network Ads**: predict whether a user buys an SUV from their **Age** and **Estimated Salary**.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Algorithm, distance metrics, worked example by hand, choosing K, scaling, weighted voting, K-NN regression, complexity, curse of dimensionality, interview questions | **1. Read first** |
| [`01-knn-classification.ipynb`](01-knn-classification.ipynb) | Explore data → scale → train K-NN → see which neighbours voted → evaluate → decision boundary → scaling experiment → choosing K with cross-validation | **2** |
| [`02-knn-from-scratch.ipynb`](02-knn-from-scratch.ipynb) | Distance → sort → vote in NumPy, the hand example verified, a `MyKNN` class that matches scikit-learn exactly | **3** |
| `Social_Network_Ads.csv` | 400 users: `Age`, `EstimatedSalary`, `Purchased` (0/1) | data |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

| | Test accuracy |
|---|:-:|
| K = 5, without feature scaling | 0.83 |
| K = 5, with `StandardScaler` | **0.93** |
| K = 1 (overfits: training accuracy 1.00) | 0.87 |
| K = 11, chosen by 10-fold cross-validation | 0.93 |

## Learning objectives
- Explain how K-NN classifies, and why it is a lazy, non-parametric learner
- Compute distances and a K-NN prediction by hand
- Explain why feature scaling is mandatory, and demonstrate what breaks without it
- Choose K with cross-validation and recognise over- and underfitting
- Implement K-NN from scratch

## Connections
- **Day 7–8 — Scaling**: required before K-NN.
- **Day 22 — KNN Imputer**: the same neighbour idea, used to fill missing values.
- **Day 28 — PCA**: reduces dimensions before K-NN.
- **Day 38 — Logistic Regression**: linear boundary vs K-NN's non-linear one.
- **Day 39 — Classification metrics**: confusion matrix, precision and recall used here.
- **Day 44 — K-Means**: often confused with K-NN; see the comparison in `theory.md` §13.

Source: adapted from `Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)`.
