# Day 47 — Naive Bayes

Classify by **probability** using Bayes' theorem, with the "naive" assumption that features are independent.
Dataset: **Social Network Ads**: predict whether a user buys an SUV from **Age** and **Estimated Salary** (same as Days 45–46).

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Probability basics, Bayes' theorem, the naive assumption, a spam filter worked by hand, Gaussian / Multinomial / Bernoulli NB, smoothing, comparison with other classifiers | **1. Read first** |
| [`01-naive-bayes-classification.ipynb`](01-naive-bayes-classification.ipynb) | Train GaussianNB → predict with probabilities → evaluate → **look inside the model** (means, bell curves) → **one prediction by hand** → scaling experiment → bonus spam filter | **2** |
| `Social_Network_Ads.csv` | 400 users: `Age`, `EstimatedSalary`, `Purchased` (0/1) | data |

## How to read the notebook
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results
- Test accuracy **0.90**, with a smooth curved boundary.
- The hand calculation for a new customer (age 30, salary 87,000) gives P(did not buy) = 0.899, exactly matching scikit-learn.
- Predictions are **identical with and without feature scaling**.

## Learning objectives
- Apply Bayes' theorem and explain prior, likelihood and posterior
- Explain the "naive" independence assumption
- Compute a Naive Bayes prediction by hand
- Choose between Gaussian, Multinomial and Bernoulli Naive Bayes
- Use Naive Bayes for numeric data and for text

## Classification days so far
| Day | Algorithm | Test accuracy (this dataset) |
|---|---|:-:|
| 38 | Logistic Regression *(Day 38 used other data; score computed here for comparison)* | 0.89 |
| 45 | K-NN | 0.93 |
| 46 | SVM (linear / rbf) | 0.90 / 0.93 |
| 47 | Naive Bayes | 0.90 |

Source: adapted from `Part 3 - Classification/Section 18 - Naive Bayes`.
