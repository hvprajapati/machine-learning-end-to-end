# Day 35 — Regularisation & Ridge Regression

**Part 1 of the regularisation chapter:** Day 35 Ridge → Day 36 Lasso → Day 37 Elastic Net.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Overfitting, Ridge loss, coefficient behaviour, bias–variance, closed form & gradient descent, geometry, scaling, interview questions | **1. Read first** |
| [`01-ridge-intuition.ipynb`](01-ridge-intuition.ipynb) | See overfitting → the penalty idea → Ridge on a line, a polynomial, and real diabetes data → choosing α with `RidgeCV` | **2** |
| [`02-ridge-key-understandings.ipynb`](02-ridge-key-understandings.ipynb) | Four pictures: coefficients shrink, big ones shrink most, bias–variance trade-off, loss curve & circle geometry | **3** |
| [`03-EXTRA-ridge-from-scratch.ipynb`](03-EXTRA-ridge-from-scratch.ipynb) | **EXTRA (optional).** Ridge three ways (1-feature formula, normal equation, gradient descent), each matched against sklearn | **4** |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn. These cells are collapsed; just run them and read the plot.
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Learning objectives
- Recognise overfitting and explain why large coefficients are its symptom
- Write the Ridge loss and explain the role of α
- Describe how Ridge changes coefficients (shrinks, never zeroes, big ones most)
- Explain the bias–variance trade-off controlled by α, and tune α with cross-validation
- Implement Ridge from scratch (closed form and gradient descent)
