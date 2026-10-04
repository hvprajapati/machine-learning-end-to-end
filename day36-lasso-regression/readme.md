# Day 36 — Lasso Regression

**Part 2 of the regularisation chapter:** Day 35 Ridge → Day 36 Lasso → Day 37 Elastic Net.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | L1 penalty, why exact zeros appear (soft-thresholding, loss kink, diamond geometry), coordinate descent, limitations, Ridge vs Lasso | **1. Read first** |
| [`01-lasso-intuition.ipynb`](01-lasso-intuition.ipynb) | A slope that reaches exactly 0, Lasso selecting the important diabetes features, `LassoCV`, Lasso on a polynomial | **2** |
| [`02-lasso-key-understandings.ipynb`](02-lasso-key-understandings.ipynb) | Coefficients vanishing, Ridge vs Lasso paths, the loss-curve kink, diamond vs circle, bias–variance | **3** |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn. These cells are collapsed; just run them and read the plot.
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Learning objectives
- Write the Lasso loss and contrast it with Ridge
- Explain three ways why L1 creates exact zeros (constant push, loss kink, diamond corners)
- Use Lasso for automatic feature selection and tune α with `LassoCV`
- Know Lasso's weakness with correlated features, the motivation for Elastic Net
