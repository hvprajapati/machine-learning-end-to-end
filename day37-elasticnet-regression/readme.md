# Day 37 — Elastic Net Regression

**Part 3 of the regularisation chapter:** Day 35 Ridge → Day 36 Lasso → Day 37 Elastic Net.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Combined L1 + L2 loss, `alpha` vs `l1_ratio`, grouping effect, geometry, tuning, decision guide, chapter summary | **1. Read first** |
| [`01-elastic-net.ipynb`](01-elastic-net.ipynb) | Correlated-features experiment (Linear vs Ridge vs Lasso vs Elastic Net), the `l1_ratio` dial, the three constraint shapes, final tuned comparison on diabetes data, which model to use | **2** |

## How to read the notebook
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn. These cells are collapsed; just run them and read the plot.
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Learning objectives
- Write the Elastic Net loss and explain `alpha` and `l1_ratio`
- Explain the grouping effect and why Lasso alone fails on correlated features
- Tune Elastic Net with `ElasticNetCV`
- Choose between Linear, Ridge, Lasso and Elastic Net for a given problem
