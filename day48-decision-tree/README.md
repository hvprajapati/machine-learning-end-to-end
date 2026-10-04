# Day 48 — Decision Trees (Classification & Regression)

A learned **flowchart of yes/no questions**: easy to read, no scaling needed, but prone to overfitting.

## Folder structure

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | How splits are chosen (Gini / entropy worked example), overfitting and pruning, rectangular boundaries, regression trees, feature importance | **1. Read first** |
| [`01-decision-tree-classification.ipynb`](01-decision-tree-classification.ipynb) | Social Network Ads: train → evaluate → **read the tree's rules** → boxy boundaries → overfitting vs `max_depth` → tuning → scaling proof → feature importance | **2** |
| [`02-decision-tree-regression.ipynb`](02-decision-tree-regression.ipynb) | Position Salaries: predict a salary → the **staircase** curve → read the tree → effect of depth → no extrapolation | **3** |
| `Social_Network_Ads.csv` | 400 users: `Age`, `EstimatedSalary`, `Purchased` (0/1) | data |
| `Position_Salaries.csv` | 10 job levels and salaries | data |

## How to read the notebooks
- ✍️ **Core code: learn this.** marks the code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results

| Model | Train acc. | Test acc. |
|---|:-:|:-:|
| Unlimited tree (depth 12, 50 leaves) | 1.00 | 0.91 (overfits) |
| `max_depth=2` (two readable rules) | 0.91 | **0.94** |
| Tuned (`max_depth=4`, `min_samples_leaf=10`) | – | **0.94** |

Regression: an unlimited tree predicts **150,000** for level 6.5 (exactly level 6's salary).

## Learning objectives
- Explain how a tree chooses splits using Gini / entropy (classification) or squared error (regression)
- Read and explain a trained tree's rules
- Recognise overfitting and control it with `max_depth` / `min_samples_leaf`
- Know why trees need no scaling and cannot extrapolate

Sources: adapted from `Part 3 - Classification/Section 19` and `Part 2 - Regression/Section 8`.
