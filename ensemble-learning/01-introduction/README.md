# 01 — Introduction to Ensemble Learning

Combine several models into one that predicts better than any of them alone: the **wisdom of the crowd** for machine learning.
(CampusX *Ensemble Learning* video 1, plus the proof from video 2.)

| File | What it covers | Order |
|---|---|---|
| [`theory.md`](theory.md) | Wisdom of the crowd, vote vs average, the 0.7 → 0.784 proof, the two conditions (better than random + independent), the 4 ensemble types, benefits and costs | **1. Read first** |
| [`01-ensemble-intro.ipynb`](01-ensemble-intro.ipynb) | Ox-weight simulation, the proof verified by simulation, accuracy vs number of models, why diversity matters, vote / average pictures, the 4 types | **2** |

## How to read the notebook
- ✍️ **Core code: learn this.** marks code you should understand and be able to write.
- 🎨 **Visualization only.** marks plotting code you do *not* need to learn (collapsed; just read the plot).
- 🧠 **Quick check** questions at the end: click to reveal the answers.

## Key results
| Experiment | Result |
|---|---|
| 800 noisy guesses of a 543 kg ox | individuals off by ≈ 64 kg; the average is 540.9 kg |
| 3 independent 70%-accurate models, majority vote | **0.784** (simulation and maths agree) |
| 11 models at 70%: independent vs sharing all mistakes | ≈ **0.92** vs ≈ 0.70 |

## Learning objectives
- Explain the wisdom of the crowd and how ensembles combine predictions
- Prove why majority voting can beat each member, and state the two conditions
- Name the four ensemble families: Voting, Bagging, Boosting, Stacking

➡️ Next: [`../02-voting/`](../02-voting/)
