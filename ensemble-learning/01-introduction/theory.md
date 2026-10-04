# Introduction to Ensemble Learning: Theory

> Companion to `01-ensemble-intro.ipynb`. Follows the CampusX *Ensemble Learning* playlist (video 1, and the proof from video 2).
> Section numbers (§) are referenced from the notebook.

---

## 1. What is ensemble learning?

> **Ensemble learning = combining several machine-learning models into one bigger model that predicts better than
> any of them alone.**

The individual models are called **base models** (or base learners / estimators).

### The idea behind it: Wisdom of the Crowd
When many people independently give an answer, the **combined** answer is often better than almost every individual's.

| Real-life example | The crowd's answer |
|---|---|
| KBC *audience poll* | the most-voted option is almost always correct |
| A 4.5★ product rating from 15,000 buyers | far more trustworthy than one person's 4.5★ |
| IMDb ratings, elections | the average / majority of many opinions |
| Guessing an ox's weight at a fair | individual guesses are far off; their **average** is almost exact |

Why it works: individual errors point in **different directions**, so combining opinions cancels them out.

---

## 2. How an ensemble makes predictions

1. Train several base models.
2. For a new example, get a prediction from every model.
3. Combine them:

| Task | Combination rule |
|---|---|
| **Classification** | **majority vote** (the mode). Example: 5 models predict *placed, placed, not, placed, not* → **placed** (3 vs 2) |
| **Regression** | **average** (the mean). Example: models predict 4.0, 5.0 and 6.0 LPA → **5.0 LPA** |

### Diversity is the key
If all models are identical, they make the same mistakes and combining them gains nothing. A KBC audience made only of
software engineers is weaker than a mixed audience of engineers, doctors, lawyers and teachers.

Ways to make base models **different**:
1. **Different algorithms** on the same data (e.g. LR + SVM + DT)
2. The **same algorithm** on **different samples** of the data
3. Both

---

## 3. Why can an ensemble beat its best model? (the maths)

Three **independent** classifiers, each **70%** accurate, majority vote (the ensemble is right if ≥ 2 are right):

| Outcome | Probability |
|---|---|
| all 3 correct | $0.7^3 = 0.343$ |
| exactly 2 correct (3 ways) | $3 \times 0.7^2 \times 0.3 = 0.441$ |
| exactly 1 correct | $3 \times 0.7 \times 0.3^2 = 0.189$ |
| none correct | $0.3^3 = 0.027$ |

$$
P(\text{ensemble correct}) = 0.343 + 0.441 = \mathbf{0.784} > 0.7
$$

With models only **30%** accurate: $0.3^3 + 3 \times 0.3^2 \times 0.7 = 0.027 + 0.189 = \mathbf{0.216} < 0.3$, so
the ensemble is **worse**.

### Two conditions
1. **Each base model must be better than random guessing** (> 50% for two classes). This is usually easy.
2. **The models must be independent**: they must make **different** mistakes. In the notebook, 11 models at 70%
   reach ≈ 92% when independent, but stay at ≈ 70% when they all share the same mistakes.

### Geometric intuition
- **Classification:** each model's decision boundary has its own errors; the majority vote keeps only what most
  models agree on, giving a smoother, more reliable boundary.
- **Regression:** several fitted lines tilted in different directions average into a line closer to the truth.

---

## 4. The four types of ensemble

| Type | Base models | Training data | Combination | Example |
|---|---|---|---|---|
| **Voting** | **different** algorithms | the same full dataset | majority vote / average | LR + SVM + DT |
| **Stacking** | different algorithms | the same dataset | a **meta-model** is trained on the base models' outputs and learns how much to trust each | LR + SVM + DT → KNN on top |
| **Bagging** | the **same** algorithm | different **bootstrap samples** (rows drawn with replacement) | majority vote / average | Random Forest |
| **Boosting** | the same (weak) algorithm | models trained **in sequence**, each focusing on the previous one's mistakes | weighted vote / sum | AdaBoost, Gradient Boosting, XGBoost |

**Voting vs Stacking:** in voting every model has an equal say (like a democracy). Stacking learns **weights**, so a
better model gets a bigger say.

### Benefits
- **Better performance**: usually higher accuracy than any single model.
- **Lower bias *and* lower variance**: an ensemble can escape the usual bias–variance trade-off (e.g. bagging keeps
  deep trees' low bias while cutting their variance).
- **Robustness**: performance changes less when the data changes.

### Disadvantage
- **More computation**: many models to train and run, and the result is harder to interpret.

### When to use
At the end of a project, after cleaning, preprocessing, feature engineering and trying single models. On **tabular**
data, ensembles (especially bagging and boosting) are among the strongest methods available and dominate Kaggle
competitions.

---

## 5. Quick revision questions

1. *What is ensemble learning?* Combining several models into one that predicts better than each alone.
2. *How does an ensemble combine predictions?* Majority vote (classification), mean (regression).
3. *Two conditions for an ensemble to help?* Base models better than random, and independent / diverse.
4. *Three 70% independent classifiers: majority-vote accuracy?* 0.784.
5. *Name the four ensemble types and one key feature of each.* Voting (different algorithms), Bagging (same algorithm, bootstrap samples), Boosting (sequential, fixes previous mistakes), Stacking (meta-model combines the models).

---

## Summary

- **Ensemble = wisdom of the crowd** for models: combine diverse, better-than-random models.
- Classification combines by **majority vote**, regression by **averaging**.
- Four families: **Voting, Bagging, Boosting, Stacking**, each covered in its own folder of `ensemble-learning/`.
