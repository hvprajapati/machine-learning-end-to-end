# Decision Trees: Theory (Classification & Regression)

> Read this first, then open `01-decision-tree-classification.ipynb` and `02-decision-tree-regression.ipynb`.

---

## 1. The idea

A decision tree is a **flowchart of yes/no questions** learned from data. This is the actual tree learned in the
notebook (depth 2):

```
                 Is Age ≤ 44.5 ?
                /               \
             yes                 no
              |                   |
   Is Salary ≤ 90,500 ?        BUYS
       /          \
     yes           no
      |             |
  DOES NOT BUY     BUYS
```

To predict, start at the top and follow the answers until you reach a **leaf**:
- **Classification tree:** the leaf predicts the **majority class** of the training points in it.
- **Regression tree:** the leaf predicts the **average value** of the training points in it.

---

## 2. Key words

| Term | Meaning |
|---|---|
| **Root node** | the first question (top) |
| **Internal node** | a question further down |
| **Leaf** | an end point that gives the prediction |
| **Split** | a question of the form *feature ≤ threshold* |
| **Depth** | the number of questions from the root to the deepest leaf |
| **Impurity** | how mixed the classes in a node are (0 = all one class) |

---

## 3. How does the tree choose a question?

At every node, the tree tries **every feature** and **every possible threshold**, and picks the split that makes the
two child groups as **pure** as possible (each group dominated by one class). It then repeats the process inside each
child. This is a **greedy** algorithm: it picks the best split *right now* without looking ahead.

### Measuring impurity (classification)

With $p_k$ = the fraction of class $k$ in a node:

| Measure | Formula | Pure node | 50/50 node (2 classes) |
|---|---|:-:|:-:|
| **Gini** (sklearn default) | $1 - \sum_k p_k^2$ | 0 | 0.5 |
| **Entropy** | $-\sum_k p_k \log_2 p_k$ | 0 | 1.0 |

Both give almost the same trees in practice. Gini is slightly faster to compute.

### Worked example

A node has **10 points: 6 of class A, 4 of class B**.

$$
\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - 0.52 = 0.48
$$

Two candidate splits:

| Split | Left child | Right child | Weighted Gini after split |
|---|---|---|---|
| **X** | 5 points: 5 A, 0 B (Gini 0) | 5 points: 1 A, 4 B (Gini 0.32) | $0.5 \times 0 + 0.5 \times 0.32 = \mathbf{0.16}$ |
| **Y** | 6 points: 4 A, 2 B (Gini 0.444) | 4 points: 2 A, 2 B (Gini 0.5) | $0.6 \times 0.444 + 0.4 \times 0.5 = 0.467$ |

Gini for the right child of X: $1 - (0.2^2 + 0.8^2) = 0.32$.

- Split X reduces impurity from 0.48 to **0.16** (gain 0.32).
- Split Y only reduces it to 0.467 (gain 0.013).

**→ The tree chooses split X.** The weights are each child's share of the points.

With entropy the decision is the same: the root has entropy 0.971; after split X it is
$0.5 \times 0 + 0.5 \times 0.722 = 0.361$, an **information gain** of 0.610.

---

## 4. When does the tree stop?

By default a tree keeps splitting until every leaf is pure (or cannot be split). On real data that means
**memorising** the training set, i.e. **overfitting**.

From the notebook:

| Tree | Depth | Leaves | Train accuracy | Test accuracy |
|---|:-:|:-:|:-:|:-:|
| No limits | 12 | 50 | **1.00** | 0.91 |
| `max_depth=2` | 2 | 4 | 0.91 | **0.94** |
| Tuned by cross-validation (`max_depth=4`, `min_samples_leaf=10`) | 4 | – | – | **0.94** |

### Controlling the tree (pre-pruning)

| Parameter | Effect |
|---|---|
| `max_depth` | maximum number of questions in a row |
| `min_samples_split` | a node needs at least this many points to be split |
| `min_samples_leaf` | every leaf must keep at least this many points |
| `max_leaf_nodes` | maximum number of leaves |
| `ccp_alpha` | *post-pruning*: grow the full tree, then cut back branches that add little |

**Choose them with cross-validation** (`GridSearchCV`).

---

## 5. Properties of tree decision boundaries

- Every split is *one feature ≤ a threshold*, a **vertical or horizontal cut**.
- So the regions are always **rectangles** (boxes in higher dimensions).
- Diagonal boundaries need many small "staircase" cuts.

---

## 6. Regression trees

Same idea, two differences:

| | Classification tree | Regression tree |
|---|---|---|
| Split criterion | Gini / entropy (purity) | **squared error (MSE)**: make each group's values as close to its average as possible |
| Leaf prediction | majority class | **average** of the target values in the leaf |
| Prediction curve | rectangles of classes | a **staircase** of flat steps |

From the notebook (Position Salaries): an unlimited tree predicts **150,000** for level 6.5, the salary of level 6,
because every level gets its own leaf.

**Trees cannot extrapolate.** Beyond the training range they keep predicting the last leaf's value: levels 10, 12
and 20 all get 1,000,000.

---

## 7. Feature scaling is NOT needed

A split only compares one feature with a threshold. Multiplying a feature by 1,000 just multiplies its threshold by
1,000; the tree and its predictions stay identical (verified in the notebook). No `StandardScaler` needed, and the
rules stay in real units.

---

## 8. Feature importance

`model.feature_importances_` measures how much each feature reduced impurity across all its splits, normalised to
sum to 1. In the notebook: Age ≈ 0.55, Salary ≈ 0.45.

---

## 9. Using decision trees in scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=10, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)                       # class shares in the leaf
print(export_text(clf, feature_names=['Age', 'EstimatedSalary']))
plot_tree(clf, filled=True)

reg = DecisionTreeRegressor(max_depth=3, random_state=0)
reg.fit(X, y)
```

| Parameter | Default | Meaning |
|---|---|---|
| `criterion` | `'gini'` / `'squared_error'` | split quality measure (`'entropy'` also available for classification) |
| `max_depth` | None (unlimited) | maximum depth |
| `min_samples_leaf` | 1 | minimum points per leaf |
| `random_state` | None | fix for reproducible trees (ties between equally good splits) |

---

## 10. Strengths and limitations

| Strengths | Limitations |
|---|---|
| **Easy to understand and explain**: the rules can be printed | **Overfits** easily if not limited |
| No feature scaling needed | **Unstable**: small data changes can produce a very different tree |
| Handles non-linear relationships and interactions | Boundaries are boxy (axis-parallel cuts) |
| Works for classification and regression | Regression: staircase predictions, **no extrapolation** |
| Fast to train and predict | A single tree is usually less accurate than an ensemble |

The instability and overfitting are exactly what **Random Forest** (`ensemble-learning/04-random-forest/`) fixes.

---

## 11. Quick revision questions

1. *How does a tree choose a split?* It tries every feature and threshold and picks the one that most reduces impurity (Gini / entropy, or MSE for regression).
2. *What does a leaf predict?* Majority class (classification) or the average value (regression).
3. *Gini of a node with 50% / 50% classes?* 0.5. *Of a pure node?* 0.
4. *Why do unlimited trees overfit?* They keep splitting until every training point is classified perfectly, including noise.
5. *How do you control overfitting?* `max_depth`, `min_samples_leaf`, `min_samples_split`, or pruning, tuned by cross-validation.
6. *Do trees need scaling?* No.
7. *Can a regression tree predict a value higher than any training value?* No; it cannot extrapolate.

---

## Summary

- A decision tree is a learned **flowchart of feature ≤ threshold questions**.
- Splits are chosen greedily to maximise **purity** (Gini / entropy) or minimise **squared error** (regression).
- Easy to read, no scaling needed, but **overfits** unless its growth is limited.
- Classification boundaries are **rectangles**; regression predictions are **staircases**.
