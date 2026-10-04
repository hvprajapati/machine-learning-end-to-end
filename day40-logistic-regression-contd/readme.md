# Day 40 — Logistic Regression (continued)

More than two classes (softmax) and curved decision boundaries (polynomial features).

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | [`01-softmax-regression.ipynb`](01-softmax-regression.ipynb) | Iris with 3 classes: `LogisticRegression(multi_class='multinomial')`, accuracy, confusion matrix, `predict_proba`, decision regions |
| 2 | [`02-polynomial-logistic-regression.ipynb`](02-polynomial-logistic-regression.ipynb) | U-shaped data: plain logistic regression (a straight-line boundary), `PolynomialFeatures` + logistic regression, decision boundary for different degrees |

Extra tool: `streamlit-viz-tool.py`, an interactive Streamlit app to play with logistic regression settings (`streamlit run streamlit-viz-tool.py`).  
Note: notebook 2 reads `ushape.csv`, which is not in this folder.  
Note: both notebook 1 and the Streamlit app pass `multi_class=...` to `LogisticRegression`; that parameter no longer exists in recent scikit-learn (it is gone in 1.9). Just use `LogisticRegression()`: for 3+ classes it is already multinomial (softmax).
