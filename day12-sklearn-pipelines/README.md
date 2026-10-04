# Day 12 — Scikit-learn Pipelines

Chain preprocessing and a model into one object, then use it to predict.

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | [`01-titanic-without-pipeline.ipynb`](01-titanic-without-pipeline.ipynb) | Titanic: imputation, one-hot encoding and a decision tree done step by step by hand; saves the pieces to `models/` |
| 2 | [`02-predict-without-pipeline.ipynb`](02-predict-without-pipeline.ipynb) | Predict for a new passenger by loading every saved piece from `model/` and applying them in order |
| 3 | [`03-titanic-using-pipeline.ipynb`](03-titanic-using-pipeline.ipynb) | The same work as one `Pipeline`: `ColumnTransformer`s, scaling, `SelectKBest`, model, `make_pipeline`, cross-validation, grid search, export to `pipe.pkl` |
| 4 | [`04-predict-using-pipeline.ipynb`](04-predict-using-pipeline.ipynb) | Predict for a new passenger with a single `pipe.predict` call |

Saved files: `pipe.pkl`, `model/` (read by notebook 02) and `models/` (written by notebook 01).

Data: `train.csv`
