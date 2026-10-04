# Day 21 — Random Sample Imputation, Missing Indicator, Auto-tuning Imputers

More ways to handle missing values.

## Teaching order

| # | Notebook | What it covers |
|:-:|---|---|
| 1 | [`01-random-sample-imputation.ipynb`](01-random-sample-imputation.ipynb) | Fill missing values with random samples from the column, keeping its distribution |
| 2 | [`02-missing-indicator.ipynb`](02-missing-indicator.ipynb) | Add a 0/1 "was missing" column (`MissingIndicator`, `add_indicator=True`) and compare model accuracy |
| 3 | [`03-automatically-select-imputer-parameters.ipynb`](03-automatically-select-imputer-parameters.ipynb) | Let `GridSearchCV` choose the imputation strategy inside a pipeline |

Data: `house-train.csv`, `train.csv`
