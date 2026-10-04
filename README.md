# Machine Learning: End to End

A step-by-step machine learning course in Jupyter notebooks, from loading data to ensemble models
(following the CampusX *100 Days of Machine Learning* playlist).

## How every folder is organised

- Each topic has its own folder. Open its **README** first: it lists the notebooks in **teaching order**.
- Notebooks are numbered `01-…`, `02-…`: teach them in that order.
- Notebooks with **`EXTRA`** in the name (e.g. `02-EXTRA-knn-from-scratch.ipynb`) implement an algorithm **from scratch**.
  They are optional, just for extra understanding: teach the core notebooks first.
- Newer folders also have a `theory.md` with the theory, worked examples and revision questions.
- Data files (`.csv`) sit next to the notebooks that use them.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter notebook
```

## Course map

### 1. Working with data
| Day | Folder | Topic |
|:-:|---|---|
| 01 | [`day01 - working with csv files`](day01%20-%20working%20with%20csv%20files/) | Reading CSV files with pandas |
| 02 | [`day02 - working-with-json-and-sql`](day02%20-%20working-with-json-and-sql/) | JSON and SQL |
| 03 | [`day03-api-to-dataframe`](day03-api-to-dataframe/) | Fetching data from an API |
| 04 | [`day04-understanding-your-data-descriptive-stats`](day04-understanding-your-data-descriptive-stats/) | First look at a dataset |
| 05 | [`day05-univariate-analysis`](day05-univariate-analysis/) | Univariate analysis (EDA) |
| 06 | [`day06-bivariate-analysis`](day06-bivariate-analysis/) | Bivariate analysis (EDA) |

### 2. Feature engineering
| Day | Folder | Topic |
|:-:|---|---|
| 07 | [`day07-standardization`](day07-standardization/) | Standardization |
| 08 | [`day08-normalization`](day08-normalization/) | Normalization |
| 09 | [`day09-ordinal-encoding`](day09-ordinal-encoding/) | Ordinal encoding |
| 10 | [`day10-one-hot-encoding`](day10-one-hot-encoding/) | One-hot encoding |
| 11 | [`day11-column-transformer`](day11-column-transformer/) | Column transformer |
| 12 | [`day12-sklearn-pipelines`](day12-sklearn-pipelines/) | Scikit-learn pipelines |
| 13 | [`day13-function-transformer`](day13-function-transformer/) | Function transformer |
| 15 | [`day15-binning-and-binarization`](day15-binning-and-binarization/) | Binning and binarization |
| 16 | [`day16-handling-mixed-variables`](day16-handling-mixed-variables/) | Mixed variables |
| 17 | [`day17-handling-date-and-time`](day17-handling-date-and-time/) | Dates and times |

### 3. Missing values and outliers
| Day | Folder | Topic |
|:-:|---|---|
| 18 | [`day18-complete-case-analysis`](day18-complete-case-analysis/) | Complete case analysis |
| 19 | [`day19-imputing-numerical-data`](day19-imputing-numerical-data/) | Mean / median / arbitrary imputation |
| 20 | [`day20-handling-missing-categorical-data`](day20-handling-missing-categorical-data/) | Most frequent / missing category |
| 21 | [`day21-missing-indicator`](day21-missing-indicator/) | Random sample, missing indicator, auto-tuning imputers |
| 22 | [`day22-knn-imputer`](day22-knn-imputer/) | KNN imputer |
| 23 | [`day23-iterative-imputer`](day23-iterative-imputer/) | Iterative imputer (MICE) |
| 24 | [`day24-outlier-removal-using-zscore`](day24-outlier-removal-using-zscore/) | Outliers: z-score |
| 25 | [`day25-outlier-removal-using-iqr-method`](day25-outlier-removal-using-iqr-method/) | Outliers: IQR |
| 26 | [`day26-outlier-detection-using-percentiles`](day26-outlier-detection-using-percentiles/) | Outliers: percentiles |
| 27 | [`day27-feature-construction-and-feature-splitting`](day27-feature-construction-and-feature-splitting/) | Feature construction & splitting |
| 28 | [`day28-pca`](day28-pca/) | PCA |

### 4. Regression
| Day | Folder | Topic |
|:-:|---|---|
| 29 | [`day29-simple-linear-regression`](day29-simple-linear-regression/) | Simple linear regression |
| 30 | [`day30-regression-metrics`](day30-regression-metrics/) | Regression metrics |
| 31 | [`day31-multiple-linear-regression`](day31-multiple-linear-regression/) | Multiple linear regression |
| 32 | [`day32-gradient-descent`](day32-gradient-descent/) | Gradient descent |
| 33 | [`day33-types-of-gradient-descent`](day33-types-of-gradient-descent/) | Batch / stochastic / mini-batch GD |
| 34 | [`day34-polynomial-regression`](day34-polynomial-regression/) | Polynomial regression |
| 35 | [`day35-regularized-linear-models`](day35-regularized-linear-models/) | Ridge regression |
| 36 | [`day36-lasso-regression`](day36-lasso-regression/) | Lasso regression |
| 37 | [`day37-elasticnet-regression`](day37-elasticnet-regression/) | Elastic Net |

### 5. Classification
| Day | Folder | Topic |
|:-:|---|---|
| 38 | [`day38-logistic-regression`](day38-logistic-regression/) | Logistic regression |
| 39 | [`day39-classification-metrics`](day39-classification-metrics/) | Classification metrics |
| 40 | [`day40-logistic-regression-contd`](day40-logistic-regression-contd/) | Softmax & polynomial logistic regression |
| 45 | [`day45-knn`](day45-knn/) | K-nearest neighbours |
| 46 | [`day46-svm`](day46-svm/) | SVM and SVR |
| 47 | [`day47-naive-bayes`](day47-naive-bayes/) | Naive Bayes |
| 48 | [`day48-decision-tree`](day48-decision-tree/) | Decision trees |

### 6. Clustering
| Day | Folder | Topic |
|:-:|---|---|
| 44 | [`day44-kmeans-clustering`](day44-kmeans-clustering/) | K-Means |

### 7. Ensemble learning
[`ensemble-learning/`](ensemble-learning/): introduction, voting, bagging, random forest, AdaBoost, gradient boosting,
XGBoost, stacking & blending. Start with its README.
