# Day 44 — K-Means Clustering

The first **unsupervised learning** algorithm in this series: group unlabeled data into **K** clusters.

## Folder structure

| File | What it is | Order |
|---|---|---|
| [`theory.md`](theory.md) | Complete theory: objective, algorithm, worked example by hand, k-means++, choosing K, scaling, limitations, interview questions | **1. Read first** |
| [`01-kmeans-sklearn.ipynb`](01-kmeans-sklearn.ipynb) | K-Means with scikit-learn on student data (CGPA, IQ): elbow method, silhouette score, effect of scaling, predicting new points, 3-D example | **2** |
| [`02-kmeans-from-scratch.ipynb`](02-kmeans-from-scratch.ipynb) | Build K-Means yourself: verify the hand example, watch centroids move, see local minima, match sklearn | **3** |
| [`kmeans.py`](kmeans.py) | From-scratch `KMeans` class used by notebook 02 and `app.py` | reference |
| [`app.py`](app.py) | Script version: runs the scratch implementation, compares with sklearn, plots clusters | optional |
| `student_clustering.csv` | 200 students × 2 features (`cgpa`, `iq`), no labels | data |

## Learning objectives

After this day you should be able to:
- Explain what K-Means minimises (**WCSS / inertia**) and why the centroid is the **mean**
- Run the **assign → update** loop by hand on a small dataset
- Choose K with the **elbow method** and the **silhouette score**
- Explain why **feature scaling** is required, and show what goes wrong without it
- Explain **local minima** and how **k-means++** and `n_init` fix them
- Know when **not** to use K-Means (non-spherical clusters, outliers, categorical data)

## How to run

```bash
# from the repository root, with the project venv activated
cd day44-kmeans-clustering
jupyter notebook                # open 01, then 02
python app.py                   # optional script version
```

Requires: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `plotly` (3-D plots in notebook 01).

## Prerequisites from earlier days
- **Day 7 — Standardization** and **Day 8 — Normalization**: K-Means is distance-based, so features must be scaled.
- **Days 24–26 — Outliers**: outliers pull centroids away from the real cluster centres.
- **Day 28 — PCA**: often used before K-Means on high-dimensional data, or to visualise clusters in 2-D.
