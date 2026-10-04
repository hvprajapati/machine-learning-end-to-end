# K-Means Clustering: Complete Theory

> Read this file first, then open the notebooks. Every idea here is used in code in
> `01-kmeans-sklearn.ipynb` or `02-kmeans-from-scratch.ipynb`.

---

## 1. Where K-Means fits

| | Supervised learning (Days 29–43) | Unsupervised learning (from today) |
|---|---|---|
| Data | Features **X** + target **y** | Features **X** only, no labels |
| Goal | Learn `X → y` | Find structure hidden inside **X** |
| Example | Predict if a student gets placed | Group students who are *similar* to each other |

**Clustering** means splitting the data into groups (clusters) so that:
- points **inside** a cluster are close to each other (high similarity), and
- points in **different** clusters are far apart (low similarity).

**K-Means** is the most widely used clustering algorithm. You choose **K**, the number of
clusters, and the algorithm finds **K centre points** (called **centroids**). Each data point
belongs to the cluster of its nearest centroid.

### Real-world uses
- **Customer segmentation**: group customers by spending and visit frequency.
- **Image compression / colour quantisation**: replace millions of colours with K representative colours.
- **Document grouping**: cluster news articles by topic (after converting text to vectors).
- **Anomaly detection**: points very far from every centroid are suspicious.
- **Feature engineering**: use the cluster label (or distances to centroids) as a new feature for a supervised model.

---

## 2. Key vocabulary

| Term | Meaning |
|---|---|
| **K** | Number of clusters. **You** choose it; the algorithm does not learn it. |
| **Centroid** $\mu_k$ | Centre of cluster $k$: the **mean** of all points in that cluster. It is usually *not* an actual data point. |
| **Assignment / label** | The cluster number (0 … K-1) given to a point. |
| **Euclidean distance** | Straight-line distance: $\lVert x - \mu \rVert = \sqrt{\sum_{j=1}^{d}(x_j - \mu_j)^2}$ |
| **Inertia / WCSS** | *Within-Cluster Sum of Squares*: the quantity K-Means minimises (Section 3). |
| **Convergence** | The state where assignments (and therefore centroids) stop changing. |

---

## 3. What K-Means is actually trying to do (the objective)

Given $n$ points $x_1, \dots, x_n$ and a chosen $K$, find clusters $C_1, \dots, C_K$ with centroids
$\mu_1, \dots, \mu_K$ that **minimise**:

$$
J \;=\; \text{WCSS} \;=\; \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2
$$

In words: *for every point, square its distance to its own centroid, then add everything up.*
Small $J$ means tight, compact clusters. In scikit-learn this value is `km.inertia_`.

**Why the centroid is the mean.** For a fixed group of points, which single point $\mu$ minimises
$\sum \lVert x_i - \mu \rVert^2$? Take the derivative and set it to zero:

$$
\frac{\partial}{\partial \mu}\sum_{i}\lVert x_i-\mu\rVert^2 = -2\sum_i (x_i-\mu) = 0
\;\;\Rightarrow\;\; \mu = \frac{1}{|C|}\sum_{i} x_i
$$

So the **mean** is the best centre for squared distances, and that is where the name
**K-*Means*** comes from.

---

## 4. The algorithm (Lloyd's algorithm), step by step

```
Input: data X (n rows, d features), number of clusters K

1. INITIALISE  Pick K starting centroids (e.g. K random data points).
2. ASSIGN      For every point, compute its distance to each of the K centroids
               and assign it to the NEAREST one.
3. UPDATE      For every cluster, move its centroid to the MEAN of the points
               currently assigned to it.
4. REPEAT      Go back to step 2 until the centroids stop moving
               (or a maximum number of iterations is reached).

Output: K centroids + a cluster label for every point
```

Both steps lower the objective $J$ (or leave it unchanged):
- **Assign step**: with centroids fixed, sending each point to its *nearest* centroid gives the smallest possible distance for that point.
- **Update step**: with assignments fixed, the *mean* is the best centre (Section 3).

$J$ never goes up, and there is only a finite number of ways to split $n$ points into $K$ groups,
so **K-Means always converges**, usually within 10–50 iterations.

> ⚠️ It converges to a **local** minimum, not necessarily the **global** best answer.
> The result depends on where the centroids started (Section 6).

---

## 5. Worked example by hand (K = 2)

Six points:

| Point | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| (x, y) | (2, 3) | (3, 3) | (6, 8) | (7, 8) | (3, 4) | (8, 7) |

Looking at them, {A, B, E} and {C, D, F} are clearly two groups. We deliberately pick a **bad**
start to watch K-Means fix it: $\mu_1 = A = (2,3)$, $\mu_2 = B = (3,3)$.

### Iteration 1
**Assign** (distances rounded to 2 decimals):

| Point | dist to μ₁ (2,3) | dist to μ₂ (3,3) | Cluster |
|---|---|---|---|
| A (2,3) | 0.00 | 1.00 | 1 |
| B (3,3) | 1.00 | 0.00 | 2 |
| C (6,8) | 6.40 | 5.83 | 2 |
| D (7,8) | 7.07 | 6.40 | 2 |
| E (3,4) | 1.41 | 1.00 | 2 |
| F (8,7) | 7.21 | 6.40 | 2 |

**Update**:
- $\mu_1$ = mean of {A} = **(2, 3)**
- $\mu_2$ = mean of {B, C, D, E, F} = $\left(\frac{3+6+7+3+8}{5}, \frac{3+8+8+4+7}{5}\right)$ = **(5.4, 6.0)**

WCSS drops from **117.0** → **43.2**.

### Iteration 2
**Assign**:

| Point | dist to μ₁ (2,3) | dist to μ₂ (5.4,6) | Cluster |
|---|---|---|---|
| A | 0.00 | 4.53 | 1 |
| B | 1.00 | 3.84 | 1 |
| C | 6.40 | 2.09 | 2 |
| D | 7.07 | 2.56 | 2 |
| E | 1.41 | 3.12 | 1 |
| F | 7.21 | 2.79 | 2 |

**Update**:
- $\mu_1$ = mean of {A, B, E} = (8/3, 10/3) ≈ **(2.67, 3.33)**
- $\mu_2$ = mean of {C, D, F} = (21/3, 23/3) ≈ **(7.00, 7.67)**

WCSS drops to **4.0**.

### Iteration 3
Re-assigning gives the same clusters {A, B, E} and {C, D, F}, so the centroids do not move.
**Converged.** Final WCSS = 4.0.

`02-kmeans-from-scratch.ipynb` runs this exact example with code and checks these numbers.

---

## 6. Initialisation matters: local minima and k-means++

Because K-Means only goes "downhill", a poor start can trap it in a bad solution.
Two centroids may start in the same real group, and one true group then gets split while two
others get merged. In the notebook, a single random start on the student data gives
WCSS ≈ 2270, while the best solution has WCSS ≈ 682.

Two standard fixes, both built into scikit-learn:

### a) k-means++ initialisation (`init='k-means++'`, the default)
Spread the starting centroids out instead of picking them uniformly at random:
1. Pick the first centroid uniformly at random from the data.
2. For every point compute $D(x)$, the distance to the **nearest centroid chosen so far**.
3. Pick the next centroid at random, with probability **proportional to $D(x)^2$**.
   Far-away points are much more likely to be chosen.
4. Repeat until K centroids are chosen, then run normal K-Means.

### b) Multiple restarts (`n_init`)
Run the whole algorithm `n_init` times from different starts and **keep the run with the lowest
inertia**. In scikit-learn ≥ 1.4, `n_init='auto'` means 1 run with k-means++ and 10 runs with
`init='random'`. Setting `n_init=10` explicitly is a safe choice for small datasets.

---

## 7. Choosing K

K is a **hyperparameter**. There is no "correct" K in the data; you choose it using
evidence plus domain knowledge.

### a) Elbow method
Run K-Means for K = 1, 2, …, 10 and plot WCSS against K.
- WCSS **always decreases** as K increases. With K = n, every point is its own cluster and WCSS = 0.
  **So never pick the K with the lowest WCSS.**
- Look for the **elbow**: the K after which adding more clusters gives only a small improvement.

The student data: WCSS drops 29958 → 4184 → 2364 → **682** → 514 → … The big drops stop after
K = 4, so the elbow is at **K = 4**.

Limitation: the elbow is often not sharp, and reading it is subjective.

### b) Silhouette score
For each point $i$:
- $a(i)$ = average distance from $i$ to the **other points in its own cluster** (cohesion)
- $b(i)$ = average distance from $i$ to the points of the **nearest other cluster** (separation)

$$
s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)} \qquad -1 \le s(i) \le 1
$$

| s(i) | Meaning |
|---|---|
| close to **+1** | well inside its own cluster, far from others |
| around **0** | on the border between two clusters |
| **negative** | probably in the wrong cluster |

The **silhouette score** is the mean of $s(i)$ over all points (`sklearn.metrics.silhouette_score`).
Try K = 2…10 and pick the K with the **highest** score. It needs at least 2 clusters.

### c) Domain knowledge
If the business wants 3 customer tiers (bronze / silver / gold), K = 3 may be the right answer
even when a metric slightly prefers another value.

---

## 8. Feature scaling is essential

K-Means is **distance-based**, so a feature with a large numeric range dominates the distance.

Student data: CGPA spans about 4.6–9.3 (range ≈ 5), IQ spans 83–121 (range ≈ 38).
A 5-point difference in IQ is small, but a 5-point difference in CGPA is the entire scale.
Without scaling, IQ decides almost every assignment.

The notebook shows this directly:

| | Best K by silhouette | Silhouette at K = 4 |
|---|---|---|
| Raw features | 2 (0.754) | 0.735 |
| Standardised features | **4 (0.861)** | **0.861** |

On raw data, the silhouette score even points to the wrong K.

**Rule:** apply `StandardScaler` (Day 7) or `MinMaxScaler` (Day 8) before K-Means, unless all
features are already in the same unit and scale.

---

## 9. Using a trained model on new data

K-Means learns centroids, so it can label points it has never seen:

```python
km.fit(X_train_scaled)               # learns cluster_centers_
km.predict(X_new_scaled)             # nearest-centroid assignment, centroids do NOT move
km.transform(X_new_scaled)           # distance from each point to every centroid (n × K)
```

Always pass new data through the **same fitted scaler** first (`scaler.transform`, not `fit_transform`).

---

## 10. scikit-learn `KMeans` cheat sheet

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42)
labels = km.fit_predict(X)
```

| Parameter | Default | What it controls |
|---|---|---|
| `n_clusters` | 8 | K, the number of clusters |
| `init` | `'k-means++'` | How starting centroids are chosen (`'random'` or an array is also allowed) |
| `n_init` | `'auto'` | Number of restarts; the best (lowest inertia) run is kept |
| `max_iter` | 300 | Maximum iterations per run |
| `tol` | 1e-4 | Convergence threshold on how much the centroids move |
| `random_state` | None | Fix it for reproducible results |
| `algorithm` | `'lloyd'` | `'elkan'` uses the triangle inequality to skip distance calculations |

| Attribute (after fit) | Meaning |
|---|---|
| `cluster_centers_` | Array (K × d) of centroid coordinates |
| `labels_` | Cluster label for every training point |
| `inertia_` | Final WCSS |
| `n_iter_` | Iterations used by the best run |

---

## 11. Assumptions and limitations

K-Means works best when clusters are **roughly spherical (round), of similar size and similar
density, and well separated**. It struggles when:

| Problem | Why | What to use instead |
|---|---|---|
| Non-spherical shapes (moons, rings, long chains) | Nearest-centroid boundaries are always straight lines (a Voronoi partition) | DBSCAN, Agglomerative (single linkage), Spectral clustering |
| Very different cluster sizes / densities | A big cluster gets split to reduce WCSS | Gaussian Mixture Models, DBSCAN |
| Outliers | The mean is pulled toward extreme points | K-Medoids, remove outliers first (Days 24–26) |
| Categorical features | "Mean" of categories is meaningless | K-Modes, K-Prototypes |
| Unknown K | K must be given up front | DBSCAN / HDBSCAN find the number of clusters themselves |
| Very large datasets | Every iteration touches every point | `MiniBatchKMeans` |

**Every point is always assigned** to some cluster. K-Means has no notion of "noise".

---

## 12. Complexity

Each iteration computes the distance from every point to every centroid:

$$
O(n \cdot K \cdot d) \text{ per iteration}, \qquad O(n \cdot K \cdot d \cdot t) \text{ in total}
$$

($n$ points, $K$ clusters, $d$ features, $t$ iterations). This is linear in $n$, so K-Means
scales to millions of rows, which is one reason it is so popular.

---

## 13. K-Means vs KNN (commonly confused)

| | **K-Means** | **K-Nearest Neighbours** |
|---|---|---|
| Type | Unsupervised (clustering) | Supervised (classification / regression) |
| Needs labels? | No | Yes |
| What is K? | Number of clusters | Number of neighbours to vote |
| Training | Iteratively learns K centroids | None: stores the training data |
| Output | Cluster id (meaningless number) | A real class / value |

Both are distance-based, so both need feature scaling.

---

## 14. Common mistakes

1. **Not scaling features.** The largest-range feature decides the clusters.
2. **Choosing K from the lowest inertia.** Inertia always falls as K grows; use the elbow or silhouette.
3. **Treating cluster numbers as meaningful.** "Cluster 0" is just a name; rerunning can swap labels. Interpret clusters by their centroids.
4. **Forgetting `random_state`.** Results change on every run, making notebooks irreproducible.
5. **Using K-Means on non-round clusters** and trusting the result without plotting it.
6. **Re-fitting the scaler on new data** before `predict` instead of reusing the fitted one.

---

## 15. Quick revision / interview questions

1. *What does K-Means minimise?* WCSS / inertia: the sum of squared distances of points to their own centroid.
2. *Why is it called K-"Means"?* Each centroid is the mean of its cluster, and the mean minimises squared distance.
3. *Does it always converge? To the global optimum?* It always converges, but only to a local optimum.
4. *How do you reduce the effect of bad initialisation?* k-means++ plus multiple restarts (`n_init`).
5. *How do you choose K?* Elbow method, silhouette score, domain knowledge.
6. *Why must features be scaled?* The algorithm is distance-based, so large-range features dominate.
7. *When does K-Means fail?* Non-spherical, unequal-size or unequal-density clusters, outliers, categorical data.
8. *Time complexity?* O(n·K·d·t), linear in the number of samples.

---

## Summary

- K-Means splits unlabeled data into **K** groups by minimising **WCSS**.
- It alternates **assign to nearest centroid** ↔ **move centroid to the mean** until nothing changes.
- It always converges, but to a **local** minimum, so use **k-means++** and **n_init** restarts.
- Pick K with the **elbow** method and the **silhouette score**, plus domain sense.
- **Scale your features** first; K-Means assumes round, similar-sized, well-separated clusters.
