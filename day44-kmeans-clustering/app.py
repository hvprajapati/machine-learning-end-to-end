"""
Run the from-scratch KMeans (kmeans.py) on the student dataset and compare it
with scikit-learn's KMeans.

    python app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans

from kmeans import KMeans

df = pd.read_csv('student_clustering.csv')
X = df[['cgpa', 'iq']].values

# --- our implementation ---
km = KMeans(n_clusters=4, max_iter=500, random_state=3)
y_means = km.fit_predict(X)
print(f"Scratch KMeans : converged in {km.n_iter_} iterations, inertia = {km.inertia_:.2f}")

# --- scikit-learn, for comparison ---
sk = SklearnKMeans(n_clusters=4, n_init=10, random_state=2).fit(X)
print(f"sklearn KMeans : inertia = {sk.inertia_:.2f}")
# Try other random_state values: some give a much higher inertia (a local minimum),
# which sklearn avoids with k-means++ init and n_init=10 restarts.

colors = ['red', 'blue', 'green', 'gold']
for k in range(4):
    plt.scatter(X[y_means == k, 0], X[y_means == k, 1], color=colors[k], label=f'Cluster {k}')
plt.scatter(km.centroids[:, 0], km.centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('K-Means from scratch (K = 4)')
plt.legend()
plt.show()
