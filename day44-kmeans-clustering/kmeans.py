"""
K-Means clustering from scratch (Lloyd's algorithm).

Written for learning, not speed: every step of the algorithm is a separate,
readable method so you can match the code line-by-line with theory.md.

    1. Initialise   -> pick K random data points as the starting centroids
    2. Assign       -> put every point in the cluster of its nearest centroid
    3. Update       -> move every centroid to the mean of the points it owns
    4. Repeat 2-3 until the centroids stop moving (or max_iter is reached)
"""
import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters      # K: how many clusters we want
        self.max_iter = max_iter          # safety cap on the number of loops
        self.tol = tol                    # "stopped moving" threshold for centroids
        self.random_state = random_state  # fix it to get the same result every run

        # learned during fit (trailing underscore = same convention as sklearn)
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.history_ = []                # centroids at every iteration, for plotting

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # Step 1: initialise - K distinct rows of X become the first centroids
        random_index = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[random_index].copy()
        self.history_ = [self.centroids.copy()]

        for i in range(self.max_iter):
            # Step 2: assign every point to its nearest centroid
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # Step 3: move each centroid to the mean of its points
            self.centroids = self.move_centroids(X, cluster_group, old_centroids)
            self.history_.append(self.centroids.copy())
            self.n_iter_ = i + 1
            # Step 4: stop once no centroid moved more than `tol`
            shift = np.linalg.norm(self.centroids - old_centroids, axis=1).max()
            if shift <= self.tol:
                break

        self.labels_ = self.assign_clusters(X)
        self.inertia_ = self.compute_inertia(X, self.labels_)
        return self.labels_

    def predict(self, X):
        """Assign new points to the nearest learned centroid (centroids do not move)."""
        return self.assign_clusters(np.asarray(X, dtype=float))

    def assign_clusters(self, X):
        # distances[i, j] = Euclidean distance between point i and centroid j
        # shape: (n_samples, 1, n_features) - (1, K, n_features) -> (n_samples, K)
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return distances.argmin(axis=1)

    def move_centroids(self, X, cluster_group, old_centroids):
        new_centroids = old_centroids.copy()

        for k in range(self.n_clusters):
            members = X[cluster_group == k]
            # an empty cluster has no mean - keep its old centroid instead of losing it
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)

        return new_centroids

    def compute_inertia(self, X, labels):
        # WCSS: sum of squared distances from each point to its own centroid
        return float(((X - self.centroids[labels]) ** 2).sum())
