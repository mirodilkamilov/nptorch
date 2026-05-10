import numpy as np
from numpy.typing import NDArray
from numpy.random import RandomState


def euclidean_distance(point1: NDArray[np.floating], point2: NDArray[np.floating]):
    if point1.shape != point2.shape:
        raise ValueError("Two points must have same dimension.")

    return np.sqrt(np.sum(np.square(point1 - point2)))


class KMeans:
    def __init__(
            self,
            n_clusters=3,
            max_iter=300,
            tol=1e-4,
            random_state: int | RandomState | None = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(seed=random_state)
        )
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _random_centroid_initialization(self, X: NDArray[np.floating]):
        n_samples = X.shape[0]
        init_indices = self.random_state.randint(
            low=0, high=n_samples, size=self.n_clusters
        )
        self.cluster_centers_ = X[init_indices]

        return self

    def _assign_labels(self, X: NDArray[np.floating]):
        labels = np.zeros((X.shape[0],), dtype=np.int16)

        for instance_idx, instance in enumerate(X):
            closest_distance = np.inf
            closest_label = 0

            for label, centroid in enumerate(self.cluster_centers_):
                distance = euclidean_distance(centroid, instance)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_label = label
            labels[instance_idx] = closest_label

        return labels

    def _update_centroids(self, X: NDArray[np.floating], labels: NDArray[np.floating]):
        for label in range(self.n_clusters):
            self.cluster_centers_[label] = X[labels == label].mean(axis=0)

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            self
        """
        labels = None

        # Random centroid initialization
        self._random_centroid_initialization(X)

        for epoch in range(1, self.max_iter + 1):
            # Assign point to a cluster
            labels = self._assign_labels(X)
            prev_centers = self.cluster_centers_.copy()

            # Update centroids
            self._update_centroids(X, labels)

            if np.linalg.norm(self.cluster_centers_ - prev_centers) < self.tol:
                break

        self.labels_ = labels
        return self

    def predict(self, X):
        """
        Predict cluster labels for new data points.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with cluster labels (integers 0 to n_clusters-1)
        """
        return self._assign_labels(X)
