import numpy as np

from ._svm_utils import compute_kernel


class SVM:
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1, degree=3):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree

    def _compute_kernel(self, X, Z):
        return compute_kernel(X, Z, kernel=self.kernel, degree=self.degree)

    def decision_function(self, X):
        """
        Compute raw decision scores.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with decision scores
        """
        if not hasattr(self, "support_vectors_"):
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
        if X.shape[1] != self.support_vectors_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.support_vectors_.shape[1]} features."
            )

        K = self._compute_kernel(X, self.support_vectors_)
        return K @ self.dual_coef_ + self.b_

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
