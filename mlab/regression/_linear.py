import numpy as np
from numpy.typing import NDArray


class LinearRegressor:
    def __init__(self):
        self.weights_ = None

    @property
    def intercept_(self):
        return self.weights_[0]

    @property
    def coef_(self):
        return self.weights_[1:]

    def _add_dummy_feature(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        dummy_feature = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate((dummy_feature, X), axis=1)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]):
        """
        Train the model on the given data with closed-form solution.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data cannot be empty.")
        X_ = self._add_dummy_feature(X)

        # Computing matrix inverse is numerically unstable: self.weights_ = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        # Use instead Singular Value Decomposition approach - SVD splits any matrix X into three pieces X = U · Σ · Vᵀ
        # where V rotates the input space, Σ stretches/shrinks along each axis, U rotates the output space.
        # Rotation doesn't change the length of the vector, therefore no introduction of numerical error. Σ might introduce one.
        # Then calculate w = (Xᵀ·X)⁻¹·Xᵀ·y where X = U·Σ·Vᵀ, which becomes simply w = V · Σ⁺ · Uᵀ · y.
        # To invert Σ, zero out nearly zero values (0.0001 ≈ 0) -> pseudo-inverse Σ⁺.
        self.weights_, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
        return self

    def predict(self, X: NDArray[np.floating]):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        X_ = self._add_dummy_feature(X)
        return X_ @ self.weights_


class SGDRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        """
        Train the model using stochastic gradient descent.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        pass

    def predict(self, X):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        pass
