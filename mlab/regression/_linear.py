import math

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
        # Then calculate w = (Xᵀ·X)⁻¹·Xᵀ·y where X = U·Σ·Vᵀ, which becomes simply w = V · Σ⁺ · Uᵀ · y
        # To invert Σ, zero out nearly zero values (0.0001 ≈ 0) to avoid numerically instability -> pseudo-inverse Σ⁺.
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
    def __init__(
            self,
            learning_rate=0.01,
            epochs=1000,
            batch_size=32,
            tolerance=1e-4,
            lambda_=0.01,
    ):
        self.weights_ = None
        self.bias_ = None

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.lambda_ = lambda_
        self.rng = np.random.default_rng(seed=42)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]):
        """
        Train the model using stochastic gradient descent.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        if self.weights_ is None and self.bias_ is None:
            self.weights_ = np.zeros((X.shape[1], 1), dtype=X.dtype)
            self.bias_ = np.zeros((1, 1), dtype=X.dtype)

        epoch = 1
        prev_loss = None
        while self.epochs >= epoch:
            indices = self.rng.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            num_batches = math.ceil(X_shuffled.shape[0] / self.batch_size)
            batches = np.array_split(X_shuffled, num_batches)
            y_batches = np.array_split(y_shuffled, num_batches)

            epoch_loss = 0
            for X_batch, y_batch in zip(batches, y_batches):
                y_hat = self._predict(X_batch)
                # Make same shape as y_hat: (32,) -> (32,1)
                y_batch = y_batch.reshape(-1, 1)

                epoch_loss += self._mse(y_batch, y_hat)

                grad = self._mse_grad(y_batch, y_hat)
                # Apply L2 regularization - L = (1/n) * Σ(y_hat - y)² + λ * Σ(w²)
                # ∂L/∂w = (2/n) * Xᵀ(y_hat - y) + 2λw
                self.weights_ = self.weights_ - self.learning_rate * (
                        X_batch.T @ grad + 2 * self.lambda_ * self.weights_
                )
                self.bias_ = self.bias_ - self.learning_rate * np.mean(grad)

            print(f"Epoch {epoch}/{self.epochs} | Loss: {epoch_loss / num_batches:.4f}")
            epoch += 1

            # Early stopping logic
            if prev_loss is None:
                prev_loss = epoch_loss
                continue

            if abs(prev_loss - epoch_loss) / (prev_loss + 1e-8) < self.tolerance:
                print(f"Early stopped with default tolerance of {self.tolerance}")
                break

            prev_loss = epoch_loss

        return self

    def _predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Returns predictions in 2D shape (n_samples, 1) for internal use."""
        return X @ self.weights_ + self.bias_

    def predict(self, X: NDArray[np.floating]):
        """
        Predict target values for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return (X @ self.weights_ + self.bias_).flatten()

    def _mse(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    def _mse_grad(self, y, y_hat):
        return 2 * (y_hat - y) / len(y)
