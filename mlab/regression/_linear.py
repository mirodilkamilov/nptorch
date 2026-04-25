import numpy as np
from numpy.typing import NDArray


class LinearRegressor:
    def __init__(self):
        self.weights_ = None

    def _add_dummy_feature(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        dummy_feature = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.concatenate((dummy_feature, X), axis=1)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]):
        """
        Train the model on the given data with closed-form solution.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        """
        X_ = self._add_dummy_feature(X)
        self.weights_ = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        return self

    def predict(self, X):
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

# def main() -> None:
#     X = np.array(
#         [
#             [4.0, 100],
#             [5.0, 110],
#             [5.5, 105],
#             [6.0, 120],
#             [6.5, 115],
#             [7.0, 125],
#             [7.5, 130],
#             [8.0, 135],
#             [8.5, 130],
#             [9.0, 140],
#         ],
#         dtype=np.float64,
#     )
#     y = np.array([10, 12, 13, 15, 16, 18, 19, 21, 22, 24], dtype=np.float64)
#
#     model = LinearRegressor()
#     model.fit(X, y)
#     print(model.weights_)
#
#     prediction = model.predict(np.array([[7.3, 130]]))
#     print(prediction)
#
#
# if __name__ == "__main__":
#     main()
