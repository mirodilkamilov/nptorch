import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -15, 15)))


def cross_entropy_loss(y, y_hat, epsilon=1e-15):
    # Clip predicted y [-epsilon, epsilon] to prevent log(0) errors
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


class LogisticRegression:
    def __init__(
            self,
            learning_rate=0.01,
            epochs=1000,
            threshold=0.5,
            tolerance=None,
            lambda_=0.01,
    ):
        self.weights_ = None
        self.bias_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.tolerance = tolerance
        self.lambda_ = lambda_
        self.loss_history = []
        self.rng = np.random.default_rng(seed=42)

    def fit(self, X, y):
        """
        Train the model on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with binary labels (0 or 1)
        """
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        for epoch in range(1, self.epochs + 1):
            y_pred = self.predict_proba(X)
            self.loss_history.append(
                {"epoch": epoch, "loss": cross_entropy_loss(y, y_pred)}
            )

            # Apply L2 regularization once per epoch - L = (1/n) * Σ(y_hat - y)² + λ * Σ(w²)
            # ∂L/∂w = (2/n) * Xᵀ(y_hat - y) + 2λw
            weight_grad = (1 / n_samples) * X.T @ (y_pred - y) + (
                    self.lambda_ / n_samples
            ) * self.weights_
            bias_grad = np.mean(y_pred - y)  # (1 / n_samples) * np.sum(y_pred - y)

            self.weights_ -= self.learning_rate * weight_grad
            self.bias_ -= self.learning_rate * bias_grad

            if self.tolerance is not None and len(self.loss_history) > 1:
                prev_loss = self.loss_history[-2]["loss"]
                curr_loss = self.loss_history[-1]["loss"]
                if abs(prev_loss - curr_loss) / (prev_loss + 1e-8) < self.tolerance:
                    print(f"Early stopped with tolerance of {self.tolerance}")
                    break

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted labels (0 or 1)
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input. (Optional but recommended)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, 2) with probabilities for each class
        """
        return sigmoid(X @ self.weights_ + self.bias_)
