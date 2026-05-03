import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -15, 15)))


def cross_entropy_loss(y, y_pred, epsilon=1e-15):
    # Clip predicted y [-epsilon, epsilon] to prevent log(0) errors
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def confusion_matrix(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = np.mean(y == y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, accuracy, f1_score


class LogisticRegression:
    def __init__(
            self,
            learning_rate=0.01,
            epochs=1000,
            threshold=None,
            tolerance=1e-5,
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
        self.threshold_ = self.threshold
        self.rng = np.random.default_rng(seed=42)

    def fit(self, X, y):
        """
        Train the model on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with binary labels (0 or 1)
        """
        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain at least two classes.")

        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

        for epoch in range(1, self.epochs + 1):
            y_pred = self.predict_proba(X)

            precision, recall, accuracy, f1_score = confusion_matrix(y, y_pred)
            self.loss_history.append(
                {
                    "epoch": epoch,
                    "loss": cross_entropy_loss(y, y_pred),
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "f1": f1_score,
                }
            )

            # Apply L2 regularization once per epoch - L = (1/n) * Σ(y_pred - y)² + λ * Σ(w²)
            # ∂L/∂w = (2/n) * Xᵀ(y_pred - y) + 2λw
            weight_grad = (1 / n_samples) * X.T @ (y_pred - y) + (
                    self.lambda_ / n_samples
            ) * self.weights_
            bias_grad = np.mean(y_pred - y)  # (1 / n_samples) * np.sum(y_pred - y)

            self.weights_ -= self.learning_rate * weight_grad
            self.bias_ -= self.learning_rate * bias_grad

            if self.tolerance is not None and len(self.loss_history) > 1:
                prev_loss = self.loss_history[-2]["loss"]
                curr_loss = self.loss_history[-1]["loss"]
                if abs(prev_loss - curr_loss) < self.tolerance:
                    print(f"Early stopped with tolerance of {self.tolerance}")
                    break

        self.threshold_ = self.threshold if self.threshold is not None else np.mean(y)

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted labels (0 or 1)
        """
        return (self.predict_proba(X) >= self.threshold_).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input. (Optional but recommended)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, 2) with probabilities for each class
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return sigmoid(X @ self.weights_ + self.bias_)


class SGDClassifier:
    def __init__(
            self,
            learning_rate=0.01,
            epochs=1000,
            batch_size=32,
            threshold=None,
            tolerance=1e-5,
            lambda_=0.01,
    ):
        self.weights_ = None
        self.bias_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.tolerance = tolerance
        self.lambda_ = lambda_

        self.loss_history = []
        self.threshold_ = self.threshold
        self.rng = np.random.default_rng(seed=42)

    def fit(self, X, y):
        """
        Train the model using stochastic gradient descent.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with binary labels (0 or 1)
        """
        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain at least two classes.")

        self.threshold_ = self.threshold if self.threshold is not None else np.mean(y)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        prev_loss = None
        log_every = max(1, self.epochs // 100)  # log every 100th

        for epoch in range(1, self.epochs + 1):
            indices = self.rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                y_pred = self.predict_proba(X_batch)

                weight_grad = (1 / len(batch_idx)) * X_batch.T @ (y_pred - y_batch) + (
                        self.lambda_ / n_samples
                ) * self.weights_
                bias_grad = np.mean(y_pred - y_batch)

                self.weights_ -= self.learning_rate * weight_grad
                self.bias_ -= self.learning_rate * bias_grad

            if epoch % log_every == 0:
                y_full = self.predict_proba(X)
                curr_loss = cross_entropy_loss(y, y_full)
                precision, recall, accuracy, f1 = confusion_matrix(
                    y, (y_full >= self.threshold_).astype(int)
                )
                self.loss_history.append(
                    {
                        "epoch": epoch,
                        "loss": curr_loss,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "f1": f1,
                    }
                )

                if self.tolerance is not None and prev_loss is not None:
                    if abs(prev_loss - curr_loss) < self.tolerance:
                        print(
                            f"Early stopped at epoch {epoch} with tolerance {self.tolerance}"
                        )
                        break
                prev_loss = curr_loss

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted labels (0 or 1)
        """
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return (self.predict_proba(X) >= self.threshold_).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input. (Optional but recommended)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, 2) with probabilities for each class
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return sigmoid(X @ self.weights_ + self.bias_)
