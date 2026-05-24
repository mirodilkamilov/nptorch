import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None  # unique class labels, shape (n_classes,)
        self.priors_ = None  # prior probability per class, shape (n_classes,)
        self.mean_ = None  # mean per class/feature, shape (n_classes, n_features)
        self.variance_ = None  # smoothed variance per class/feature, shape (n_classes, n_features)

    def fit(self, X, y):
        """
        Train the model by computing class priors, means, and variances.

        Args:
            X: numpy array of shape (n_samples, n_features) - continuous features.
            y: numpy array of shape (n_samples,) - class labels
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data cannot be empty.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.mean_ = np.zeros((n_classes, n_features))
        self.variance_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = X_c.shape[0] / X.shape[0]
            self.mean_[i] = X_c.mean(axis=0)
            self.variance_[i] = X_c.var(axis=0)

        # Add smoothing: a fraction of the largest per-feature variance seen in the
        # full dataset. Keeps epsilon data-scale-relative so the default 1e-9 is
        # meaningful whether features are in [0,1] or [0,1000].
        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.variance_ += epsilon

        return self

    def _log_likelihood(self, X):
        """
        Compute log P(X | C_k) for every class under the Gaussian assumption.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        # X[:, np.newaxis, :]  → (n_samples, 1,        n_features)
        # self.mean_           → (n_classes, n_features) broadcast to same shape
        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        for i in range(len(self.classes_)):
            mu = self.mean_[i]  # (n_features,)
            var = self.variance_[i]  # (n_features,)

            # log N(x; μ, σ²) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]
            log_norm = -0.5 * np.log(2 * np.pi * var)
            log_exp = -0.5 * ((X - mu) ** 2) / var
            log_likelihoods[:, i] = (log_norm + log_exp).sum(axis=1)

        return log_likelihoods  # (n_samples, n_classes)

    def _log_posterior(self, X):
        """
        Compute unnormalised log posterior: log P(C_k) + log P(X | C_k).

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        return np.log(self.priors_) + self._log_likelihood(X)

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        if self.priors_ is None or self.mean_ is None or self.variance_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        log_post = self._log_posterior(X)  # (n_samples, n_classes)

        # Subtract row-wise max before exp for numerical stability
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        return proba / proba.sum(axis=1, keepdims=True)
