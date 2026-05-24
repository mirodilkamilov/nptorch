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


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """
        Train the model by computing class priors and feature likelihoods.

        Args:
            X: numpy array of shape (n_samples, n_features) - non-negative count features
            y: numpy array of shape (n_samples,) - class labels
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Training data cannot be empty.")

        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        # One-hot encode y: Y[i, k] = 1 if sample i belongs to class k
        # Shape: (n_samples, n_classes)
        Y = (y[:, np.newaxis] == self.classes_[np.newaxis, :]).astype(X.dtype)

        # log P(C_k) = log(count(C_k) / n_samples)
        class_counts = Y.sum(axis=0)                               # (n_classes,)
        self.class_log_prior_ = np.log(class_counts / n_samples)  # (n_classes,)

        # Feature counts per class via matrix multiply — no per-class loop:
        #   feature_counts[k, i] = Σ_{samples in class k} X[:, i]
        # Shape: (n_classes, n_features)
        # Laplace smoothing: p_ki = (count_ki + α) / (Σ_i count_ki + α * n_features)
        feature_counts = Y.T @ X                                                    # (n_classes, n_features)
        smoothed_counts = feature_counts + self.alpha
        smoothed_totals = smoothed_counts.sum(axis=1, keepdims=True)               # (n_classes, 1)
        self.feature_log_prob_ = np.log(smoothed_counts / smoothed_totals)         # (n_classes, n_features)

        return self

    def _log_posterior(self, X):
        """
        Compute unnormalised log posterior: log P(C_k) + Σ x_i * log P(feat_i | C_k).

        The Σ x_i * log p_ki term is a dot product: X @ feature_log_prob_.T

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return self.class_log_prior_ + X @ self.feature_log_prob_.T

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        log_post = self._log_posterior(X)

        # Log-sum-exp trick for numerical stability
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        return proba / proba.sum(axis=1, keepdims=True)
