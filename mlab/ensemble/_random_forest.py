import numpy as np

from mlab.trees._decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators=20,
        max_depth=5,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None

    def _resolve_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features  # None → use all features (plain DT behaviour)

    def fit(self, X, y):
        """
        Build the random forest from training data using bootstrap sampling.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        max_features = self._resolve_max_features(n_features)

        self.trees_ = []
        importances_sum = np.zeros(n_features)

        for _ in range(self.n_estimators):
            indices = rng.integers(0, n_samples, size=n_samples)
            X_boot, y_boot = X[indices], y[indices]

            tree_seed = int(rng.integers(0, 2**31))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=max_features,
                random_state=tree_seed,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            importances_sum += tree.feature_importances_

        self.feature_importances_ = importances_sum / self.n_estimators
        return self

    def predict(self, X):
        """
        Predict class labels using majority voting across all trees.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        if not self.trees_:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        # (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        n_samples = all_preds.shape[1]
        result = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            result[i] = vals[np.argmax(counts)]
        return result
