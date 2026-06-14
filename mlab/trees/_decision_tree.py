import numpy as np

from mlab.trees._tree_utils import build_tree, traverse


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root_ = None
        self.feature_importances_ = None  # set after fit

    def fit(self, X, y):
        """
        Build the decision tree from training data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
        """
        root, importances = build_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        self.root_ = root

        raw = np.array([importances[i] for i in range(X.shape[1])], dtype=float)
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        if self.root_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        return np.array([traverse(x, self.root_) for x in X])
