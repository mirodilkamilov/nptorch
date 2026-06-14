import numpy as np

from mlab.trees._tree_utils import build_tree, LeafNode, _most_common_leaf_label


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root_ = None
        self.feature_importances_ = None
        self._label_dtype = None

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
        self._label_dtype = y.dtype

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

        n = len(X)
        result = np.empty(n, dtype=object)
        stack = [(np.arange(n), self.root_)]

        while stack:
            indices, node = stack.pop()
            if not len(indices):
                continue

            if isinstance(node, LeafNode):
                result[indices] = node.label_
                continue

            if node.threshold_ is not None:
                vals = X[indices, node.feature_]
                nan_mask = np.isnan(vals)
                left_mask = (vals <= node.threshold_) | (
                    nan_mask & bool(node.nan_goes_left_)
                )
                stack.append((indices[left_mask], node.children_[True]))
                stack.append((indices[~left_mask], node.children_[False]))
            else:
                feature_vals = X[indices, node.feature_]
                routed = np.zeros(len(indices), dtype=bool)
                for child_val, child_node in node.children_.items():
                    mask = feature_vals == child_val
                    if mask.any():
                        stack.append((indices[mask], child_node))
                        routed |= mask
                if not routed.all():
                    result[indices[~routed]] = _most_common_leaf_label(node)

        return result.astype(self._label_dtype)
