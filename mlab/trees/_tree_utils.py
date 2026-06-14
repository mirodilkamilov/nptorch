import json

import numpy as np


class Node:
    def __repr__(self):
        return json.dumps(self._to_dict(), indent=2)

    def _to_dict(self):
        raise NotImplementedError


class InternalNode(Node):
    def __init__(
        self,
        feature: int,
        children: dict[float | int | bool, Node],
        threshold=None,     # None → categorical multi-way split; float → binary threshold split
        nan_goes_left=None, # for threshold nodes: True = NaN routes to True child, False = False child
    ):
        self.feature_ = feature
        self.children_ = children
        self.threshold_ = threshold
        self.nan_goes_left_ = nan_goes_left

    def _to_dict(self):
        base: dict = {"feature": int(self.feature_)}
        if self.threshold_ is not None:
            base["threshold"] = float(self.threshold_)
        base["children"] = {
            str(k): child._to_dict() for k, child in self.children_.items()
        }
        return base


class LeafNode(Node):
    def __init__(self, label: float | int):
        self.label_ = label

    def _to_dict(self):
        return {"label": int(self.label_)}


def entropy(target_column_arr):
    if len(target_column_arr) == 0:
        return 0.0
    _, counts = np.unique(target_column_arr, return_counts=True)
    proportions = counts / len(target_column_arr)
    return -np.sum(proportions * np.log2(proportions))


def information_gain(data, target_column_arr, split_feature_name_or_index):
    if len(data) == 0:
        return 0.0

    # Calculate total entropy of the current dataset
    total_entropy = entropy(target_column_arr)

    # Calculate weighted entropy after splitting on the feature
    feature_values, value_counts = np.unique(
        data[:, split_feature_name_or_index], return_counts=True
    )
    weighted_entropy = 0
    total_items = len(data)

    for i in range(len(feature_values)):
        value = feature_values[i]
        subset_mask = data[:, split_feature_name_or_index] == value

        subset_entropy = entropy(target_column_arr[subset_mask])
        proportion = value_counts[i] / total_items
        weighted_entropy += proportion * subset_entropy

    # Information Gain is the difference
    information_gain_ = total_entropy - weighted_entropy
    return information_gain_


def _best_threshold_split(feature_col, y):
    """Scan all midpoint thresholds for a continuous feature; return (best_ig, best_threshold).

    NaN values are excluded from the IG computation.
    """
    valid = ~np.isnan(feature_col)
    feature_col = feature_col[valid]
    y = y[valid]

    if len(feature_col) < 2:
        return -1.0, 0.0

    total_entropy = entropy(y)
    n = len(y)
    sorted_unique = np.sort(np.unique(feature_col))

    best_ig = -1.0
    best_threshold = sorted_unique[0]

    for i in range(len(sorted_unique) - 1):
        threshold = (sorted_unique[i] + sorted_unique[i + 1]) / 2
        left_mask = feature_col <= threshold
        n_left = left_mask.sum()
        n_right = n - n_left
        weighted_entropy = (
            (n_left / n) * entropy(y[left_mask]) +
            (n_right / n) * entropy(y[~left_mask])
        )
        ig = total_entropy - weighted_entropy
        if ig > best_ig:
            best_ig = ig
            best_threshold = threshold

    return best_ig, best_threshold


def _most_common_leaf_label(node):
    """Collect all leaf labels reachable from node and return the majority."""
    labels = []
    stack = list(node.children_.values())
    while stack:
        current = stack.pop()
        if isinstance(current, LeafNode):
            labels.append(current.label_)
        else:
            stack.extend(current.children_.values())
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def traverse(x, node):
    if isinstance(node, LeafNode):
        return node.label_

    if node.threshold_ is not None:
        val = x[node.feature_]
        if val != val:  # NaN: val != val is True only for NaN
            key = node.nan_goes_left_  # follow the same side NaN was routed to at training
        else:
            key = bool(val <= node.threshold_)
        return traverse(x, node.children_[key])

    feature_value = x[node.feature_]
    if feature_value not in node.children_:
        return _most_common_leaf_label(node)
    return traverse(x, node.children_[feature_value])


def build_tree(X, y, max_depth=None, min_samples_split=5, min_impurity_decrease=0.0):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples, got {len(X)} and {len(y)}."
        )
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")

    n_features = X.shape[1]
    features = set(range(n_features))
    importances = {i: 0.0 for i in range(n_features)}

    root = _build_tree(
        X,
        y,
        features,
        depth=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_impurity_decrease=min_impurity_decrease,
        n_total_samples=max(len(X), 1),
        importances=importances,
    )
    return root, importances


def _build_tree(
    X_subset,
    y_subset,
    features: set[int],
    depth,
    max_depth,
    min_samples_split,
    min_impurity_decrease,
    n_total_samples,
    importances,
):
    target_values, target_val_counts = np.unique(y_subset, return_counts=True)
    majority_label = target_values[np.argmax(target_val_counts)]

    if len(target_values) == 1:
        return LeafNode(target_values[0])

    if len(features) == 0:
        return LeafNode(majority_label)

    if max_depth is not None and depth >= max_depth:
        return LeafNode(majority_label)

    if len(X_subset) < min_samples_split:
        return LeafNode(majority_label)

    best_ig = -1.0
    best_feature = next(iter(features))
    best_threshold = None

    for feature_index in features:
        feature_col = X_subset[:, feature_index]
        if np.issubdtype(feature_col.dtype, np.floating):
            ig, threshold = _best_threshold_split(feature_col, y_subset)
        else:
            ig = information_gain(X_subset, y_subset, feature_index)
            threshold = None
        if ig > best_ig:
            best_ig = ig
            best_feature = feature_index
            best_threshold = threshold

    if best_ig < min_impurity_decrease:
        return LeafNode(majority_label)

    importances[best_feature] += (len(X_subset) / n_total_samples) * best_ig

    if best_threshold is not None:
        # Continuous: binary split; keep feature available in subtrees (CART style)
        feature_col = X_subset[:, best_feature]
        nan_mask = np.isnan(feature_col)
        left_mask = (feature_col <= best_threshold) & ~nan_mask
        right_mask = (feature_col > best_threshold) & ~nan_mask
        # Route NaN training samples to whichever side has more samples;
        # record the direction so traverse can follow the same path at prediction time.
        nan_goes_left = bool(left_mask.sum() >= right_mask.sum())
        if nan_mask.any():
            if nan_goes_left:
                left_mask = left_mask | nan_mask
            else:
                right_mask = right_mask | nan_mask
        root_node = InternalNode(best_feature, {}, threshold=best_threshold, nan_goes_left=nan_goes_left)
        for key, mask in [(True, left_mask), (False, right_mask)]:
            root_node.children_[key] = _build_tree(
                X_subset[mask],
                y_subset[mask],
                features=features,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_impurity_decrease=min_impurity_decrease,
                n_total_samples=n_total_samples,
                importances=importances,
            )
    else:
        # Categorical: multi-way split; remove feature so subtrees don't reuse it
        feature_col = X_subset[:, best_feature]
        root_node = InternalNode(best_feature, {})
        for value in np.unique(feature_col):
            mask = feature_col == value
            root_node.children_[value] = _build_tree(
                X_subset[mask],
                y_subset[mask],
                features=features - {best_feature},
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_impurity_decrease=min_impurity_decrease,
                n_total_samples=n_total_samples,
                importances=importances,
            )

    return root_node
