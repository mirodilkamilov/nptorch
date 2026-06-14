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
        children: dict[
            float | int, Node
        ],  # key represents edge - feature value, value represent the actual child node
    ):
        self.feature_ = feature
        self.children_ = children

    def _to_dict(self):
        return {
            "feature": int(self.feature_),
            "children": {
                str(value): child._to_dict() for value, child in self.children_.items()
            },
        }


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

    feature_value = x[node.feature_]

    if feature_value not in node.children_:
        return _most_common_leaf_label(node)

    return traverse(x, node.children_[feature_value])


def build_tree(X, y, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
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

    highest_ig = -1.0
    highest_ig_feature_index = next(iter(features))

    for feature_index in features:
        current_ig = information_gain(X_subset, y_subset, feature_index)
        if current_ig > highest_ig:
            highest_ig = current_ig
            highest_ig_feature_index = feature_index

    if highest_ig < min_impurity_decrease:
        return LeafNode(majority_label)

    importances[highest_ig_feature_index] += (
        len(X_subset) / n_total_samples
    ) * highest_ig

    root_node = InternalNode(highest_ig_feature_index, children={})
    feature_column_arr = X_subset[:, highest_ig_feature_index]
    feature_values = np.unique(feature_column_arr)

    for feature_value in feature_values:
        mask = feature_column_arr == feature_value
        child_node = _build_tree(
            X_subset[mask],
            y_subset[mask],
            features=features - {highest_ig_feature_index},
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            n_total_samples=n_total_samples,
            importances=importances,
        )
        root_node.children_[feature_value] = child_node

    return root_node
