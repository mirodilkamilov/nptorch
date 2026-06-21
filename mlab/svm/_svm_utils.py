import numpy as np


def linear_kernel(X, Z):
    """K(x, z) = x · z"""
    return X @ Z.T


def rbf_kernel(X, Z):
    """K(x, z) = exp(-||x - z||²)"""
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Z_sq = np.sum(Z ** 2, axis=1, keepdims=True)
    sq_dists = X_sq + Z_sq.T - 2.0 * (X @ Z.T)
    return np.exp(-sq_dists)


def poly_kernel(X, Z, degree=3):
    """K(x, z) = (1 + x · z)^degree"""
    return (1.0 + X @ Z.T) ** degree


def compute_kernel(X, Z, kernel="rbf", degree=3):
    """Dispatch to the requested kernel and return the (n, m) kernel matrix."""
    if kernel == "linear":
        return linear_kernel(X, Z)
    elif kernel == "rbf":
        return rbf_kernel(X, Z)
    elif kernel == "poly":
        return poly_kernel(X, Z, degree)
    else:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Choose from 'linear', 'rbf', 'poly'."
        )
