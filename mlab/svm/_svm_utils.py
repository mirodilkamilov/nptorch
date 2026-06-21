import numpy as np


def linear_kernel(X, Z):
    """K(x, z) = x · z"""
    return X @ Z.T


def rbf_kernel(X, Z, gamma):
    """K(x, z) = exp(-gamma · ||x - z||²)"""
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Z_sq = np.sum(Z ** 2, axis=1, keepdims=True)
    sq_dists = X_sq + Z_sq.T - 2.0 * (X @ Z.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)  # clip tiny negatives from round-off
    return np.exp(-gamma * sq_dists)


def poly_kernel(X, Z, gamma, coef0, degree):
    """K(x, z) = (gamma · x · z + coef0)^degree"""
    return (gamma * (X @ Z.T) + coef0) ** degree


def compute_kernel(X, Z, kernel="rbf", gamma=1.0, coef0=0.0, degree=3):
    """Dispatch to the requested kernel and return the (n, m) kernel matrix."""
    if kernel == "linear":
        return linear_kernel(X, Z)
    elif kernel == "rbf":
        return rbf_kernel(X, Z, gamma)
    elif kernel == "poly":
        return poly_kernel(X, Z, gamma, coef0, degree)
    else:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Choose from 'linear', 'rbf', 'poly'."
        )


def resolve_gamma(gamma, X):
    """Resolve sklearn-style gamma ('scale', 'auto', or a number) to a float.

    'scale' -> 1 / (n_features · X.var())   (sklearn default)
    'auto'  -> 1 / n_features
    """
    if isinstance(gamma, str):
        if gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        elif gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(
            f"Unknown gamma {gamma!r}. Choose from 'scale', 'auto', or a float."
        )
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    return float(gamma)
