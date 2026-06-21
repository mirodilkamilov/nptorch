import numpy as np

from ._svm import SVM
from ._svm_utils import resolve_gamma


class SVC(SVM):
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", coef0=0.0, degree=3,
                 tol=1e-3, max_iter=1000, random_state=None):
        super().__init__(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree)
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train the SVM on the given data using Platt's SMO.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.size == 0 or y.size == 0:
            raise ValueError("Training data cannot be empty.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )

        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError(
                f"SVC supports binary classification only, but found "
                f"{self.classes_.shape[0]} class(es) in y."
            )

        n = X.shape[0]
        C = self.C
        tol = self.tol
        # Map labels to {-1, +1}
        y_ = np.where(y == self.classes_[1], 1.0, -1.0)

        self.gamma_ = resolve_gamma(self.gamma, X)
        K = self._compute_kernel(X, X)  # (n, n) — precomputed once

        alphas = np.zeros(n)
        b = 0.0
        E = -y_.copy()  # E[i] = f(xi) - yi; initially f = 0
        rng = np.random.default_rng(self.random_state)

        def take_step(i, j):
            """Jointly optimise (alpha_i, alpha_j). Return True if a step was taken."""
            nonlocal b, E
            if i == j:
                return False

            ai_old, aj_old = alphas[i], alphas[j]
            # Box constraints for alpha_j (from linear equality + box [0, C])
            if y_[i] != y_[j]:
                L = max(0.0, aj_old - ai_old)
                H = min(C, C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - C)
                H = min(C, ai_old + aj_old)
            if L >= H:
                return False

            # Second derivative of dual objective (must be positive for a maximum)
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            if eta <= 0.0:
                return False

            Ei, Ej = E[i], E[j]
            # Scalar clip to [L, H] — plain min/max avoids NumPy dispatch overhead
            aj_new = min(H, max(L, aj_old + y_[j] * (Ei - Ej) / eta))
            # Relative progress check (Platt): skip negligible moves
            if abs(aj_new - aj_old) < 1e-5 * (aj_new + aj_old + 1e-5):
                return False

            alphas[j] = aj_new
            delta_j = aj_new - aj_old
            alphas[i] = ai_old - y_[i] * y_[j] * delta_j  # maintain sum(alpha*y)=0
            delta_i = alphas[i] - ai_old

            # Bias: b1 zeros out Ei_new, b2 zeros out Ej_new
            b_old = b
            b1 = b - Ei - y_[i] * delta_i * K[i, i] - y_[j] * delta_j * K[i, j]
            b2 = b - Ej - y_[i] * delta_i * K[i, j] - y_[j] * delta_j * K[j, j]
            if 0.0 < alphas[i] < C:
                b = b1
            elif 0.0 < alphas[j] < C:
                b = b2
            else:
                b = (b1 + b2) / 2.0

            # Incremental error update: O(n) per pair instead of O(n²)
            E += y_[i] * delta_i * K[i] + y_[j] * delta_j * K[j] + (b - b_old)
            return True

        def examine(i):
            """Optimise alpha_i against the best partner; fall back if it fails."""
            ri = y_[i] * E[i]
            # Skip if KKT conditions are satisfied within tolerance
            if not ((ri < -tol and alphas[i] < C) or (ri > tol and alphas[i] > 0.0)):
                return False

            non_bound = np.where((alphas > 0.0) & (alphas < C))[0]
            # 1st choice: partner that maximises |Ei - Ej|
            if non_bound.size > 1:
                j = int(np.argmax(np.abs(E[i] - E)))
                if take_step(i, j):
                    return True
            # 2nd choice: all non-bound examples, random start
            for j in rng.permutation(non_bound):
                if take_step(i, int(j)):
                    return True
            # 3rd choice: all examples, random start
            for j in rng.permutation(n):
                if take_step(i, int(j)):
                    return True
            return False

        # Platt outer loop: alternate full sweeps with non-bound-only sweeps
        num_changed = 0
        examine_all = True
        for _ in range(self.max_iter):
            if not (num_changed > 0 or examine_all):
                break
            num_changed = 0
            if examine_all:
                indices = range(n)
            else:
                indices = np.where((alphas > 0.0) & (alphas < C))[0]
            for i in indices:
                num_changed += examine(int(i))

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        sv_mask = alphas > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = (alphas * y_)[sv_mask]  # alpha_i * y_i for each SV
        self.b_ = b

        return self

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        return np.where(
            self.decision_function(X) >= 0, self.classes_[1], self.classes_[0]
        )
