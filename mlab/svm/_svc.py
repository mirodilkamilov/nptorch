import numpy as np

from ._svm import SVM


class SVC(SVM):
    def __init__(self, kernel="rbf", C=1.0, degree=3, tol=1e-3, max_iter=1000):
        super().__init__(kernel=kernel, C=C, degree=degree)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Train the SVM on the given data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
        """
        n = X.shape[0]

        self.classes_ = np.unique(y)
        # Map labels to {-1, +1}
        y_ = np.where(y == self.classes_[1], 1.0, -1.0)

        K = self._compute_kernel(X, X)  # (n, n) — precomputed once
        alphas = np.zeros(n)
        b = 0.0
        E = -y_.copy()  # E[i] = f(xi) - yi; initially f = 0

        for _ in range(self.max_iter):
            num_changed = 0

            for i in range(n):
                Ei = E[i]
                # Skip if KKT conditions are satisfied within tolerance
                if not (
                    (y_[i] * Ei < -self.tol and alphas[i] < self.C)
                    or (y_[i] * Ei > self.tol and alphas[i] > 0.0)
                ):
                    continue

                # Second alpha: pick j that maximises |Ei - Ej|
                diffs = np.abs(Ei - E)
                diffs[i] = -1.0
                j = int(np.argmax(diffs))
                Ej = E[j]

                ai_old, aj_old = alphas[i], alphas[j]

                # Box constraints for alpha_j (from linear equality + box [0, C])
                if y_[i] != y_[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)

                if L >= H:
                    continue

                # Second derivative of dual objective (must be positive for a maximum)
                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 0.0:
                    continue

                # Unconstrained optimum clipped to [L, H]
                alphas[j] = np.clip(aj_old + y_[j] * (Ei - Ej) / eta, L, H)
                delta_j = alphas[j] - aj_old
                if abs(delta_j) < 1e-5:
                    continue

                # Maintain sum(alpha * y) = 0
                alphas[i] = ai_old + y_[i] * y_[j] * (-delta_j)
                delta_i = alphas[i] - ai_old

                # Bias: b1 zeros out Ei_new, b2 zeros out Ej_new
                b_old = b
                b1 = b - Ei - y_[i] * delta_i * K[i, i] - y_[j] * delta_j * K[i, j]
                b2 = b - Ej - y_[i] * delta_i * K[i, j] - y_[j] * delta_j * K[j, j]
                if 0.0 < alphas[i] < self.C:
                    b = b1
                elif 0.0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # Incremental error update: O(n) per pair instead of O(n²)
                E += y_[i] * delta_i * K[i] + y_[j] * delta_j * K[j] + (b - b_old)

                num_changed += 1

            if num_changed == 0:
                break

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
