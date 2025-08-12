from typing import Optional
import numpy as np

try:
    from sklearn.decomposition import IncrementalPCA as SkIncrementalPCA
except Exception:  # pragma: no cover
    SkIncrementalPCA = None

class IncrementalPCA:
    """Wrapper that uses sklearn if available, otherwise a simple PCA fallback."""
    def __init__(self, n_components: int = 50, batch_size: Optional[int] = None):
        self.n_components = n_components
        self.batch_size = batch_size
        self._sk = SkIncrementalPCA(n_components=n_components, batch_size=batch_size) if SkIncrementalPCA else None
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self._n_seen = 0

    def partial_fit(self, X: np.ndarray):
        if self._sk is not None:
            self._sk.partial_fit(X)
            self.components_ = self._sk.components_
            self.mean_ = self._sk.mean_
            self._n_seen += X.shape[0]
            return self
        # Fallback: accumulate and do SVD once (for demo/prototyping)
        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            self._buf = X - self.mean_
        else:
            self._buf = np.vstack([self._buf, X - self.mean_])
        self._n_seen += X.shape[0]
        if self._n_seen >= max(1000, 5 * self.n_components):
            U, S, Vt = np.linalg.svd(self._buf, full_matrices=False)
            self.components_ = Vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._sk is not None:
            return self._sk.transform(X)
        if self.components_ is None:
            # not fitted enough yet; return X as-is
            return X[:, : self.n_components] if X.shape[1] >= self.n_components else X
        Xc = X - (self.mean_ if self.mean_ is not None else 0.0)
        return Xc @ self.components_.T
