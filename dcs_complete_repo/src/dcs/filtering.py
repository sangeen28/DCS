from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class StableIncPCAKNNFilter:
    """Stable IncPCA + kNN distance anomaly filter.

    Why "stable":
    - IncPCA requires n_samples >= n_components on each partial_fit.
    - In FL, the number of client updates per round can vary (dropouts), which can cause
      the PCA output dimension to change if you "adapt" components to batch size.

    We avoid that by keeping a fixed `proj_dim` and using:
    - Random projection fallback when a batch is too small for IncPCA.
    - Resetting the reference bank the first time we switch from random projection to PCA
      (to avoid mixing two incompatible embedding spaces).
    """

    proj_dim: int
    n_neighbors: int = 6
    contamination: float = 0.10
    max_ref: int = 2000
    seed: int = 42

    def __post_init__(self):
        self.proj_dim = int(max(2, self.proj_dim))
        self.n_neighbors = int(max(1, self.n_neighbors))
        self.contamination = float(self.contamination)
        self.max_ref = int(max(1, self.max_ref))
        self.rng = np.random.default_rng(self.seed)

        self._pca = IncrementalPCA(n_components=self.proj_dim)
        self._pca_fitted = False
        self._ref: Optional[np.ndarray] = None
        self._W = self.rng.standard_normal(size=(1, self.proj_dim)).astype(np.float32)  # will resize on demand

    def _ensure_W(self, in_dim: int):
        if self._W.shape[0] != in_dim:
            self._W = self.rng.standard_normal(size=(in_dim, self.proj_dim)).astype(np.float32)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            return X.astype(np.float32)

        n, d = X.shape

        # If the batch is too small, fall back to random projection.
        if n < self.proj_dim:
            self._ensure_W(d)
            return (X @ self._W).astype(np.float32)

        # If we haven't fitted PCA yet, this partial_fit will succeed.
        previously_fitted = self._pca_fitted
        if not self._pca_fitted:
            self._pca.partial_fit(X)
            self._pca_fitted = True

            # IMPORTANT: if we were using random projection earlier (due to small batches),
            # clear reference bank when PCA becomes active.
            if not previously_fitted and self._ref is not None:
                self._ref = None

        return self._pca.transform(X).astype(np.float32)

    def update_reference(self, Xc: np.ndarray) -> None:
        Xc = np.asarray(Xc, dtype=np.float32)
        if Xc.ndim != 2 or Xc.shape[0] == 0:
            return
        if self._ref is None:
            self._ref = Xc.astype(np.float32)
        else:
            self._ref = np.vstack([self._ref, Xc.astype(np.float32)])
        if len(self._ref) > self.max_ref:
            self._ref = self._ref[-self.max_ref :]

    def detect(self, Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (anomalies, scores) where scores are kNN distance scores."""
        Xc = np.asarray(Xc, dtype=np.float32)
        if Xc.ndim != 2 or Xc.shape[0] == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

        if self._ref is None or len(self._ref) < max(5, self.n_neighbors):
            return np.zeros(len(Xc), dtype=bool), np.zeros(len(Xc), dtype=np.float64)

        k = min(self.n_neighbors, len(self._ref))
        if k <= 1:
            return np.zeros(len(Xc), dtype=bool), np.zeros(len(Xc), dtype=np.float64)

        nnbr = NearestNeighbors(n_neighbors=k)
        nnbr.fit(self._ref)
        distances, _ = nnbr.kneighbors(Xc)
        scores = distances[:, -1].astype(np.float64)

        thr = np.percentile(scores, 100.0 * (1.0 - self.contamination))
        anomalies = scores > thr
        return anomalies.astype(bool), scores
