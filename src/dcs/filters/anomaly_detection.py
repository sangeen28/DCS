from typing import Optional
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None

class AnomalyDetector:
    """kNN-based anomaly detector on compressed update vectors."""
    def __init__(self, n_neighbors: int = 6):
        self.n_neighbors = max(1, n_neighbors)
        self._nn = None
        self._fit_X: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def fit(self, X: np.ndarray):
        if NearestNeighbors is not None:
            self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto").fit(X)
        else:
            self._fit_X = X.copy()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._nn is not None:
            dists, _ = self._nn.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)
            # use distance to k-th neighbor
            return dists[:, -1]
        # naive fallback
        dlist = []
        for x in X:
            dd = np.sqrt(((self._fit_X - x) ** 2).sum(axis=1))
            dd.sort()
            dlist.append(dd[min(self.n_neighbors - 1, len(dd) - 1)])
        return np.array(dlist)

    def set_quantile_threshold(self, scores: np.ndarray, q: float = 0.95):
        self.threshold_ = float(np.quantile(scores, q))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        scores = self.score_samples(X)
        thr = self.threshold_ if threshold is None else threshold
        if thr is None:
            thr = float(np.quantile(scores, 0.95))
        return (scores > thr).astype(np.int32), scores
