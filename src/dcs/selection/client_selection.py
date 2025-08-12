from typing import Dict, List, Tuple
import numpy as np

class DCSSelector:
    """Trust + Latency + (optional) anomaly-aware scoring for client selection."""
    def __init__(self, num_clients: int, alpha: float = 0.6, beta: float = 0.6, lam: float = 0.5):
        self.num_clients = num_clients
        self.alpha = alpha  # trust decay
        self.beta = beta    # historical perf decay
        self.lam = lam      # weighting TS vs 1/L in composite score
        self.trust = np.zeros(num_clients, dtype=np.float32)
        self.hist_perf = np.zeros(num_clients, dtype=np.float32)
        self.latency = np.ones(num_clients, dtype=np.float32)  # start with 1 to avoid div by 0

    def update_trust(self, client_deltas: Dict[int, float]):
        for cid, q in client_deltas.items():
            prev = self.trust[cid]
            self.trust[cid] = self.alpha * prev + (1.0 - self.alpha) * float(q)

    def update_hist_perf(self, client_acc_improve: Dict[int, float]):
        for cid, inc in client_acc_improve.items():
            prev = self.hist_perf[cid]
            self.hist_perf[cid] = self.beta * prev + (1.0 - self.beta) * float(inc)

    def update_latency(self, latency_measurements: Dict[int, float]):
        for cid, l in latency_measurements.items():
            prev = self.latency[cid]
            self.latency[cid] = 0.8 * prev + 0.2 * max(1e-6, float(l))

    def composite_scores(self) -> np.ndarray:
        inv_latency = 1.0 / np.maximum(1e-6, self.latency)
        ts = (self.trust - self.trust.min()) / (1e-9 + (self.trust.max() - self.trust.min()))
        il = (inv_latency - inv_latency.min()) / (1e-9 + (inv_latency.max() - inv_latency.min()))
        return self.lam * ts + (1.0 - self.lam) * il

    def select(self, k: int, mask_available: List[int] = None) -> List[int]:
        scores = self.composite_scores()
        if mask_available is not None:
            mask = np.zeros_like(scores, dtype=bool)
            mask[mask_available] = True
            scores = np.where(mask, scores, -1e9)
        return np.argpartition(-scores, kth=min(k-1, len(scores)-1))[:k].tolist()
