from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def jain_fairness_index(values: List[float]) -> float:
    if not values:
        return 1.0
    v = np.array(values, dtype=np.float64)
    num = (v.sum() ** 2)
    den = len(v) * (v ** 2).sum()
    return float(num / den) if den > 0 else 0.0


@dataclass
class TrustManager:
    alpha: float = 0.6
    ts: Dict[int, float] = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = {}

    def get(self, cid: int) -> float:
        return float(self.ts.get(int(cid), 0.5))

    def update(self, cid: int, quality: float) -> None:
        cid = int(cid)
        old = self.get(cid)
        new = self.alpha * old + (1.0 - self.alpha) * float(quality)
        self.ts[cid] = float(np.clip(new, 0.0, 1.0))


@dataclass
class LatencyPredictor:
    ema: float = 0.7
    pred: Dict[int, float] = None

    def __post_init__(self):
        if self.pred is None:
            self.pred = {}

    def get(self, cid: int) -> float:
        return float(self.pred.get(int(cid), 1.0))

    def update(self, cid: int, observed_latency: float) -> None:
        cid = int(cid)
        old = self.get(cid)
        new = self.ema * old + (1.0 - self.ema) * float(observed_latency)
        self.pred[cid] = float(max(1e-6, new))


class ScoreSelector:
    """Trust/latency scoring + top-K selection."""

    def __init__(self, num_clients: int, trust_alpha: float = 0.6, lat_ema: float = 0.7):
        self.num_clients = int(num_clients)
        self.trust = TrustManager(alpha=float(trust_alpha))
        self.lat = LatencyPredictor(ema=float(lat_ema))
        self.part_cnt = np.zeros(self.num_clients, dtype=np.int32)

    def score(self, cid: int, lam: float) -> float:
        ts = self.trust.get(cid)
        L = self.lat.get(cid)
        invL = 1.0 / max(1e-6, L)
        return float(lam * ts + (1.0 - lam) * invL)

    def select_topk(self, candidates, k: int, lam: float):
        if not candidates:
            return []
        k = int(min(max(1, k), len(candidates)))
        scored = [(self.score(d.device_id, lam), d) for d in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [d for _, d in scored[:k]]
        for d in chosen:
            self.part_cnt[d.device_id] += 1
        return chosen

    def update_after_round(self, cids: List[int], qualities: List[float], latencies: List[float]) -> None:
        for cid, q, L in zip(cids, qualities, latencies):
            self.trust.update(cid, q)
            self.lat.update(cid, L)

    def fairness(self) -> float:
        vals = self.part_cnt[self.part_cnt > 0].tolist()
        return jain_fairness_index(vals)
