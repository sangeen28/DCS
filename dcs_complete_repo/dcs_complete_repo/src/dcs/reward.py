from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    w_perf: float = 1.0
    w_comp: float = 0.10
    w_comm: float = 0.05
    w_anom: float = 0.50
    w_lat: float = 0.10
    w_energy: float = 0.02
    w_fair: float = 0.20


class RewardCalc:
    def __init__(self, w: RewardWeights):
        self.w = w
        self.prev_val = 0.0

    def reward(
        self,
        val_acc: float,
        comp_cost: float,
        comm_cost_mb: float,
        num_anom: int,
        avg_lat: float,
        energy_wh: float,
        fairness: float,
        k_eff: int,
    ) -> float:
        dperf = float(val_acc - self.prev_val)
        self.prev_val = float(val_acc)

        return float(
            self.w.w_perf * dperf
            - self.w.w_comp * comp_cost
            - self.w.w_comm * comm_cost_mb
            - self.w.w_anom * (float(num_anom) / max(1.0, float(k_eff)))
            - self.w.w_lat * avg_lat
            - self.w.w_energy * energy_wh
            + self.w.w_fair * fairness
        )
