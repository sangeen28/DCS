from dataclasses import dataclass

@dataclass
class RewardWeights:
    accuracy: float = 1.0
    cost: float = 0.1
    anomaly: float = 0.5
    latency: float = 0.1

class RewardCalculator:
    """R = w_acc * Δacc - w_cost * cost - w_anom * num_anomalies - w_lat * latency"""
    def __init__(self, weights: RewardWeights = RewardWeights()):
        self.w = weights

    def __call__(self, delta_acc: float, cost: float, num_anomalies: int, latency: float) -> float:
        return (
            self.w.accuracy * float(delta_acc)
            - self.w.cost * float(cost)
            - self.w.anomaly * float(num_anomalies)
            - self.w.latency * float(latency)
        )
