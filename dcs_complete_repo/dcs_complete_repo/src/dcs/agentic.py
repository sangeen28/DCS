from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class AgenticPlan:
    k: int
    lam: float
    mode: str
    risk_budget: float
    trust_floor: float


class AgenticCoordinator:
    """Lightweight multi-agent coordinator for client selection.

    This module provides an operational implementation of a mission planner,
    strategist, risk gate, and fairness-aware client intelligence layer.
    """

    def __init__(self, num_clients: int, total_rounds: int):
        self.num_clients = int(num_clients)
        self.total_rounds = int(max(1, total_rounds))

        self.last_mode = "explore"
        self.history: List[Dict[str, float]] = []

    def mission_mode(self, round_idx: int, last_anom_rate: float, last_lat: float, last_val: float) -> str:
        progress = float(round_idx / max(1, self.total_rounds - 1))

        # Risk-first override
        if last_anom_rate > 0.25:
            return "robust"

        # Phase scheduling with latency-aware correction
        if progress < 0.25:
            mode = "explore"
        elif progress < 0.75:
            mode = "quality"
        else:
            mode = "converge"

        if last_lat > 8.0 and mode != "robust":
            mode = "speed"
        if last_val < 0.2 and progress > 0.4:
            mode = "quality"
        return mode

    def propose_plan(
        self,
        base_k: int,
        base_lam: float,
        mode: str,
        k_min: int,
        k_max: int,
        lam_grid: Sequence[float],
    ) -> AgenticPlan:
        k = int(base_k)
        lam = float(base_lam)

        if mode == "robust":
            k = max(k_min, int(round(0.8 * k)))
            lam = min(0.9, max(0.6, lam + 0.15))
            risk_budget = 0.20
            trust_floor = 0.45
        elif mode == "speed":
            k = max(k_min, int(round(0.85 * k)))
            lam = float(np.clip(lam - 0.15, 0.15, 0.85))
            risk_budget = 0.35
            trust_floor = 0.30
        elif mode == "quality":
            k = min(k_max, int(round(1.15 * k)))
            lam = min(0.95, max(0.55, lam + 0.1))
            risk_budget = 0.30
            trust_floor = 0.40
        elif mode == "converge":
            k = max(k_min, int(round(0.95 * k)))
            lam = min(0.95, max(0.65, lam + 0.1))
            risk_budget = 0.25
            trust_floor = 0.45
        else:  # explore
            k = min(k_max, int(round(1.1 * k)))
            lam = float(np.clip(lam, 0.2, 0.8))
            risk_budget = 0.40
            trust_floor = 0.25

        # Snap lambda to closest configured grid value.
        if lam_grid:
            lam = float(min(lam_grid, key=lambda x: abs(float(x) - lam)))

        k = int(min(max(k, k_min), k_max))
        return AgenticPlan(k=k, lam=lam, mode=mode, risk_budget=risk_budget, trust_floor=trust_floor)

    def select_clients(self, selector, candidates, plan: AgenticPlan) -> Tuple[List, Dict[str, float]]:
        if not candidates:
            return [], {"risk_score": 0.0, "fallback_used": 0.0}

        pmax = float(max(1, np.max(selector.part_cnt)))
        scored = []
        for d in candidates:
            cid = int(d.device_id)
            trust = float(selector.trust.get(cid))
            lat = float(selector.lat.get(cid))
            inv_lat = 1.0 / max(1e-6, lat)
            fairness_debt = 1.0 - float(selector.part_cnt[cid] / pmax)

            # Risk is high for low-trust / high-latency clients.
            risk = float(np.clip((1.0 - trust) * 0.7 + (1.0 - np.tanh(inv_lat)) * 0.3, 0.0, 1.0))
            utility = plan.lam * trust + (1.0 - plan.lam) * inv_lat + 0.2 * fairness_debt - 0.35 * risk
            scored.append((utility, risk, trust, d))

        scored.sort(key=lambda x: x[0], reverse=True)

        gated = [x for x in scored if x[1] <= plan.risk_budget and x[2] >= plan.trust_floor]
        pool = gated if len(gated) >= max(1, plan.k // 2) else scored

        chosen = [d for _, _, _, d in pool[: min(plan.k, len(pool))]]

        # Fallback: if hard gating leaves too few, fill from highest utility remaining clients.
        if len(chosen) < min(plan.k, len(candidates)):
            chosen_ids = {d.device_id for d in chosen}
            for _, _, _, d in scored:
                if d.device_id in chosen_ids:
                    continue
                chosen.append(d)
                chosen_ids.add(d.device_id)
                if len(chosen) >= min(plan.k, len(candidates)):
                    break
            fallback_used = 1.0
        else:
            fallback_used = 0.0

        for d in chosen:
            selector.part_cnt[d.device_id] += 1

        chosen_risks = [r for _, r, _, d in scored if d.device_id in {x.device_id for x in chosen}]
        risk_score = float(np.mean(chosen_risks)) if chosen_risks else 0.0

        return chosen, {"risk_score": risk_score, "fallback_used": fallback_used}

    def record_outcome(self, plan: AgenticPlan, anom_rate: float, avg_lat: float, val_acc: float) -> None:
        self.last_mode = plan.mode
        self.history.append(
            {
                "mode": float(hash(plan.mode) % 997),
                "anom_rate": float(anom_rate),
                "avg_lat": float(avg_lat),
                "val_acc": float(val_acc),
                "lam": float(plan.lam),
                "k": float(plan.k),
            }
        )
        if len(self.history) > 512:
            self.history = self.history[-512:]
