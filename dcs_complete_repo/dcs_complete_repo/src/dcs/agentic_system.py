from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from .config import Config
from .data import ArrayImageDataset, create_non_iid_partition, load_torchvision_dataset
from .filtering import StableIncPCAKNNFilter
from .models import build_model
from .reward import RewardCalc, RewardWeights
from .rl import DQNConfig, DoubleDQN
from .system import CloudServer, DeviceCapability, EdgeServer, IoTDevice, get_device, set_seed
from .trust import ScoreSelector


@dataclass
class MissionPlan:
    mode: str
    k: int
    lam: float
    trust_floor: float
    risk_budget: float


@dataclass
class RoundContext:
    round_idx: int
    progress: float
    anom_rate: float
    avg_lat: float
    fairness: float
    val_acc: float


class MissionPlannerAgent:
    def decide(self, round_idx: int, total_rounds: int, anom_rate: float, avg_lat: float) -> str:
        progress = float(round_idx / max(1, total_rounds - 1))
        if anom_rate > 0.2:
            return "robust"
        if progress < 0.25:
            return "explore"
        if avg_lat > 6.0:
            return "speed"
        if progress < 0.8:
            return "quality"
        return "converge"


class BanditModeAgent:
    """UCB-style mode arbitration inspired by contextual bandit literature."""

    def __init__(self):
        self.modes = ["explore", "quality", "speed", "robust", "converge"]
        self.counts: Dict[str, int] = {m: 0 for m in self.modes}
        self.values: Dict[str, float] = {m: 0.0 for m in self.modes}
        self.total_steps = 0

    def choose(self, preferred_mode: str, context: RoundContext) -> str:
        self.total_steps += 1
        ucb_scores = {}
        for m in self.modes:
            n = max(1, self.counts[m])
            bonus = np.sqrt(2.0 * np.log(max(2, self.total_steps)) / n)
            prior = 0.12 if m == preferred_mode else 0.0
            robust_bias = 0.15 if (m == "robust" and context.anom_rate > 0.15) else 0.0
            speed_bias = 0.08 if (m == "speed" and context.avg_lat > 5.0) else 0.0
            ucb_scores[m] = self.values[m] + bonus + prior + robust_bias + speed_bias
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def update(self, mode: str, reward: float) -> None:
        mode = str(mode)
        self.counts[mode] = self.counts.get(mode, 0) + 1
        n = self.counts[mode]
        old = self.values.get(mode, 0.0)
        self.values[mode] = old + (float(reward) - old) / max(1, n)


class EpisodicMemoryAgent:
    """Compact kNN memory over contexts for plan adaptation."""

    def __init__(self, max_items: int = 512):
        self.max_items = int(max_items)
        self.items: List[Tuple[np.ndarray, float, int, float]] = []  # (ctx, reward, k, lam)

    def _ctx_vec(self, context: RoundContext) -> np.ndarray:
        return np.array(
            [
                context.progress,
                context.anom_rate,
                context.avg_lat,
                context.fairness,
                context.val_acc,
            ],
            dtype=np.float32,
        )

    def suggest(self, context: RoundContext) -> Tuple[int | None, float | None]:
        if len(self.items) < 8:
            return None, None
        q = self._ctx_vec(context)
        dists = []
        for idx, (vec, reward, k, lam) in enumerate(self.items):
            d = float(np.linalg.norm(q - vec))
            dists.append((d, idx, reward, k, lam))
        dists.sort(key=lambda x: x[0])
        top = dists[:5]
        weights = np.array([1.0 / max(1e-6, t[0]) for t in top], dtype=np.float64)
        weights = weights / max(1e-12, weights.sum())
        k_hat = int(round(float(np.sum([w * t[3] for w, t in zip(weights, top)]))))
        lam_hat = float(np.sum([w * t[4] for w, t in zip(weights, top)]))
        return k_hat, lam_hat

    def update(self, context: RoundContext, reward: float, k: int, lam: float) -> None:
        self.items.append((self._ctx_vec(context), float(reward), int(k), float(lam)))
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]


class StrategistAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def refine(self, mode: str, k: int, lam: float) -> Tuple[int, float, float, float]:
        if mode == "robust":
            return max(self.cfg.K_MIN, int(round(0.8 * k))), min(0.9, lam + 0.2), 0.45, 0.25
        if mode == "speed":
            return max(self.cfg.K_MIN, int(round(0.85 * k))), max(0.2, lam - 0.2), 0.25, 0.35
        if mode == "quality":
            return min(self.cfg.K_MAX, int(round(1.1 * k))), min(0.95, lam + 0.1), 0.35, 0.30
        if mode == "converge":
            return max(self.cfg.K_MIN, int(round(0.95 * k))), min(0.95, lam + 0.15), 0.4, 0.25
        return min(self.cfg.K_MAX, int(round(1.15 * k))), float(lam), 0.2, 0.40


class ClientIntelligenceAgent:
    def rank(self, selector: ScoreSelector, candidates, lam: float) -> List[Tuple[float, int, object]]:
        pmax = float(max(1, np.max(selector.part_cnt)))
        ranked: List[Tuple[float, int, object]] = []
        for d in candidates:
            cid = int(d.device_id)
            trust = float(selector.trust.get(cid))
            lat = float(selector.lat.get(cid))
            inv_l = 1.0 / max(1e-6, lat)
            fairness_debt = 1.0 - float(selector.part_cnt[cid] / pmax)
            utility = lam * trust + (1.0 - lam) * inv_l + 0.15 * fairness_debt
            ranked.append((utility, cid, d))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked


class RiskGuardAgent:
    def gate(self, selector: ScoreSelector, ranked: List[Tuple[float, int, object]], trust_floor: float, risk_budget: float):
        gated = []
        for util, cid, d in ranked:
            trust = float(selector.trust.get(cid))
            lat = float(selector.lat.get(cid))
            risk = (1.0 - trust) * 0.7 + (lat / (lat + 1.0)) * 0.3
            if trust >= trust_floor and risk <= risk_budget:
                gated.append((util, cid, d, float(risk)))
        return gated


class ComplianceAgent:
    def ensure_minimum(self, chosen: List[object], ranked: List[Tuple[float, int, object]], k: int) -> List[object]:
        if len(chosen) >= k:
            return chosen[:k]
        used = {d.device_id for d in chosen}
        for _, _, d in ranked:
            if d.device_id in used:
                continue
            chosen.append(d)
            used.add(d.device_id)
            if len(chosen) >= k:
                break
        return chosen


class IntelligentAgenticSelector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mission = MissionPlannerAgent()
        self.bandit = BanditModeAgent()
        self.memory = EpisodicMemoryAgent(max_items=1024)
        self.strategist = StrategistAgent(cfg)
        self.cia = ClientIntelligenceAgent()
        self.risk = RiskGuardAgent()
        self.compliance = ComplianceAgent()

    def _diversified_pick(self, ranked, k: int) -> List[object]:
        """MMR-inspired diversity over device types."""
        chosen: List[object] = []
        used_types: Dict[str, int] = {}
        pool = ranked[:]
        while pool and len(chosen) < k:
            best_idx = 0
            best_score = -1e18
            for i, (util, _cid, d, _risk) in enumerate(pool):
                dtype = str(getattr(getattr(d, "cap", None), "device_type", "unknown"))
                diversity_bonus = 0.1 if used_types.get(dtype, 0) == 0 else -0.05 * used_types[dtype]
                score = float(util + diversity_bonus)
                if score > best_score:
                    best_score = score
                    best_idx = i
            util, cid, d, risk = pool.pop(best_idx)
            chosen.append(d)
            dtype = str(getattr(getattr(d, "cap", None), "device_type", "unknown"))
            used_types[dtype] = used_types.get(dtype, 0) + 1
        return chosen

    def select(
        self,
        selector: ScoreSelector,
        candidates,
        base_k: int,
        base_lam: float,
        context: RoundContext,
    ):
        preferred_mode = self.mission.decide(context.round_idx, self.cfg.ROUNDS, context.anom_rate, context.avg_lat)
        mode = self.bandit.choose(preferred_mode=preferred_mode, context=context)
        k, lam, trust_floor, risk_budget = self.strategist.refine(mode, base_k, base_lam)

        mem_k, mem_lam = self.memory.suggest(context)
        if mem_k is not None:
            k = int(round(0.7 * k + 0.3 * mem_k))
        if mem_lam is not None:
            lam = float(0.7 * lam + 0.3 * mem_lam)

        lam_grid = [float(x) for x in self.cfg.LAM_GRID]
        lam = float(min(lam_grid, key=lambda x: abs(x - lam)))
        k = int(min(max(k, self.cfg.K_MIN), self.cfg.K_MAX))

        ranked = self.cia.rank(selector, candidates, lam=lam)
        gated = self.risk.gate(selector, ranked, trust_floor=trust_floor, risk_budget=risk_budget)

        if gated:
            chosen = self._diversified_pick(gated, k=min(k, len(gated)))
            chosen_ids = {x.device_id for x in chosen}
            risk_score = float(np.mean([r for _u, _c, d, r in gated if d.device_id in chosen_ids]))
        else:
            chosen = []
            risk_score = 1.0

        chosen = self.compliance.ensure_minimum(chosen, ranked, k=min(k, len(candidates)))
        fallback = int(1 if len(gated) < min(k, len(candidates)) else 0)

        for d in chosen:
            selector.part_cnt[d.device_id] += 1

        return MissionPlan(mode=mode, k=k, lam=lam, trust_floor=trust_floor, risk_budget=risk_budget), chosen, risk_score, fallback

    def update_after_round(self, context: RoundContext, plan: MissionPlan, reward: float) -> None:
        self.bandit.update(plan.mode, reward=float(reward))
        self.memory.update(context=context, reward=float(reward), k=int(plan.k), lam=float(plan.lam))


def _state_vector(
    cfg: Config,
    selector: ScoreSelector,
    round_idx: int,
    last_anom_rate: float,
    last_comm_mb: float,
    last_lat: float,
    last_val: float,
    eps: float,
) -> np.ndarray:
    ts = [selector.trust.get(i) for i in range(cfg.NUM_CLIENTS)]
    ls = [selector.lat.get(i) for i in range(cfg.NUM_CLIENTS)]
    s = np.array(
        [
            float(np.mean(ts)),
            float(np.std(ts)),
            float(np.mean([1.0 / max(1e-6, x) for x in ls])),
            float(selector.fairness()),
            float(round_idx / max(1, cfg.ROUNDS - 1)),
            float(last_val),
            float(last_anom_rate),
            float(last_comm_mb),
            float(last_lat),
            float(eps),
            float(cfg.TRUST_ALPHA),
            float(cfg.DIRICHLET_ALPHA),
        ],
        dtype=np.float32,
    )
    mu, sd = float(s.mean()), float(s.std())
    return (s - mu) / sd if sd > 1e-8 else s


def _build_synthetic_dataset(cfg: Config, seed: int):
    """Fallback dataset to keep the full agentic pipeline executable without network."""
    rng = np.random.default_rng(seed)
    n_train = int(cfg.MAX_TRAIN_SAMPLES or 2000)
    n_test = int(cfg.MAX_TEST_SAMPLES or 500)
    n_classes = 10
    if str(cfg.DATASET).lower() == "cifar10":
        xtr = rng.normal(size=(n_train, 3, 32, 32)).astype(np.float32)
        xte = rng.normal(size=(n_test, 3, 32, 32)).astype(np.float32)
    else:
        xtr = rng.normal(size=(n_train, 1, 28, 28)).astype(np.float32)
        xte = rng.normal(size=(n_test, 1, 28, 28)).astype(np.float32)
    ytr = rng.integers(0, n_classes, size=(n_train,), dtype=np.int64)
    yte = rng.integers(0, n_classes, size=(n_test,), dtype=np.int64)

    full_train = ArrayImageDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    test_ds = ArrayImageDataset(torch.from_numpy(xte), torch.from_numpy(yte))

    val_size = max(1, len(full_train) // 10)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_ds, val_ds, test_ds


def run_agentic_experiment(cfg: Config, seed: int = 42, allow_synthetic_fallback: bool = True) -> pd.DataFrame:
    """Run a standalone intelligent agentic FL experiment without modifying baseline DCS modes."""

    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()

    try:
        train_ds, val_ds, test_ds = load_torchvision_dataset(
            cfg.DATASET,
            max_train_samples=cfg.MAX_TRAIN_SAMPLES,
            max_test_samples=cfg.MAX_TEST_SAMPLES,
            seed=seed,
        )
    except Exception:
        if not allow_synthetic_fallback:
            raise
        train_ds, val_ds, test_ds = _build_synthetic_dataset(cfg, seed=seed)
    client_indices = create_non_iid_partition(
        train_ds,
        num_clients=cfg.NUM_CLIENTS,
        alpha=cfg.DIRICHLET_ALPHA,
        min_samples=cfg.MIN_SAMPLES_PER_CLIENT,
        seed=seed,
    )

    edges = [EdgeServer(i) for i in range(cfg.NUM_EDGES)]
    device_types = ["smartphone", "raspberry_pi", "jetson_edge"]
    dist = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    dist = dist / dist.sum()

    devices: List[IoTDevice] = []
    for cid in range(cfg.NUM_CLIENTS):
        dtype = str(rng.choice(device_types, p=dist))
        cap = DeviceCapability.sample(device_type=dtype, rng=rng)
        edge_id = cid % cfg.NUM_EDGES
        dev = IoTDevice(
            device_id=cid,
            dataset_indices=client_indices[cid],
            capability=cap,
            edge_server_id=edge_id,
            train_dataset=train_ds,
            batch_size=cfg.BATCH_SIZE,
            seed=seed,
        )
        edges[edge_id].add_device(dev)
        devices.append(dev)

    global_model = build_model(cfg.DATASET).to(device)
    cloud = CloudServer(global_model)
    selector = ScoreSelector(cfg.NUM_CLIENTS, trust_alpha=cfg.TRUST_ALPHA, lat_ema=cfg.LAT_EMA)
    agentic = IntelligentAgenticSelector(cfg)

    proj_dim = cfg.projection_dim()
    flt = StableIncPCAKNNFilter(
        proj_dim=proj_dim,
        n_neighbors=cfg.ANN_NEIGHBORS,
        contamination=cfg.CONTAMINATION,
        max_ref=cfg.MAX_REF,
        seed=seed,
    )

    ks = list(range(cfg.K_MIN, cfg.K_MAX + 1, cfg.K_STEP))
    lams = [float(x) for x in cfg.LAM_GRID]
    action_map: List[Tuple[int, float]] = [(k, lam) for k in ks for lam in lams]

    ddql = DoubleDQN(
        cfg=DQNConfig(
            state_dim=12,
            hidden_dim=cfg.DDQL_HIDDEN,
            lr=cfg.DDQL_LR,
            gamma=cfg.DDQL_GAMMA,
            tau=cfg.DDQL_TAU,
            eps_start=cfg.DDQL_EPS_START,
            eps_end=cfg.DDQL_EPS_END,
            eps_decay=cfg.DDQL_EPS_DECAY,
            buffer_size=cfg.DDQL_BUFFER,
            batch_size=cfg.DDQL_BATCH,
        ),
        action_dim=len(action_map),
        device=device,
    )

    rewarder = RewardCalc(
        RewardWeights(
            w_perf=cfg.W_PERF,
            w_comp=cfg.W_COMP,
            w_comm=cfg.W_COMM,
            w_anom=cfg.W_ANOM,
            w_lat=cfg.W_LAT,
            w_energy=cfg.W_ENERGY,
            w_fair=cfg.W_FAIR,
        )
    )

    rows: List[Dict] = []
    last_anom_rate = 0.0
    last_comm_mb = 0.0
    last_lat = 0.0
    last_val = 0.0

    for t in tqdm(range(cfg.ROUNDS), desc="AgenticFL rounds"):
        s = _state_vector(cfg, selector, t, last_anom_rate, last_comm_mb, last_lat, last_val, ddql.eps)
        a = ddql.act(s, training=True)
        base_k, base_lam = action_map[a]
        context = RoundContext(
            round_idx=t,
            progress=float(t / max(1, cfg.ROUNDS - 1)),
            anom_rate=float(last_anom_rate),
            avg_lat=float(last_lat),
            fairness=float(selector.fairness()),
            val_acc=float(last_val),
        )

        candidates = [d for d in devices if len(d.subset) > 0]
        if not candidates:
            break

        plan, chosen, risk_score, fallback_used = agentic.select(
            selector=selector,
            candidates=candidates,
            base_k=base_k,
            base_lam=base_lam,
            context=context,
        )

        m = int(round(cfg.MALICIOUS_RATIO_SELECTED * len(chosen)))
        mal_ids = set(random.sample([d.device_id for d in chosen], k=m)) if m > 0 else set()

        infos: List[Dict] = []
        deltas: List[np.ndarray] = []
        weights: List[int] = []
        qualities: List[float] = []
        latencies: List[float] = []
        energies: List[float] = []
        cids: List[int] = []

        for d in chosen:
            info = d.local_train(
                global_model=cloud.global_model,
                local_epochs=cfg.LOCAL_EPOCHS,
                lr=cfg.LR,
                momentum=cfg.MOMENTUM,
                is_malicious=(d.device_id in mal_ids),
                flip_pair=cfg.flip_pair_tuple(),
                device=device,
                max_quality_batches=cfg.MAX_QUALITY_BATCHES,
            )
            if info.get("ok", False):
                infos.append(info)
                deltas.append(info["delta_vec"])
                weights.append(info["n"])
                qualities.append(info["quality"])
                latencies.append(info["lat_total"])
                energies.append(info["energy_wh"])
                cids.append(d.device_id)

        if not deltas:
            val_acc = cloud.evaluate_accuracy(val_ds, device=device)
            test_acc = cloud.evaluate_accuracy(test_ds, device=device)
            rows.append(
                {
                    "round": t + 1,
                    "mode": "AgenticStandalone",
                    "plan_mode": plan.mode,
                    "K_base": int(base_k),
                    "K_plan": int(plan.k),
                    "lam_base": float(base_lam),
                    "lam_plan": float(plan.lam),
                    "selected": int(len(chosen)),
                    "kept": 0,
                    "anom_rate": 0.0,
                    "avg_lat": 0.0,
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "risk_score": float(risk_score),
                    "fallback_used": int(fallback_used),
                    "reward": 0.0,
                }
            )
            agentic.update_after_round(context=context, plan=plan, reward=0.0)
            last_val = float(val_acc)
            continue

        x = np.vstack(deltas).astype(np.float32)
        xc = flt.project(x)
        anomalies, _ = flt.detect(xc)
        keep = ~anomalies
        if np.any(keep):
            flt.update_reference(xc[keep])
        else:
            keep[:] = True
            flt.update_reference(xc)
            anomalies[:] = False

        num_anom = int(np.sum(anomalies))
        anom_rate = float(num_anom / max(1, len(xc)))

        deltas_f = [dv for dv, k_ in zip(deltas, keep) if k_]
        weights_f = [w for w, k_ in zip(weights, keep) if k_]
        qualities_f = [q for q, k_ in zip(qualities, keep) if k_]
        latencies_f = [l for l, k_ in zip(latencies, keep) if k_]
        energies_f = [e for e, k_ in zip(energies, keep) if k_]
        cids_f = [cid for cid, k_ in zip(cids, keep) if k_]

        edge_updates: List[np.ndarray] = []
        edge_weights: List[int] = []
        for es in edges:
            es_ids = {dd.device_id for dd in es.devices}
            es_deltas, es_ws = [], []
            for cid, dv, w in zip(cids_f, deltas_f, weights_f):
                if cid in es_ids:
                    es_deltas.append(dv)
                    es_ws.append(w)
            if es_deltas:
                edge_updates.append(es.aggregate(es_deltas, es_ws))
                edge_weights.append(int(sum(es_ws)))

        if edge_updates:
            ew = np.array(edge_weights, dtype=np.float64)
            ew = ew / max(1e-12, ew.sum())
            agg = np.zeros_like(edge_updates[0], dtype=np.float64)
            for dv, ww in zip(edge_updates, ew):
                agg += dv.astype(np.float64) * ww
            cloud.apply_delta(agg.astype(np.float32), device=device)

        selector.update_after_round(cids_f, qualities_f, latencies_f)

        val_acc = cloud.evaluate_accuracy(val_ds, device=device)
        test_acc = cloud.evaluate_accuracy(test_ds, device=device)
        avg_lat = float(np.mean(latencies_f)) if latencies_f else 0.0
        energy = float(np.sum(energies_f)) if energies_f else 0.0

        comp_bytes = int(proj_dim * 4)
        comm_mb = float((len(deltas_f) * 2.0 * comp_bytes) / (1024**2))
        comp_cost = float(np.sum([info["lat_comp"] for info, k_ in zip(infos, keep) if k_]))
        fairness = float(selector.fairness())

        rwd = rewarder.reward(
            val_acc=val_acc,
            comp_cost=comp_cost,
            comm_cost_mb=comm_mb,
            num_anom=num_anom,
            avg_lat=avg_lat,
            energy_wh=energy,
            fairness=fairness,
            k_eff=len(deltas_f),
        )
        sp = _state_vector(cfg, selector, t + 1, anom_rate, comm_mb, avg_lat, val_acc, ddql.eps)
        ddql.mem.push(s, a, rwd, sp, 0.0)
        rl_loss = ddql.train_step()
        ddql.update_eps()
        agentic.update_after_round(context=context, plan=plan, reward=rwd)

        rows.append(
            {
                "round": t + 1,
                "mode": "AgenticStandalone",
                "plan_mode": plan.mode,
                "K_base": int(base_k),
                "K_plan": int(plan.k),
                "lam_base": float(base_lam),
                "lam_plan": float(plan.lam),
                "selected": int(len(chosen)),
                "kept": int(len(deltas_f)),
                "anom": int(num_anom),
                "anom_rate": float(anom_rate),
                "avg_lat": float(avg_lat),
                "comm_mb": float(comm_mb),
                "energy_wh": float(energy),
                "fairness": float(fairness),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "risk_score": float(risk_score),
                "fallback_used": int(fallback_used),
                "reward": float(rwd),
                "rl_loss": float(rl_loss),
                "epsilon": float(ddql.eps),
            }
        )

        last_anom_rate = float(anom_rate)
        last_comm_mb = float(comm_mb)
        last_lat = float(avg_lat)
        last_val = float(val_acc)

    return pd.DataFrame(rows)
