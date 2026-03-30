from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .agentic import AgenticCoordinator
from .config import Config
from .data import create_non_iid_partition, load_torchvision_dataset
from .filtering import StableIncPCAKNNFilter
from .models import build_model
from .reward import RewardCalc, RewardWeights
from .rl import DQNConfig, DoubleDQN
from .system import CloudServer, DeviceCapability, EdgeServer, IoTDevice, get_device, set_seed
from .trust import ScoreSelector


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
    t_mean, t_std = float(np.mean(ts)), float(np.std(ts))
    inv_l_mean = float(np.mean([1.0 / max(1e-6, lat) for lat in ls]))
    fairness = float(selector.fairness())
    frac = float(round_idx / max(1, cfg.ROUNDS - 1))

    s = np.array(
        [
            t_mean,
            t_std,
            inv_l_mean,
            fairness,
            frac,
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
    if sd > 1e-8:
        s = (s - mu) / sd
    return s


def run_experiment(cfg: Config, mode: str = "DCS", seed: int = 42) -> pd.DataFrame:
    """Run one experiment.

    mode: "DCS" | "AgenticDCS" | "FedAvg" | "ScoreOnly"
    Returns a dataframe with per-round metrics.
    """
    mode = mode.strip()
    if mode not in {"DCS", "AgenticDCS", "FedAvg", "ScoreOnly"}:
        raise ValueError("mode must be one of: DCS, AgenticDCS, FedAvg, ScoreOnly")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()

    train_ds, val_ds, test_ds = load_torchvision_dataset(
        cfg.DATASET,
        max_train_samples=cfg.MAX_TRAIN_SAMPLES,
        max_test_samples=cfg.MAX_TEST_SAMPLES,
        seed=seed,
    )

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
    agentic = AgenticCoordinator(cfg.NUM_CLIENTS, cfg.ROUNDS)

    rows: List[Dict] = []
    last_anom_rate = 0.0
    last_comm_mb = 0.0
    last_lat = 0.0
    last_val = 0.0

    for t in tqdm(range(cfg.ROUNDS), desc=f"{mode} rounds"):
        s = _state_vector(cfg, selector, t, last_anom_rate, last_comm_mb, last_lat, last_val, ddql.eps)

        plan_mode = "none"
        risk_score = 0.0
        fallback_used = 0

        if mode in {"DCS", "AgenticDCS"}:
            a = ddql.act(s, training=True)
            k, lam = action_map[a]
            if mode == "AgenticDCS":
                mission = agentic.mission_mode(
                    round_idx=t,
                    last_anom_rate=last_anom_rate,
                    last_lat=last_lat,
                    last_val=last_val,
                )
                plan = agentic.propose_plan(
                    base_k=k,
                    base_lam=lam,
                    mode=mission,
                    k_min=cfg.K_MIN,
                    k_max=cfg.K_MAX,
                    lam_grid=lams,
                )
                k, lam = int(plan.k), float(plan.lam)
                plan_mode = plan.mode
        elif mode == "ScoreOnly":
            a = None
            k = int((cfg.K_MIN + cfg.K_MAX) // 2)
            lam = float(cfg.LAM_DEFAULT)
        else:
            a = None
            k = int((cfg.K_MIN + cfg.K_MAX) // 2)
            lam = float("nan")

        candidates = [d for d in devices if len(d.subset) > 0]
        if not candidates:
            break

        if mode == "FedAvg":
            chosen = random.sample(candidates, k=min(k, len(candidates)))
        elif mode == "AgenticDCS":
            chosen, meta = agentic.select_clients(selector=selector, candidates=candidates, plan=plan)
            risk_score = float(meta["risk_score"])
            fallback_used = int(meta["fallback_used"])
        else:
            chosen = selector.select_topk(candidates, k=min(k, len(candidates)), lam=float(lam))

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

        if len(deltas) == 0:
            val_acc = cloud.evaluate_accuracy(val_ds, device=device)
            test_acc = cloud.evaluate_accuracy(test_ds, device=device)
            fairness = selector.fairness()
            rows.append(
                {
                    "round": t + 1,
                    "mode": mode,
                    "K": int(k),
                    "lam": float(lam),
                    "selected": int(len(chosen)),
                    "kept": 0,
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "avg_lat": 0.0,
                    "comm_mb": 0.0,
                    "anom": 0,
                    "anom_rate": 0.0,
                    "energy_wh": 0.0,
                    "fairness": float(fairness),
                    "reward": 0.0,
                    "rl_loss": 0.0,
                    "epsilon": float(ddql.eps),
                    "agent_mode": str(plan_mode),
                    "risk_score": float(risk_score),
                    "fallback_used": int(fallback_used),
                }
            )
            last_val = float(val_acc)
            continue

        x = np.vstack(deltas).astype(np.float32)
        xc = flt.project(x)

        if mode == "FedAvg":
            anomalies = np.zeros(len(xc), dtype=bool)
            keep = np.ones(len(xc), dtype=bool)
        else:
            anomalies, _scores = flt.detect(xc)
            keep = ~anomalies

        num_anom = int(np.sum(anomalies))
        anom_rate = float(num_anom / max(1, len(xc)))

        if mode != "FedAvg":
            if np.any(keep):
                flt.update_reference(xc[keep])
            else:
                keep[:] = True
                num_anom = 0
                anom_rate = 0.0
                flt.update_reference(xc)

        deltas_f = [dv for dv, k_ in zip(deltas, keep) if k_]
        weights_f = [w for w, k_ in zip(weights, keep) if k_]
        qualities_f = [q for q, k_ in zip(qualities, keep) if k_]
        latencies_f = [lat for lat, k_ in zip(latencies, keep) if k_]
        energies_f = [e for e, k_ in zip(energies, keep) if k_]
        cids_f = [cid for cid, k_ in zip(cids, keep) if k_]

        edge_updates: List[np.ndarray] = []
        edge_weights: List[int] = []
        for es in edges:
            es_ids = {dd.device_id for dd in es.devices}
            es_deltas = []
            es_ws = []
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

        if mode != "FedAvg":
            selector.update_after_round(cids_f, qualities_f, latencies_f)

        val_acc = cloud.evaluate_accuracy(val_ds, device=device)
        test_acc = cloud.evaluate_accuracy(test_ds, device=device)

        avg_lat = float(np.mean(latencies_f)) if latencies_f else 0.0
        energy = float(np.sum(energies_f)) if energies_f else 0.0

        comp_bytes = int(proj_dim * 4)
        comm_mb = float((len(deltas_f) * 2.0 * comp_bytes) / (1024**2))

        comp_cost = float(np.sum([info["lat_comp"] for info, k_ in zip(infos, keep) if k_]))
        fairness = float(selector.fairness())

        if mode in {"DCS", "AgenticDCS"}:
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
            if mode == "AgenticDCS":
                agentic.record_outcome(plan=plan, anom_rate=anom_rate, avg_lat=avg_lat, val_acc=val_acc)
        else:
            rwd = 0.0
            rl_loss = 0.0

        rows.append(
            {
                "round": t + 1,
                "mode": mode,
                "K": int(k),
                "lam": float(lam),
                "selected": int(len(chosen)),
                "kept": int(len(deltas_f)),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "avg_lat": float(avg_lat),
                "comm_mb": float(comm_mb),
                "anom": int(num_anom),
                "anom_rate": float(anom_rate),
                "energy_wh": float(energy),
                "fairness": float(fairness),
                "reward": float(rwd),
                "rl_loss": float(rl_loss),
                "epsilon": float(ddql.eps),
                "agent_mode": str(plan_mode),
                "risk_score": float(risk_score),
                "fallback_used": int(fallback_used),
            }
        )

        last_anom_rate = float(anom_rate)
        last_comm_mb = float(comm_mb)
        last_lat = float(avg_lat)
        last_val = float(val_acc)

    return pd.DataFrame(rows)
