#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dcs.config import load_config
from dcs.runner import run_experiment
from dcs.plotting import bar_sensitivity


def run_one(cfg, seed: int) -> tuple[float, float]:
    df = run_experiment(cfg, mode="DCS", seed=seed)
    final_test = float(df["test_acc"].iloc[-1]) if len(df) else 0.0
    avg_lat = float(df["avg_lat"].mean()) if len(df) else 0.0
    return final_test, avg_lat


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter sensitivity (computed, bar charts only).")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=str, default="outputs/sensitivity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=None, help="Override rounds for faster sensitivity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg.ROUNDS = int(args.rounds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep sensitivity practical: optionally reduce dataset size via config
    # (Use MAX_TRAIN_SAMPLES / MAX_TEST_SAMPLES in YAML)

    results = []

    # 1) TRUST_ALPHA sweep
    trust_grid = [0.1,0.3,0.5,0.6,0.7,0.8,0.9]
    for v in trust_grid:
        c = load_config(args.config)
        if args.rounds is not None:
            c.ROUNDS = int(args.rounds)
        c.TRUST_ALPHA = float(v)
        test_acc, avg_lat = run_one(c, seed=args.seed)
        results.append({"param":"TRUST_ALPHA","value":v,"test_acc":test_acc,"avg_lat":avg_lat})

    # 2) DDQL_GAMMA sweep
    gamma_grid = [0.80,0.85,0.90,0.93,0.95,0.97,0.99]
    for v in gamma_grid:
        c = load_config(args.config)
        if args.rounds is not None:
            c.ROUNDS = int(args.rounds)
        c.DDQL_GAMMA = float(v)
        test_acc, avg_lat = run_one(c, seed=args.seed)
        results.append({"param":"DDQL_GAMMA","value":v,"test_acc":test_acc,"avg_lat":avg_lat})

    # 3) ANN_NEIGHBORS sweep
    nn_grid = [1,2,3,4,5,6,7,8,9,10]
    for v in nn_grid:
        c = load_config(args.config)
        if args.rounds is not None:
            c.ROUNDS = int(args.rounds)
        c.ANN_NEIGHBORS = int(v)
        test_acc, avg_lat = run_one(c, seed=args.seed)
        results.append({"param":"ANN_NEIGHBORS","value":v,"test_acc":test_acc,"avg_lat":avg_lat})

    # 4) LAM_DEFAULT sweep (used in ScoreOnly; in DCS it is used in lam grid + policy)
    # Here we override lam grid to focus around the swept value for a cleaner trend.
    lam_grid = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for v in lam_grid:
        c = load_config(args.config)
        if args.rounds is not None:
            c.ROUNDS = int(args.rounds)
        c.LAM_GRID = [float(v)]  # force policy to use the swept lambda
        test_acc, avg_lat = run_one(c, seed=args.seed)
        results.append({"param":"LAMBDA","value":v,"test_acc":test_acc,"avg_lat":avg_lat})

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sensitivity_results.csv", index=False)

    # Plot bars (computed, not copied)
    for param in df["param"].unique():
        sub = df[df["param"] == param].copy()
        bar_sensitivity(
            sub,
            xcol="value",
            title_prefix=param,
            out_dir=str(out_dir),
        )

    print(f"Saved sensitivity outputs to: {out_dir}")


if __name__ == "__main__":
    main()
