#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dcs.config import load_config
from dcs.runner import run_experiment
from dcs.plotting import bar_compare_by_round, bar_summary_comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DCS vs baselines (bar charts only).")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--out", type=str, default="outputs/demo", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CONFIG ===")
    for k, v in cfg.__dict__.items():
        print(f"{k:>24}: {v}")

    df_dcs = run_experiment(cfg, mode="DCS", seed=args.seed)
    df_fed = run_experiment(cfg, mode="FedAvg", seed=args.seed)
    df_score = run_experiment(cfg, mode="ScoreOnly", seed=args.seed)

    df_dcs.to_csv(out_dir / "results_dcs.csv", index=False)
    df_fed.to_csv(out_dir / "results_fedavg.csv", index=False)
    df_score.to_csv(out_dir / "results_scoreonly.csv", index=False)

    # Per-round bar charts
    bar_compare_by_round(df_dcs, df_fed, "test_acc", "DCS", "FedAvg", step=1,
                        title="Test Accuracy vs Round (bar)", ylabel="Accuracy",
                        save_path=str(out_dir / "bar_test_acc_dcs_vs_fedavg.png"))
    bar_compare_by_round(df_dcs, df_fed, "avg_lat", "DCS", "FedAvg", step=1,
                        title="Avg Latency vs Round (bar)", ylabel="Seconds",
                        save_path=str(out_dir / "bar_latency_dcs_vs_fedavg.png"))
    bar_compare_by_round(df_dcs, df_fed, "comm_mb", "DCS", "FedAvg", step=1,
                        title="Comm Proxy vs Round (bar)", ylabel="MB",
                        save_path=str(out_dir / "bar_comm_dcs_vs_fedavg.png"))
    bar_compare_by_round(df_dcs, df_fed, "anom_rate", "DCS", "FedAvg", step=1,
                        title="Anomaly Rate vs Round (bar)", ylabel="Rate",
                        save_path=str(out_dir / "bar_anom_rate_dcs_vs_fedavg.png"))
    bar_compare_by_round(df_dcs, df_fed, "fairness", "DCS", "FedAvg", step=1,
                        title="Fairness (JFI) vs Round (bar)", ylabel="JFI",
                        save_path=str(out_dir / "bar_fairness_dcs_vs_fedavg.png"))

    # Summary bar charts
    summary_df = bar_summary_comparison(df_dcs, df_fed, name_a="DCS", name_b="FedAvg", out_dir=str(out_dir))
    summary_df.to_csv(out_dir / "summary_dcs_vs_fedavg.csv", index=False)

    # Quick table print
    print("\n=== SUMMARY (DCS vs FedAvg) ===")
    print(summary_df)


if __name__ == "__main__":
    main()
