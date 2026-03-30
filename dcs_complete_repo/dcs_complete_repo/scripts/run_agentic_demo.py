#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dcs.agentic_system import run_agentic_experiment
from dcs.config import load_config
from dcs.runner import run_experiment
from dcs.plotting import bar_summary_comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Run standalone intelligent agentic FL selector against baseline DCS.")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--out", type=str, default="outputs/agentic_demo", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_agentic = run_agentic_experiment(cfg, seed=args.seed, allow_synthetic_fallback=True)
    df_dcs = None
    try:
        df_dcs = run_experiment(cfg, mode="DCS", seed=args.seed)
    except Exception as e:
        print(f"[warn] Baseline DCS run failed (likely dataset/network issue): {e}")

    df_agentic.to_csv(out_dir / "results_agentic_standalone.csv", index=False)
    if df_dcs is not None:
        df_dcs.to_csv(out_dir / "results_dcs_baseline.csv", index=False)
        summary = bar_summary_comparison(df_agentic, df_dcs, name_a="AgenticStandalone", name_b="DCS", out_dir=str(out_dir))
        summary.to_csv(out_dir / "summary_agentic_standalone_vs_dcs.csv", index=False)
        print("\n=== SUMMARY (AgenticStandalone vs DCS) ===")
        print(summary)
    else:
        print("\n=== SUMMARY (AgenticStandalone only) ===")
        print(df_agentic.tail(1))


if __name__ == "__main__":
    main()
