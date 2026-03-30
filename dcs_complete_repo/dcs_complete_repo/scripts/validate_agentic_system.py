#!/usr/bin/env python3
from __future__ import annotations

from dcs.agentic_system import run_agentic_experiment
from dcs.config import Config


def main() -> None:
    cfg = Config(
        DATASET="mnist",
        NUM_CLIENTS=12,
        NUM_EDGES=3,
        ROUNDS=2,
        K_MIN=3,
        K_MAX=5,
        K_STEP=1,
        MAX_TRAIN_SAMPLES=800,
        MAX_TEST_SAMPLES=200,
    )

    df = run_agentic_experiment(cfg, seed=11, allow_synthetic_fallback=True)

    required_cols = {
        "plan_mode",
        "K_base",
        "K_plan",
        "lam_base",
        "lam_plan",
        "risk_score",
        "fallback_used",
        "reward",
        "test_acc",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if len(df) == 0:
        raise RuntimeError("Agentic validation produced empty dataframe.")

    if not ((df["risk_score"] >= 0.0).all() and (df["risk_score"] <= 1.0).all()):
        raise RuntimeError("risk_score is out of [0, 1] range.")

    if not ((df["K_plan"] >= cfg.K_MIN).all() and (df["K_plan"] <= cfg.K_MAX).all()):
        raise RuntimeError("K_plan violates config bounds.")

    print("Agentic system validation passed.")
    print(df[["round", "plan_mode", "K_base", "K_plan", "lam_base", "lam_plan", "risk_score", "fallback_used"]].head())


if __name__ == "__main__":
    main()
