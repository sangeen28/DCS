# Intelligent Agentic Client Selection (Standalone Extension)

This extension adds a **new standalone agentic system** for federated client selection without changing baseline DCS code paths.

## What is new

- `src/dcs/agentic_system.py`
  - `MissionPlannerAgent`: picks round objective mode.
  - `BanditModeAgent`: UCB-style online mode arbitration.
  - `EpisodicMemoryAgent`: kNN memory over prior contexts for plan adaptation.
  - `StrategistAgent`: refines DDQL base `(K, λ)` to plan `(K_plan, λ_plan)`.
  - `ClientIntelligenceAgent`: computes utility from trust, inverse latency, and fairness debt.
  - `RiskGuardAgent`: gates clients by trust floor and risk budget.
  - `ComplianceAgent`: fallback completion to ensure enough clients are selected.
  - Diversity-aware client selection (MMR-style) to reduce over-concentration on one device type.
  - `IntelligentAgenticSelector`: orchestrates the above agents.
  - `run_agentic_experiment(cfg, seed)`: full FL run loop using the standalone agentic selector.

- `scripts/run_agentic_demo.py`
  - Runs `AgenticStandalone` and baseline `DCS` side-by-side.
  - Exports CSVs + summary artifacts.
  - Uses synthetic-data fallback when external dataset mirrors are unavailable, so end-to-end validation still works offline.

## Why standalone

This keeps the existing `run_experiment(..., mode="DCS" | "AgenticDCS" | "FedAvg" | "ScoreOnly")` architecture intact while introducing a novel intelligent agentic system as an additive module.

## Output columns (AgenticStandalone)

- `plan_mode`, `K_base`, `K_plan`, `lam_base`, `lam_plan`
- `risk_score`, `fallback_used`
- Standard FL metrics (`test_acc`, `avg_lat`, `comm_mb`, `anom_rate`, `fairness`, etc.)

## Run

```bash
PYTHONPATH=$PWD/src python scripts/run_agentic_demo.py --config configs/default.yaml --out outputs/agentic_demo
```
