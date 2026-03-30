# Agentic DCS: Complete Intelligent Multi-Agent Client-Selection System

## 1) Goal and design principle

This design upgrades the current DCS-style client-selection simulator from a **single-policy loop** (DDQL chooses `(K, λ)`) into a **fully agentic control system** with explicit planner/critic/safety/recovery behaviors, memory, and self-improvement.

The resulting system preserves the original optimization target (global accuracy under latency/trust constraints), while adding:

- **goal decomposition** (what objective to optimize now),
- **multi-agent deliberation** (why a selection should be made),
- **runtime guardrails** (when not to trust an otherwise high-reward action), and
- **continual adaptation** (how to improve policy and rules over time).

---

## 2) Baseline architecture we inherit

From the existing DCS simulator:

- Action space: `a_t = (K_t, λ_t)`.
- DDQL policy outputs action each round.
- Client scoring combines trust and latency (weighted by `λ`).
- Trust and latency are maintained via EMA.
- Anomaly filtering uses stable projection + kNN outlier detection.
- Round result contains latency, updates, and quality signals.

This is already a strong **reactive selector**, but it is not yet a complete autonomous agent system.

---

## 3) Target: complete agentic system

### 3.1 Agent graph (roles)

Implement a coordinated graph of specialized agents:

1. **Mission Agent (MA)**
   - Interprets global objectives and SLOs for the current phase.
   - Creates a per-round intent: "speed-first", "robustness-first", "exploration", etc.

2. **Selection Strategist Agent (SSA)**
   - Proposes candidate `(K, λ)` actions.
   - Uses DDQL policy + model-based rollouts + exploration heuristics.

3. **Client Intelligence Agent (CIA)**
   - Builds per-client utility vectors from trust, latency, drift, energy, fairness debt, and reliability.
   - Produces ranked pools and backup pools.

4. **Risk & Adversary Agent (RAA)**
   - Extends anomaly detection beyond update-level outliers.
   - Detects collusion, burst attacks, and temporal anomalies.

5. **Fairness & Compliance Agent (FCA)**
   - Enforces constraints: minimum participation quotas, geography/device fairness, privacy budgets.

6. **Execution Orchestrator Agent (EOA)**
   - Dispatches selected clients.
   - Handles dropouts/retries, invokes contingency selection from backup pools.

7. **Critic & Learning Agent (CLA)**
   - Performs post-round evaluation and credit assignment.
   - Updates DDQL, risk thresholds, and agent prompts/policies.

8. **Memory Agent (MemA)**
   - Maintains episodic, semantic, and statistical memory.
   - Supports retrieval for planning (e.g., similar round contexts).

### 3.2 Control loop (each FL round)

1. **Observe**: collect state `s_t` from system + memory.
2. **Plan** (MA): set round objective profile and constraints.
3. **Generate candidates** (SSA/CIA): produce action and client subsets.
4. **Deliberate** (RAA/FCA/CLA): score risk, fairness impact, expected reward.
5. **Decide**: select primary plan + fallback plans.
6. **Execute** (EOA): run round, monitor live signals, trigger contingency if needed.
7. **Critique + Learn** (CLA/MemA): update policies, memories, thresholds, and confidence.

---

## 4) Formal problem statement

### 4.1 State

Augment baseline state with:

- Global training state: round, model delta norms, validation trend.
- System state: latency distributions, dropout rates, queue depth.
- Client state vector `x_i`: trust EMA, latency EMA, energy class, participation debt, gradient novelty, anomaly history.
- Risk context: attack likelihood, uncertainty estimates, detector drift.
- Governance context: fairness deficits, privacy budgets.

### 4.2 Action

Hierarchical action:

- Macro-action: `(K, λ, mode)` where `mode ∈ {speed, quality, robust, recovery, explore}`.
- Micro-action: selected client set `C_t`, backup set `B_t`, and retry policy.

### 4.3 Multi-objective reward

Use constrained multi-objective reward:

`R_t = w_acc * ΔAcc_t - w_lat * Lat_t - w_energy * Energy_t - w_risk * Risk_t - w_unfair * FairnessPenalty_t`

subject to:

- attack risk below threshold,
- fairness constraints,
- round deadline SLO,
- privacy/accounting constraints.

Use dynamic weights from MA based on training phase (cold start vs convergence).

---

## 5) Agent internals

### 5.1 Mission Agent

- Input: training phase classifier + business policy.
- Output:
  - objective profile vector `g_t`,
  - constraint bundle `Ω_t`,
  - exploration budget `ε_t`.

### 5.2 Selection Strategist Agent

Hybrid decision engine:

- **DDQL head**: baseline action proposer.
- **Model-based simulator head**: short horizon rollouts under dropout/attack scenarios.
- **Bandit head**: uncertainty-aware exploration over `(K, λ, mode)`.

Produces top-M action candidates with confidence intervals.

### 5.3 Client Intelligence Agent

Per-client utility:

`u_i = α1*trust_i - α2*lat_i + α3*novelty_i - α4*risk_i - α5*fairness_debt_i + α6*availability_i`

- Builds Pareto front of candidate clients.
- Outputs primary pool and resilient backup pool.

### 5.4 Risk & Adversary Agent

Detection stack:

- projection + kNN (existing),
- temporal change-point detector on trust/gradient stats,
- cluster-level collusion detector,
- uncertainty calibration for false positive control.

If risk spikes, it can force `mode=robust`, shrink `K`, and raise trust threshold.

### 5.5 Fairness & Compliance Agent

Tracks participation parity and caps over-selection.

Adds constraint-violation penalties and hard blocks for non-compliant plans.

### 5.6 Execution Orchestrator Agent

- Non-blocking dispatch.
- Adaptive timeout per client class.
- Retry from `B_t` when live failures exceed threshold.

### 5.7 Critic & Learning Agent

- Counterfactual analysis: compare executed action vs top rejected candidates.
- Updates:
  - DDQL replay buffer,
  - trust/risk calibration,
  - policy selection priors,
  - prompt/policy artifacts used by planners.

### 5.8 Memory Agent

Memory tiers:

- **Episodic**: round trajectories and outcomes.
- **Semantic**: distilled rules (e.g., "high dropout on class-X devices after 200s timeout").
- **Vector retrieval**: nearest historical contexts for reuse.

---

## 6) System architecture (runtime)

### 6.1 Services

- `state-service`: aggregates metrics and client telemetry.
- `planner-service`: MA + SSA.
- `risk-service`: RAA.
- `compliance-service`: FCA.
- `orchestrator-service`: EOA.
- `learning-service`: CLA + MemA.
- `policy-store`: DDQL checkpoints, thresholds, rule bundles.
- `event-bus`: async inter-agent communication.

### 6.2 Recommended tech choices

- Control plane: Python + FastAPI services.
- Messaging: Kafka / NATS.
- Feature store: Redis + parquet snapshots.
- Policy/model store: MLflow / object store.
- Observability: OpenTelemetry + Prometheus + Grafana.

---

## 7) Decision protocol and conflict resolution

1. SSA proposes top-M plans.
2. RAA and FCA independently veto/penalize.
3. MA applies priority policy:
   - Safety vetoes are hard.
   - Compliance vetoes are hard.
   - Performance penalties are soft-scored.
4. If all plans vetoed, trigger **recovery template**:
   - conservative `K`, robust mode, strict trust floor.

This prevents unsafe exploitation by a reward-maximizing policy.

---

## 8) Agentic data contracts

Use typed schemas for all exchanges.

- `RoundState`
- `ObjectiveBundle`
- `ActionCandidate`
- `RiskReport`
- `ComplianceReport`
- `ExecutionPlan`
- `RoundOutcome`
- `LearningUpdate`

Each payload includes: `trace_id`, `round_id`, `policy_version`, `confidence`, `explanations`.

---

## 9) Online learning and adaptation strategy

- **Fast loop (per round):** DDQL step + threshold calibration.
- **Medium loop (per N rounds):** retrain risk models, refresh fairness debts, tune utility coefficients.
- **Slow loop (daily/weekly):** offline policy distillation and A/B policy rollout.

Use canary deployment for new policies with rollback on SLO regression.

---

## 10) Safety, robustness, and governance

- Hard constraints for fairness/compliance.
- Adversarial simulation in staging (label flip, sybil, collusion, high-dropout storms).
- Explainability logs:
  - why this `(K, λ, mode)` was selected,
  - why clients were excluded,
  - which constraint was binding.
- Human override mode for production incidents.

---

## 11) Metrics and success criteria

### 11.1 Primary KPIs

- Final global test accuracy.
- Time-to-target-accuracy.
- Mean/95p round latency.
- Robust accuracy under attack.
- Fairness index across client cohorts.

### 11.2 Agent quality KPIs

- Veto precision/recall for RAA and FCA.
- Decision stability (unnecessary action churn).
- Recovery efficacy after incident trigger.
- Counterfactual regret vs best available candidate.

---

## 12) Reference implementation plan for this repository

### Phase 1: Agentic wrappers around current simulator

- Keep existing DDQL and scoring logic.
- Add explicit agent interfaces and message schemas.
- Insert risk/compliance veto stage before final selection.

### Phase 2: Memory + deliberation

- Add episodic memory store.
- Add candidate ranking explanations.
- Add fallback plan generation and retry policies.

### Phase 3: Production hardening

- Service decomposition with event bus.
- Online/offline policy lifecycle management.
- Observability dashboards and canary rollouts.

---

## 13) Pseudocode for round controller

```python
state = state_service.get_round_state(round_id)
objective = mission_agent.plan(state)

candidates = strategist_agent.propose(state, objective)      # top-M (K, λ, mode, pools)
risk_reports = risk_agent.evaluate(state, candidates)
compliance_reports = compliance_agent.evaluate(state, candidates)

feasible = apply_veto_and_penalties(candidates, risk_reports, compliance_reports)
if not feasible:
    plan = recovery_template(state, objective)
else:
    plan = choose_best(feasible, objective)

outcome = orchestrator.execute(plan)
learning_agent.update(state, plan, outcome, candidates)
memory_agent.write(round_id, state, plan, outcome)
```

---

## 14) Practical migration mapping from current components

- Existing trust EMA -> CIA feature + RAA signal.
- Existing latency EMA -> CIA + EOA timeout calibration.
- Existing DDQL -> SSA DDQL head.
- Existing anomaly filtering -> RAA baseline detector.
- Existing round loop -> EOA + CLA pipeline.

This minimizes refactor risk while enabling full agentic behavior.

---

## 15) Minimal first deliverable (MVP)

If you need the fastest path to "agentic":

1. Add `MissionAgent`, `RiskAgent`, `ComplianceAgent` wrappers.
2. Keep DDQL as sole proposal engine initially.
3. Add hard veto + fallback conservative plan.
4. Log structured explanations and counterfactual candidates.

That alone converts the architecture from reactive selection into a governed autonomous system.
