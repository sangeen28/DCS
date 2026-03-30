# DCS

This repo contains a **complete, runnable** reference implementation of a DCS-style federated learning simulator:

- **DDQL (Double DQN)** chooses the action *(K, λ)* each round
  - **K** = number of selected clients
  - **λ** = trade-off between **trust** and **latency** in the scoring function
- **Trust score** updated via EMA from a lightweight local “quality” proxy
- **Latency prediction** via EMA
- **Stable projection + kNN anomaly filtering**
  - Uses **IncrementalPCA** *when feasible*; otherwise falls back to a fixed random projection
- **AgenticDCS mode**
  - Adds mission-aware planning, risk gating, and fairness-aware fallback selection on top of DDQL actions

> This is intentionally a **demo-friendly** simulator: it models device heterogeneity, dropouts, and a malicious label-flip attack, while keeping runtime reasonable.

---

## Quick start (local)

- We recommend running in a fresh virtual environment with Python 3.x to improve compatibility across package versions and reproducibility.
- The repo uses a src/ layout (e.g., src/dcs/), so you must add src to PYTHONPATH. Such as: !PYTHONPATH=$PWD python -u scripts/run_demo.py --config configs/default.yaml


```bash
pip install -r requirements.txt

# Run the main demo (DCS vs FedAvg vs ScoreOnly) + bar charts
python scripts/run_demo.py --config configs/default.yaml

# Run hyperparameter sensitivity sweeps (bars generated from runs, not copied from tables)
python scripts/run_sensitivity.py --config configs/default.yaml --out outputs/sensitivity
```

Outputs (CSVs + PNGs) go to `outputs/` by default.

---

## Colab usage

In Colab, you can:

1. Upload this repository as a zip OR `git clone` it.
2. Run:

```bash
!pip -q install -r requirements.txt
!python scripts/run_demo.py --config configs/default.yaml
```

---

## How to change datasets

Edit `configs/default.yaml`:

```yaml
DATASET: mnist            # mnist | fashion_mnist | cifar10
```

Notes:
- `mnist` and `fashion_mnist` are faster.
- `cifar10` is slower due to a larger CNN.

You can also reduce runtime by setting:

```yaml
MAX_TRAIN_SAMPLES: 20000
MAX_TEST_SAMPLES: 5000
ROUNDS: 10
```

---

## Hyperparameters you will commonly tune

All hyperparameters live in `configs/default.yaml`.

### Federated setup
- `NUM_CLIENTS`, `NUM_EDGES`
- `ROUNDS`, `LOCAL_EPOCHS`, `BATCH_SIZE`
- `LR`, `MOMENTUM`

### Non-IID partition
- `DIRICHLET_ALPHA` (α)

### Client selection action space
- `K_MIN`, `K_MAX`, `K_STEP`
- `LAM_GRID` (λ values DDQL can choose)

### Trust/latency models
- `TRUST_ALPHA` (EMA factor)
- `LAT_EMA` (EMA factor)

### Anomaly filter
- `PCA_RANK`
- `ANN_NEIGHBORS`
- `CONTAMINATION`

### DDQL
- `DDQL_GAMMA` (discount γ)
- `DDQL_EPS_*`, `DDQL_TAU`, `DDQL_LR`

---

## Hyperparameter sensitivity bars 

`scripts/run_sensitivity.py` runs short experiments across grids and produces bar charts like:
- α (Dirichlet non-IID)
- γ (DDQL discount)
- `ANN_NEIGHBORS`
- λ-grid sweep (fixed scoring mode)

These bars come from **measured runs** with the same simulator.

---

## Repo layout

- `src/dcs/` – core simulator and algorithms
- `scripts/run_demo.py` – DCS vs baselines + bar charts
- `scripts/run_sensitivity.py` – hyperparameter sweeps + bar charts
- `configs/default.yaml` – editable config

---

