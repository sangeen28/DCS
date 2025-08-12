# DCS (Dynamic Client Selector) — Federated Learning Prototype

This repository contains a cleaned-up, modular implementation of the DCS system:
client selection via Double-DQN, trust/latency-aware scoring, incremental PCA
compression, and anomaly filtering. It keeps the prior filenames and extends them
inside a `src/dcs/` package, plus reproducible experiment scripts.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run experiments
MNIST:
```bash
python -m experiments.mnist_experiment --rounds 5 --clients 10 --select 5
```

CIFAR-10:
```bash
python -m experiments.cifar10_experiment --rounds 5 --clients 10 --select 5
```

> Tip: Start with small numbers to validate wiring, then scale up.
