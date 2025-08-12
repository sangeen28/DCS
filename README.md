# Federated Learning Client Selection using Double Deep Q-Learning (DDQL)

This project implements an optimized client selection mechanism for Federated Learning using reinforcement learning. It is based on Double Deep Q-Learning (DDQL) and includes components for trust score evaluation, historical performance tracking, latency management, and anomaly detection using Approximate Nearest Neighbor (ANN) search.

## Features:
- **Reinforcement Learning** for client selection.
- **Incremental PCA** for dimensionality reduction in model updates.
- **Anomaly detection** using Approximate Nearest Neighbor (ANN) search.
- **Comparison with other federated learning methods** like FedAvg, FAVOR, and FLASH-RL.
- **Non-IID Data Partitioning**: Supports Dirichlet-based and shard-based partitioning strategies.

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
