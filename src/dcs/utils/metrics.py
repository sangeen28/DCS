from typing import Dict
import torch
import torch.nn.functional as F

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean().item()

def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)

def jains_fairness(counts: torch.Tensor) -> float:
    # Jain's index: (sum x)^2 / (n * sum x^2)
    if counts.numel() == 0:
        return 1.0
    s = counts.sum()
    n = counts.numel()
    denom = n * (counts.pow(2).sum() + 1e-12)
    return float((s * s) / denom) if denom > 0 else 1.0

def dict_smooth(d: Dict[str, float], momentum: float = 0.9, state: Dict[str, float] = None):
    """EMA smoothing for logged scalars."""
    state = {} if state is None else state
    out = {}
    for k, v in d.items():
        prev = state.get(k, v)
        out[k] = momentum * prev + (1 - momentum) * v
        state[k] = out[k]
    return out, state
