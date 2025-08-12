from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

def partition_dirichlet(num_items: int, num_clients: int, alpha: float = 0.5) -> List[np.ndarray]:
    """Return list of index arrays per client using Dirichlet(α) split."""
    idxs = np.arange(num_items)
    np.random.shuffle(idxs)
    proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
    splits = (proportions.cumsum() * num_items).astype(int)
    slices = np.split(idxs, splits[:-1])
    return [np.array(s, dtype=np.int64) for s in slices]

def make_client_loaders(dataset, client_indices: List[np.ndarray], batch_size: int = 64, shuffle: bool = True) -> Dict[int, DataLoader]:
    loaders = {}
    for cid, idx in enumerate(client_indices):
        subset = Subset(dataset, idx.tolist())
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loaders

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
