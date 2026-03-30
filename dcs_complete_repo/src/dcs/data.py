from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split


class ArrayImageDataset(Dataset):
    """Simple torch Dataset backed by in-memory tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx].item())


def _torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def load_torchvision_dataset(
    name: str,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load dataset + create train/val/test torch datasets.

    Primary path uses torchvision.
    Fallback path uses tensorflow.keras.datasets (useful if torchvision is unavailable/broken).

    Returns:
        train_ds, val_ds, test_ds
    """
    name = str(name).lower().strip()

    if _torchvision_available():
        # Import inside the branch to avoid hard-failing on environments where torchvision is broken.
        from torchvision import datasets, transforms

        if name == "mnist":
            tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            tr = datasets.MNIST("./data", train=True, download=True, transform=tfm)
            te = datasets.MNIST("./data", train=False, download=True, transform=tfm)
        elif name == "fashion_mnist":
            tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
            tr = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
            te = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)
        elif name == "cifar10":
            tf_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            tf_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            tr = datasets.CIFAR10("./data", train=True, download=True, transform=tf_train)
            te = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
        else:
            raise ValueError(f"Unsupported dataset: {name}")

        if max_train_samples is not None:
            idx = np.random.RandomState(seed).permutation(len(tr))[: int(max_train_samples)]
            tr = Subset(tr, idx.tolist())
        if max_test_samples is not None:
            idx = np.random.RandomState(seed + 1).permutation(len(te))[: int(max_test_samples)]
            te = Subset(te, idx.tolist())

        val_size = max(1, len(tr) // 10)
        train_size = len(tr) - val_size
        train_ds, val_ds = random_split(tr, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        return train_ds, val_ds, te

    # -------------------------
    # Fallback: tensorflow.keras.datasets
    # -------------------------
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.datasets import cifar10, fashion_mnist, mnist
    except Exception as e:
        raise RuntimeError(
            "Neither torchvision nor tensorflow.keras datasets are available. "
            "Install one of them (recommended: torchvision) to load MNIST/FashionMNIST/CIFAR-10."
        ) from e

    if name == "mnist":
        (xtr, ytr), (xte, yte) = mnist.load_data()
        xtr = xtr.astype(np.float32) / 255.0
        xte = xte.astype(np.float32) / 255.0
        # [N, 1, 28, 28]
        xtr = xtr[:, None, :, :]
        xte = xte[:, None, :, :]
        mean, std = 0.1307, 0.3081
        xtr = (xtr - mean) / std
        xte = (xte - mean) / std
    elif name == "fashion_mnist":
        (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
        xtr = xtr.astype(np.float32) / 255.0
        xte = xte.astype(np.float32) / 255.0
        xtr = xtr[:, None, :, :]
        xte = xte[:, None, :, :]
        mean, std = 0.2860, 0.3530
        xtr = (xtr - mean) / std
        xte = (xte - mean) / std
    elif name == "cifar10":
        (xtr, ytr), (xte, yte) = cifar10.load_data()
        ytr = ytr.reshape(-1)
        yte = yte.reshape(-1)
        xtr = xtr.astype(np.float32) / 255.0
        xte = xte.astype(np.float32) / 255.0
        # [N, 3, 32, 32]
        xtr = np.transpose(xtr, (0, 3, 1, 2))
        xte = np.transpose(xte, (0, 3, 1, 2))
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)[:, None, None]
        xtr = (xtr - mean) / std
        xte = (xte - mean) / std
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    rs = np.random.RandomState(seed)
    if max_train_samples is not None:
        idx = rs.permutation(len(xtr))[: int(max_train_samples)]
        xtr, ytr = xtr[idx], np.array(ytr)[idx]
    if max_test_samples is not None:
        idx = rs.permutation(len(xte))[: int(max_test_samples)]
        xte, yte = xte[idx], np.array(yte)[idx]

    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(np.array(ytr, dtype=np.int64))
    xte_t = torch.from_numpy(xte)
    yte_t = torch.from_numpy(np.array(yte, dtype=np.int64))

    full_train = ArrayImageDataset(xtr_t, ytr_t)
    test_ds = ArrayImageDataset(xte_t, yte_t)

    val_size = max(1, len(full_train) // 10)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train_ds, val_ds, test_ds


def create_non_iid_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.8,
    min_samples: int = 10,
    seed: int = 42,
) -> List[np.ndarray]:
    """Dirichlet non-IID partitioning.

    Returns list of index arrays (indices are within `dataset` indexing).
    """
    num_clients = int(num_clients)
    n = len(dataset)
    if n == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]

    rng = np.random.RandomState(seed)

    if alpha <= 0:
        indices = rng.permutation(n)
        return [arr.astype(np.int64) for arr in np.array_split(indices, num_clients)]

    labels = np.array([dataset[i][1] for i in range(n)], dtype=np.int64)
    num_classes = int(labels.max() + 1)

    # class x clients proportions
    label_distribution = rng.dirichlet([alpha] * num_clients, num_classes)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        props = label_distribution[c]
        props = props / props.sum()
        splits = (np.cumsum(props) * len(class_idx)).astype(int)[:-1]
        chunks = np.split(class_idx, splits) if len(splits) else [class_idx]

        for client_id, chunk in enumerate(chunks[:num_clients]):
            client_indices[client_id].extend(chunk.tolist())

    # Ensure min samples by borrowing
    for i in range(num_clients):
        if len(client_indices[i]) < min_samples:
            needed = min_samples - len(client_indices[i])
            for j in range(num_clients):
                if j == i:
                    continue
                extra = len(client_indices[j]) - min_samples
                if extra > 0:
                    take = min(needed, extra)
                    client_indices[i].extend(client_indices[j][:take])
                    client_indices[j] = client_indices[j][take:]
                    needed -= take
                    if needed <= 0:
                        break

    return [np.array(idx, dtype=np.int64) for idx in client_indices]
