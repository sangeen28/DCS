from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_num_bytes(model: nn.Module) -> int:
    # float32 params assumed
    n_params = sum(p.numel() for p in model.parameters())
    return int(n_params * 4)


def quick_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 2) -> float:
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            tot += float(loss.item())
            n += 1
    return float(tot / max(1, n))


def eval_accuracy(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 256) -> float:
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    return float(correct / max(1, total))


class LabelFlipWrapper(Dataset):
    def __init__(self, base: Dataset, a: int = 0, b: int = 1):
        self.base = base
        self.a = int(a)
        self.b = int(b)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        y = int(y)
        if y == self.a:
            y = self.b
        elif y == self.b:
            y = self.a
        return x, y


@dataclass
class DeviceCapability:
    device_type: str
    cpu_ghz: float
    ram_gb: float
    bandwidth_mbps: float
    energy_eff: float
    base_power_w: float
    dropout_p: float

    @staticmethod
    def sample(device_type: str, rng: np.random.Generator) -> "DeviceCapability":
        device_type = device_type.strip().lower()

        if device_type == "smartphone":
            cpu_ghz = float(rng.uniform(1.8, 2.4))
            ram_gb = float(rng.uniform(3.0, 6.0))
            bandwidth_mbps = float(rng.uniform(10.0, 80.0))
            energy_eff = float(rng.uniform(0.70, 0.95))
            base_power_w = float(rng.uniform(3.5, 6.0))
            dropout_p = float(rng.uniform(0.02, 0.08))
        elif device_type == "raspberry_pi":
            cpu_ghz = float(rng.uniform(1.2, 1.8))
            ram_gb = float(rng.uniform(1.5, 3.0))
            bandwidth_mbps = float(rng.uniform(5.0, 30.0))
            energy_eff = float(rng.uniform(0.50, 0.75))
            base_power_w = float(rng.uniform(2.0, 4.0))
            dropout_p = float(rng.uniform(0.03, 0.10))
        elif device_type == "jetson_edge":
            cpu_ghz = float(rng.uniform(2.0, 3.2))
            ram_gb = float(rng.uniform(6.0, 10.0))
            bandwidth_mbps = float(rng.uniform(40.0, 150.0))
            energy_eff = float(rng.uniform(0.80, 0.98))
            base_power_w = float(rng.uniform(6.0, 12.0))
            dropout_p = float(rng.uniform(0.01, 0.05))
        else:
            raise ValueError(f"Unknown device type: {device_type}")

        return DeviceCapability(
            device_type=device_type,
            cpu_ghz=cpu_ghz,
            ram_gb=ram_gb,
            bandwidth_mbps=bandwidth_mbps,
            energy_eff=energy_eff,
            base_power_w=base_power_w,
            dropout_p=dropout_p,
        )

    def maybe_dropout(self) -> bool:
        return random.random() < self.dropout_p

    def latency_components(self, model_bytes: int, local_steps: int, data_size: int) -> Tuple[float, float, float]:
        # Proxy model for demo: compute scales with local_steps and data, comm scales with model size / bandwidth
        comp = (local_steps * max(1, data_size) * 1e-6) / max(1e-6, self.cpu_ghz)
        comm = (2.0 * model_bytes * 8.0) / max(1e-6, self.bandwidth_mbps * 1e6)
        overhead = float(np.random.exponential(0.20))
        return float(comp), float(comm), float(overhead)

    def energy_wh(self, wall_s: float) -> float:
        power = self.base_power_w * (1.0 / max(1e-6, self.energy_eff))
        return float(power * wall_s / 3600.0)


class IoTDevice:
    def __init__(
        self,
        device_id: int,
        dataset_indices: np.ndarray,
        capability: DeviceCapability,
        edge_server_id: int,
        train_dataset: Dataset,
        batch_size: int,
    ):
        self.device_id = int(device_id)
        self.edge_server_id = int(edge_server_id)
        self.cap = capability

        valid = [int(i) for i in dataset_indices.tolist() if 0 <= int(i) < len(train_dataset)]
        if not valid:
            valid = [0]

        self.subset = Subset(train_dataset, valid)
        self.batch_size = int(batch_size)
        self.loader = DataLoader(self.subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.eval_loader = DataLoader(self.subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.connected = True
        self.participations = 0
        self.energy_total_wh = 0.0

    def local_train(
        self,
        global_model: nn.Module,
        device: torch.device,
        local_epochs: int,
        lr: float,
        momentum: float,
        is_malicious: bool,
        flip_pair: Tuple[int, int] = (0, 1),
        max_quality_batches: int = 2,
    ) -> Dict[str, object]:
        # dropout
        self.connected = not self.cap.maybe_dropout()
        if (not self.connected) or len(self.subset) == 0:
            return {"ok": False}

        if is_malicious:
            train_ds = LabelFlipWrapper(self.subset, a=flip_pair[0], b=flip_pair[1])
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        else:
            train_loader = self.loader

        pre_loss = quick_loss(global_model, self.eval_loader, device=device, max_batches=max_quality_batches)

        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()

        opt = torch.optim.SGD(local_model.parameters(), lr=float(lr), momentum=float(momentum))
        crit = nn.CrossEntropyLoss()

        total_loss, nb = 0.0, 0
        t0 = time.time()
        for _ in range(max(1, int(local_epochs))):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = local_model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                nb += 1

        train_wall = float(time.time() - t0)

        post_loss = quick_loss(local_model, self.eval_loader, device=device, max_batches=max_quality_batches)
        imp = max(0.0, pre_loss - post_loss)
        quality = float(np.clip(imp / max(1e-6, pre_loss), 0.0, 1.0))

        # delta vector
        gvec = parameters_to_vector(global_model.parameters()).detach()
        lvec = parameters_to_vector(local_model.parameters()).detach()
        delta_vec = (lvec - gvec).detach().cpu().numpy().astype(np.float32)

        steps_proxy = int(local_epochs * math.ceil(len(self.subset) / max(1, self.batch_size)))
        comp_s, comm_s, over_s = self.cap.latency_components(model_num_bytes(global_model), steps_proxy, len(self.subset))
        latency_total = float(train_wall + comp_s + comm_s + over_s)

        energy_wh = float(self.cap.energy_wh(latency_total))
        self.energy_total_wh += energy_wh
        self.participations += 1

        return {
            "ok": True,
            "delta_vec": delta_vec,
            "avg_loss": float(total_loss / max(1, nb)),
            "quality": quality,
            "lat_comp": comp_s,
            "lat_comm": comm_s,
            "lat_over": over_s,
            "lat_total": latency_total,
            "energy_wh": energy_wh,
            "n": int(len(self.subset)),
        }


class EdgeServer:
    def __init__(self, server_id: int):
        self.server_id = int(server_id)
        self.devices: List[IoTDevice] = []

    def add_device(self, dev: IoTDevice) -> None:
        self.devices.append(dev)

    @staticmethod
    def aggregate(deltas: List[np.ndarray], weights: List[int]) -> Optional[np.ndarray]:
        if not deltas:
            return None
        w = np.array(weights, dtype=np.float64)
        w = w / max(1e-12, w.sum())
        agg = np.zeros_like(deltas[0], dtype=np.float64)
        for dv, ww in zip(deltas, w):
            agg += dv.astype(np.float64) * ww
        return agg.astype(np.float32)


class CloudServer:
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
        self.round = 0

    def apply_delta(self, device: torch.device, delta_vec: np.ndarray) -> None:
        if delta_vec is None:
            return
        gvec = parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy().astype(np.float32)
        new = gvec + delta_vec
        vector_to_parameters(torch.from_numpy(new).to(device), self.global_model.parameters())
        self.round += 1
