from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dcs.utils.metrics import accuracy, classification_loss
from dcs.filters.pca import IncrementalPCA
from dcs.filters.anomaly_detection import AnomalyDetector
from dcs.selection.client_selection import DCSSelector
from dcs.selection.reward_calculator import RewardCalculator, RewardWeights
from dcs.selection.q_learning import RLAgent, DQNConfig

@dataclass
class FLConfig:
    rounds: int = 5
    local_epochs: int = 1
    clients_per_round: int = 5
    lr: float = 1e-2
    momentum: float = 0.9
    lam: float = 0.5
    alpha: float = 0.6
    beta: float = 0.6
    pca_components: int = 50
    neighbors_k: int = 6
    epsilon: float = 0.1
    device: str = "cpu"

class FederatedClient:
    def __init__(self, cid: int, model: nn.Module, loader: DataLoader, device: str = "cpu"):
        self.cid = cid
        self.device = device
        self.loader = loader
        self.criterion = classification_loss
        self.model = copy.deepcopy(model).to(self.device)

    def set_global(self, global_state: Dict[str, torch.Tensor]):
        self.model.load_state_dict(global_state)

    def train_local(self, epochs: int = 1, lr: float = 1e-2, momentum: float = 0.9) -> Tuple[Dict[str, torch.Tensor], float, float]:
        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        start = time.time()
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                opt.step()
        end = time.time()
        delta = {k: p.detach().cpu().clone() for k, p in self.model.state_dict().items()}
        latency = end - start
        return delta, float(loss.item()), float(latency)

class FederatedServer:
    def __init__(self, model: nn.Module, clients: Dict[int, FederatedClient], cfg: FLConfig):
        self.device = cfg.device
        self.global_model = model.to(self.device)
        self.clients = clients
        self.cfg = cfg
        self.selector = DCSSelector(num_clients=len(clients), alpha=cfg.alpha, beta=cfg.beta, lam=cfg.lam)
        self.pca = IncrementalPCA(n_components=cfg.pca_components)
        self.detector = AnomalyDetector(n_neighbors=cfg.neighbors_k)
        state_dim = 3
        action_dim = len(clients)
        self.agent = RLAgent(DQNConfig(state_dim=state_dim, action_dim=action_dim, device=cfg.device))
        self.rewarder = RewardCalculator(RewardWeights())

        self.history = { "acc": [], "loss": [], "reward": [], "latency": [], "selected": [] }

    def broadcast(self):
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(self, deltas: List[Dict[str, torch.Tensor]], weights: List[int]):
        new_state = {}
        total = float(sum(weights))
        for k in deltas[0].keys():
            acc = None
            for d, w in zip(deltas, weights):
                v = d[k].float() * (w / total)
                acc = v if acc is None else acc + v
            new_state[k] = acc
        self.global_model.load_state_dict(new_state)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.global_model(x)
                tot_loss += float(torch.nn.functional.cross_entropy(logits, y).item()) * x.size(0)
                tot_acc += (logits.argmax(dim=1) == y).float().sum().item()
                n += x.size(0)
        return tot_loss / max(1, n), tot_acc / max(1, n)

    def run_round(self, round_idx: int, test_loader: DataLoader):
        global_state = self.broadcast()
        import numpy as np
        inv_lat = 1.0 / np.maximum(1e-6, self.selector.latency)
        st = [ float(self.selector.trust.mean()), float(inv_lat.mean()), float(round_idx / max(1, self.cfg.rounds - 1)) ]
        action = self.agent.act(st, epsilon=self.cfg.epsilon)
        K = max(1, min(len(self.clients), action + 1))
        sel = self.selector.select(K, mask_available=list(self.clients.keys()))
        deltas, losses, latencies, weights = [], [], [], []
        for cid in sel:
            cl = self.clients[cid]
            cl.set_global(global_state)
            delta, loss, latency = cl.train_local(epochs=self.cfg.local_epochs, lr=self.cfg.lr, momentum=self.cfg.momentum)
            deltas.append(delta); losses.append(loss); latencies.append(latency); weights.append(len(cl.loader.dataset))
        flat_updates = []
        for d in deltas:
            import torch
            vec = torch.cat([p.flatten() for p in d.values()]).numpy()
            flat_updates.append(vec)
        X = np.vstack(flat_updates)
        self.pca.partial_fit(X)
        Xc = self.pca.transform(X)
        self.detector.fit(Xc)
        flags, scores = self.detector.predict(Xc)
        num_anomalies = int(flags.sum())
        keep = [i for i, f in enumerate(flags.tolist()) if f == 0]
        if not keep:
            keep = list(range(len(deltas)))
        deltas_agg = [deltas[i] for i in keep]
        weights_agg = [weights[i] for i in keep]
        self.aggregate(deltas_agg, weights_agg)
        _, acc_after = self.evaluate(test_loader)
        delta_acc = acc_after - (self.history["acc"][-1] if self.history["acc"] else 0.0)
        cost = float(sum(v.numel() for v in self.global_model.state_dict().values())) / 1e6
        latency_mean = float(np.mean(latencies)) if latencies else 0.0
        reward = self.rewarder(delta_acc=delta_acc, cost=cost, num_anomalies=num_anomalies, latency=latency_mean)
        st2 = [ float(self.selector.trust.mean()), float((1.0 / np.maximum(1e-6, self.selector.latency)).mean()), float((round_idx + 1) / max(1, self.cfg.rounds - 1)) ]
        self.agent.remember(st, action, reward, st2, 0.0)
        loss_rl = self.agent.update()
        q_per = {cid: float(1.0 / (1.0 + l)) for cid, l in zip(sel, losses)}
        self.selector.update_trust(q_per)
        lat_per = {cid: float(lat) for cid, lat in zip(sel, latencies)}
        self.selector.update_latency(lat_per)
        perf_per = {cid: float(delta_acc / max(1, len(sel))) for cid in sel}
        self.selector.update_hist_perf(perf_per)
        self.history["acc"].append(acc_after)
        self.history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        self.history["reward"].append(float(reward))
        self.history["latency"].append(latency_mean)
        self.history["selected"].append(sel)
        return { "round": round_idx, "selected": sel, "acc": acc_after, "reward": reward, "rl_loss": float(loss_rl) if loss_rl is not None else 0.0, "anomalies": num_anomalies, "lat": latency_mean, "K": K }
