from typing import Tuple, Deque, List
from dataclasses import dataclass
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 50_000
    target_sync: int = 250  # steps
    device: str = "cpu"

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple] = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )
    def __len__(self): return len(self.buf)

class RLAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.q = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden).to(cfg.device)
        self.tgt = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden).to(cfg.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.steps = 0

    def act(self, state: List[float], epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.q(s)
            return int(q.argmax(dim=1).item())

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return 0.0
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        s = s.to(self.cfg.device); a = a.to(self.cfg.device)
        r = r.to(self.cfg.device); s2 = s2.to(self.cfg.device)
        d = d.to(self.cfg.device)
        # Q(s,a)
        q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
        # Double DQN: a* = argmax_a Q(s2,a; online)
        with torch.no_grad():
            a_star = self.q(s2).argmax(dim=1, keepdim=True)
            q_tgt = self.tgt(s2).gather(1, a_star).squeeze(1)
            target = r + self.cfg.gamma * (1.0 - d) * q_tgt
        loss = torch.nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.steps += 1
        if self.steps % self.cfg.target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def remember(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)
