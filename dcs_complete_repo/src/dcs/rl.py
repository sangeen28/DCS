from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNConfig:
    state_dim: int = 12
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    buffer_size: int = 20000
    batch_size: int = 128


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=int(capacity))

    def push(self, s, a, r, sp, done):
        self.buf.append((s, int(a), float(r), sp, float(done)))

    def sample(self, n: int):
        if len(self.buf) < n:
            return None
        batch = random.sample(self.buf, n)
        s, a, r, sp, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(sp), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQN:
    def __init__(self, cfg: DQNConfig, action_dim: int, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.q = QNet(cfg.state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.qt = QNet(cfg.state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.qt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.mem = ReplayBuffer(cfg.buffer_size)

        self.eps = cfg.eps_start
        self.action_dim = action_dim

    def act(self, s: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.q(st).argmax(dim=1).item())

    def update_eps(self) -> None:
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def soft_update(self) -> None:
        for tp, p in zip(self.qt.parameters(), self.q.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tp.data)

    def train_step(self) -> float:
        batch = self.mem.sample(self.cfg.batch_size)
        if batch is None:
            return 0.0
        s, a, r, sp, d = batch
        s, a, r, sp, d = s.to(self.device), a.to(self.device), r.to(self.device), sp.to(self.device), d.to(self.device)

        qsa = self.q(s).gather(1, a.unsqueeze(1))

        with torch.no_grad():
            ap = self.q(sp).argmax(dim=1, keepdim=True)
            qtp = self.qt(sp).gather(1, ap)
            y = r.unsqueeze(1) + (1.0 - d.unsqueeze(1)) * self.cfg.gamma * qtp

        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self.soft_update()
        return float(loss.item())
