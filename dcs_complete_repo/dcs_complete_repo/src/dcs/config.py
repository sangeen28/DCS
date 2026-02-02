'''

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class Config:
    # Dataset
    DATASET: str = "mnist"  # mnist | fashion_mnist | cifar10

    # Federation
    NUM_CLIENTS: int = 50
    NUM_EDGES: int = 5
    ROUNDS: int = 20
    LOCAL_EPOCHS: int = 1
    BATCH_SIZE: int = 64
    LR: float = 0.05
    MOMENTUM: float = 0.9

    # Data non-IID
    DIRICHLET_ALPHA: float = 0.8
    MIN_SAMPLES_PER_CLIENT: int = 20

    # Selection action space
    K_MIN: int = 5
    K_MAX: int = 15
    K_STEP: int = 2
    LAM_GRID: Tuple[float, ...] = (0.3, 0.5, 0.7)
    LAM_DEFAULT: float = 0.5

    # Trust / latency
    TRUST_ALPHA: float = 0.6
    LAT_EMA: float = 0.7

    # Malicious simulation
    MALICIOUS_RATIO_SELECTED: float = 0.0
    FLIP_PAIR: Tuple[int, int] = (0, 1)

    # Stable projection + ANN
    PCA_RANK: int = 20
    ANN_NEIGHBORS: int = 6
    CONTAMINATION: float = 0.10
    MAX_REF: int = 2000

    # DDQL
    DDQL_HIDDEN: int = 128
    DDQL_LR: float = 1e-3
    DDQL_GAMMA: float = 0.95
    DDQL_TAU: float = 0.01
    DDQL_EPS_START: float = 1.0
    DDQL_EPS_END: float = 0.05
    DDQL_EPS_DECAY: float = 0.995
    DDQL_BUFFER: int = 20000
    DDQL_BATCH: int = 128

    # Reward weights
    W_PERF: float = 1.0
    W_COMP: float = 0.10
    W_COMM: float = 0.05
    W_ANOM: float = 0.50
    W_LAT: float = 0.10
    W_ENERGY: float = 0.02
    W_FAIR: float = 0.20

    # Fast mode
    MAX_TRAIN_SAMPLES: int | None = None
    MAX_TEST_SAMPLES: int | None = None

    # Local quality proxy
    MAX_QUALITY_BATCHES: int = 2

    def projection_dim(self) -> int:
        """Fixed projection dimension used by the anomaly filter.

        Key stability rule:
        - IncrementalPCA requires n_samples >= n_components each partial_fit.
        - In FL, per-round updates can be as low as ~K_MIN (or lower with dropouts).

        To avoid shape mismatch across rounds, we use a fixed dimension:
            proj_dim = min(PCA_RANK, max(2, K_MIN))
        """
        return int(min(self.PCA_RANK, max(2, self.K_MIN)))


def _coerce_tuple(v: Any) -> Tuple[float, ...]:
    if isinstance(v, tuple):
        return tuple(float(x) for x in v)
    if isinstance(v, list):
        return tuple(float(x) for x in v)
    raise TypeError(f"Expected list/tuple, got {type(v)}")


def _coerce_pair(v: Any) -> Tuple[int, int]:
    if isinstance(v, tuple) and len(v) == 2:
        return int(v[0]), int(v[1])
    if isinstance(v, list) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise TypeError("FLIP_PAIR must be a length-2 list/tuple")


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw: Dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        if k == "LAM_GRID":
            setattr(cfg, k, _coerce_tuple(v))
        elif k == "FLIP_PAIR":
            setattr(cfg, k, _coerce_pair(v))
        else:
            setattr(cfg, k, v)

    # sanity
    if cfg.NUM_EDGES <= 0:
        raise ValueError("NUM_EDGES must be >= 1")
    if cfg.NUM_CLIENTS <= 0:
        raise ValueError("NUM_CLIENTS must be >= 1")
    if cfg.K_MIN <= 0 or cfg.K_MAX <= 0 or cfg.K_MIN > cfg.K_MAX:
        raise ValueError("Invalid K_MIN/K_MAX")
    if cfg.K_STEP <= 0:
        raise ValueError("K_STEP must be >= 1")
    if not (0.0 <= cfg.MALICIOUS_RATIO_SELECTED <= 1.0):
        raise ValueError("MALICIOUS_RATIO_SELECTED must be in [0,1]")
    if not (0.0 < cfg.CONTAMINATION < 1.0):
        raise ValueError("CONTAMINATION must be in (0,1)")
    if cfg.ANN_NEIGHBORS < 1:
        raise ValueError("ANN_NEIGHBORS must be >= 1")
    if cfg.PCA_RANK < 2:
        raise ValueError("PCA_RANK must be >= 2")

    return cfg


def to_dict(cfg: Config) -> Dict[str, Any]:
    d = asdict(cfg)
    d["LAM_GRID"] = list(cfg.LAM_GRID)
    d["FLIP_PAIR"] = list(cfg.FLIP_PAIR)
    return d

    '''

#writefile src/dcs/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class Config:
    # Dataset
    DATASET: str = "mnist"  # mnist | fashion_mnist | cifar10

    # Federation
    NUM_CLIENTS: int = 50
    NUM_EDGES: int = 5
    ROUNDS: int = 20
    LOCAL_EPOCHS: int = 1
    BATCH_SIZE: int = 64
    LR: float = 0.05
    MOMENTUM: float = 0.9

    # Data non-IID
    DIRICHLET_ALPHA: float = 0.8
    MIN_SAMPLES_PER_CLIENT: int = 20

    # Selection action space
    K_MIN: int = 5
    K_MAX: int = 15
    K_STEP: int = 2
    LAM_GRID: Tuple[float, ...] = (0.3, 0.5, 0.7)
    LAM_DEFAULT: float = 0.5

    # Trust / latency
    TRUST_ALPHA: float = 0.6
    LAT_EMA: float = 0.7

    # Malicious simulation
    MALICIOUS_RATIO_SELECTED: float = 0.0
    FLIP_PAIR: Tuple[int, int] = (0, 1)

    # Stable projection + ANN
    PCA_RANK: int = 20
    ANN_NEIGHBORS: int = 6
    CONTAMINATION: float = 0.10
    MAX_REF: int = 2000

    # DDQL
    DDQL_HIDDEN: int = 128
    DDQL_LR: float = 1e-3
    DDQL_GAMMA: float = 0.95
    DDQL_TAU: float = 0.01
    DDQL_EPS_START: float = 1.0
    DDQL_EPS_END: float = 0.05
    DDQL_EPS_DECAY: float = 0.995
    DDQL_BUFFER: int = 20000
    DDQL_BATCH: int = 128

    # Reward weights
    W_PERF: float = 1.0
    W_COMP: float = 0.10
    W_COMM: float = 0.05
    W_ANOM: float = 0.50
    W_LAT: float = 0.10
    W_ENERGY: float = 0.02
    W_FAIR: float = 0.20

    # Fast mode
    MAX_TRAIN_SAMPLES: int | None = None
    MAX_TEST_SAMPLES: int | None = None

    # Local quality proxy
    MAX_QUALITY_BATCHES: int = 2

    def projection_dim(self) -> int:
        """Fixed projection dimension used by the anomaly filter.

        Key stability rule:
        - IncrementalPCA requires n_samples >= n_components each partial_fit.
        - In FL, per-round updates can be as low as ~K_MIN (or lower with dropouts).

        To avoid shape mismatch across rounds, we use a fixed dimension:
            proj_dim = min(PCA_RANK, max(2, K_MIN))
        """
        return int(min(self.PCA_RANK, max(2, self.K_MIN)))

    # ✅ Added: runner.py expects this helper
    def flip_pair_tuple(self) -> Tuple[int, int]:
        return int(self.FLIP_PAIR[0]), int(self.FLIP_PAIR[1])


def _coerce_tuple(v: Any) -> Tuple[float, ...]:
    if isinstance(v, tuple):
        return tuple(float(x) for x in v)
    if isinstance(v, list):
        return tuple(float(x) for x in v)
    raise TypeError(f"Expected list/tuple, got {type(v)}")


def _coerce_pair(v: Any) -> Tuple[int, int]:
    if isinstance(v, tuple) and len(v) == 2:
        return int(v[0]), int(v[1])
    if isinstance(v, list) and len(v) == 2:
        return int(v[0]), int(v[1])
    raise TypeError("FLIP_PAIR must be a length-2 list/tuple")


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw: Dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Config()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        if k == "LAM_GRID":
            setattr(cfg, k, _coerce_tuple(v))
        elif k == "FLIP_PAIR":
            setattr(cfg, k, _coerce_pair(v))
        else:
            setattr(cfg, k, v)

    # sanity
    if cfg.NUM_EDGES <= 0:
        raise ValueError("NUM_EDGES must be >= 1")
    if cfg.NUM_CLIENTS <= 0:
        raise ValueError("NUM_CLIENTS must be >= 1")
    if cfg.K_MIN <= 0 or cfg.K_MAX <= 0 or cfg.K_MIN > cfg.K_MAX:
        raise ValueError("Invalid K_MIN/K_MAX")
    if cfg.K_STEP <= 0:
        raise ValueError("K_STEP must be >= 1")
    if not (0.0 <= cfg.MALICIOUS_RATIO_SELECTED <= 1.0):
        raise ValueError("MALICIOUS_RATIO_SELECTED must be in [0,1]")
    if not (0.0 < cfg.CONTAMINATION < 1.0):
        raise ValueError("CONTAMINATION must be in (0,1)")
    if cfg.ANN_NEIGHBORS < 1:
        raise ValueError("ANN_NEIGHBORS must be >= 1")
    if cfg.PCA_RANK < 2:
        raise ValueError("PCA_RANK must be >= 2")

    return cfg


# ✅ Restored: __init__.py imports this
def to_dict(cfg: Config) -> Dict[str, Any]:
    d = asdict(cfg)
    d["LAM_GRID"] = list(cfg.LAM_GRID)
    d["FLIP_PAIR"] = list(cfg.FLIP_PAIR)
    return d
