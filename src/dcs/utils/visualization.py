import os
from typing import List, Dict
import matplotlib.pyplot as plt

def plot_series(series: Dict[str, List[float]], title: str, outdir: str, fname: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for k, v in series.items():
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
    path = os.path.join(outdir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
