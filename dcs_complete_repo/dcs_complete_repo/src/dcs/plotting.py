from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_compare_by_round(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    label_a: str,
    label_b: str,
    step: int = 1,
    title: str | None = None,
    ylabel: str | None = None,
    save_path: str | None = None,
) -> None:
    da = df_a[["round", metric]].copy()
    db = df_b[["round", metric]].copy()
    da = da[da["round"] % step == 0]
    db = db[db["round"] % step == 0]

    rounds = da["round"].values
    x = np.arange(len(rounds))
    width = 0.42

    plt.figure(figsize=(12, 4))
    plt.bar(x - width / 2, da[metric].values, width=width, label=label_a)
    plt.bar(x + width / 2, db[metric].values, width=width, label=label_b)
    plt.xticks(x, rounds, rotation=0)
    plt.xlabel("Round")
    plt.ylabel(ylabel if ylabel else metric)
    plt.title(title if title else f"{metric} by round (bar)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


def bar_single_series(
    x_labels: List[str],
    y_values: List[float],
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(10, 3))
    plt.bar([str(x) for x in x_labels], y_values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


def summary_table(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "final_val": float(df["val_acc"].iloc[-1]),
        "final_test": float(df["test_acc"].iloc[-1]),
        "peak_test": float(df["test_acc"].max()),
        "avg_latency": float(df["avg_lat"].mean()),
        "total_comm_mb": float(df["comm_mb"].sum()),
        "total_anom": int(df["anom"].sum()) if "anom" in df.columns else 0,
        "avg_fairness": float(df["fairness"].mean()),
    }


def bar_summary_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str, out_dir: str | None = None) -> pd.DataFrame:
    s = pd.DataFrame([summary_table(df_a, name_a), summary_table(df_b, name_b)])

    metrics = ["final_val", "final_test", "peak_test", "avg_latency", "total_comm_mb", "total_anom", "avg_fairness"]
    for m in metrics:
        plt.figure(figsize=(6, 3))
        plt.bar(s["name"], s[m].values)
        plt.title(f"Summary: {m}")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(str(Path(out_dir) / f"summary_{m}.png"), dpi=200)
        plt.show()

    return s


def bar_sensitivity(
    df: pd.DataFrame,
    xcol: str,
    acc_col: str = "final_test",
    lat_col: str = "avg_latency",
    title_prefix: str = "Sensitivity",
    out_dir: str | None = None,
) -> None:
    """Two bar charts: accuracy and latency across hyperparameter values."""
    x = [str(v) for v in df[xcol].tolist()]

    bar_single_series(
        x_labels=x,
        y_values=df[acc_col].tolist(),
        title=f"{title_prefix} – Final Test Accuracy (bar)",
        ylabel="Accuracy",
        save_path=(str(Path(out_dir) / f"sens_{xcol}_acc.png") if out_dir else None),
    )

    bar_single_series(
        x_labels=x,
        y_values=df[lat_col].tolist(),
        title=f"{title_prefix} – Avg Latency (bar)",
        ylabel="Seconds",
        save_path=(str(Path(out_dir) / f"sens_{xcol}_lat.png") if out_dir else None),
    )
