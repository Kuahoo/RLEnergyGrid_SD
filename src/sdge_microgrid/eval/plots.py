from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory_72h(
    infos: list[dict],
    out_path: Path | str,
    *,
    pv: np.ndarray | None = None,
    title: str = "72-hour dispatch",
) -> None:
    n = min(72, len(infos))
    t = np.arange(n)
    soc = np.array([infos[i]["soc"] for i in range(n)])
    ushed = np.array([infos[i]["u_shed"] for i in range(n)])
    g = np.array([infos[i]["g"] for i in range(n)])
    if pv is None:
        pv = np.array([infos[i]["p_pv"] for i in range(n)])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, soc)
    axes[0].set_ylabel("SoC")
    axes[1].plot(t, pv[:n])
    axes[1].set_ylabel("PV (kW)")
    axes[2].plot(t, ushed, label="u_shed")
    axes[2].plot(t, g, label="grid available", linestyle="--")
    axes[2].set_ylabel("shed / g")
    axes[2].set_xlabel("Hour")
    axes[2].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learning_curves(results_dir: Path | str, out_path: Path | str) -> None:
    results_dir = Path(results_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for algo, color in (("ppo", "C0"), ("sac", "C1")):
        curves = []
        timesteps = None
        for seed_dir in sorted((results_dir / algo).glob("seed_*")):
            ev = seed_dir / "evaluations.npz"
            if not ev.exists():
                continue
            data = np.load(ev)
            curves.append(data["results"].mean(axis=1))
            timesteps = data["timesteps"]
        if not curves:
            continue
        arr = np.vstack(curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(timesteps, mean, color=color, label=algo.upper())
        ax.fill_between(timesteps, mean - std, mean + std, color=color, alpha=0.2)
        plotted = True
    if plotted:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Eval return")
        ax.legend()
        ax.set_title("Learning curves (mean ± 1 std)")
    else:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No evaluations.npz yet; run full training first",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summarize_comparison_csv(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return (
        df.groupby(["controller", "scenario"])[
            [
                "critical_survival_rate",
                "weekly_net_economic_cost",
                "flexible_curtailment_ratio",
            ]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
